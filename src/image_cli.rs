use super::{print_search_results, validate_semantic_search, DB};
use anyhow::Result;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::File;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

fn print_embedding(embedding: &candle_core::Tensor) -> Result<()> {
    let embedding = embedding.squeeze(0)?.to_device(&candle_core::Device::Cpu)?;
    let (tokens, _) = embedding.dims2()?;
    if tokens == 1 {
        println!("{}", serde_json::to_string(&embedding.squeeze(0)?.to_vec1::<f32>()?)?);
    } else {
        println!("{}", serde_json::to_string(&embedding.to_vec2::<f32>()?)?);
    }
    Ok(())
}

#[cfg(not(any(feature = "modernbert", feature = "modernbert-quantized")))]
fn cosine_similarity(lhs: &candle_core::Tensor, rhs: &candle_core::Tensor) -> Result<f32> {
    Ok((lhs * rhs)?.sum_all()?.to_scalar::<f32>()?)
}

#[cfg(any(feature = "modernbert", feature = "modernbert-quantized"))]
fn maxsim_similarity(query: &candle_core::Tensor, image: &candle_core::Tensor) -> Result<f32> {
    let image = image.to_device(query.device())?;
    let similarities = query.matmul(&image.transpose(0, 1)?)?;
    Ok(similarities.max(1)?.mean_all()?.to_scalar::<f32>()?)
}

fn bench_count(value: Option<&str>, env_name: &str, default: usize) -> Result<usize> {
    value
        .map(str::to_owned)
        .or_else(|| env::var(env_name).ok())
        .map_or(Ok(default), |value| {
            value
                .parse::<usize>()
                .map_err(|e| anyhow::anyhow!("invalid {env_name} value {value:?}: {e}"))
        })
}

fn benchmark_image_embeddings(embedder: &witchcraft::Embedder, path: &Path, args: &[String]) -> Result<()> {
    let profile = args.iter().any(|arg| matches!(arg.as_str(), "profile" | "--profile"));
    let counts = args
        .iter()
        .filter(|arg| !matches!(arg.as_str(), "profile" | "--profile"))
        .map(String::as_str)
        .collect::<Vec<_>>();
    anyhow::ensure!(counts.len() <= 2, "benchimage expects [iters] [warmup] [profile]");
    let iters = bench_count(counts.first().copied(), "WARP_BENCH_ITERS", 25)?;
    let warmup = bench_count(counts.get(1).copied(), "WARP_BENCH_WARMUP", 5)?;
    let batch_size = bench_count(None, "WARP_BENCH_BATCH", 1)?;
    anyhow::ensure!(iters > 0, "image benchmark iterations must be greater than zero");
    anyhow::ensure!(batch_size > 0, "image benchmark batch size must be greater than zero");

    let image = witchcraft::mobileclip2::load_image(path, embedder.image_size())?;
    let images = vec![image; batch_size];
    let image = candle_core::Tensor::stack(&images, 0)?.to_device(embedder.device())?;
    let run = || -> Result<candle_core::Tensor> {
        let embedding = embedder.embed_image_tensor(&image)?;
        let checksum = embedding
            .sum_all()?
            .to_device(&candle_core::Device::Cpu)?
            .to_scalar::<f32>()?;
        std::hint::black_box(checksum);
        Ok(embedding)
    };
    for _ in 0..warmup {
        run()?;
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let started = Instant::now();
        run()?;
        times.push(started.elapsed());
    }
    times.sort_unstable();
    let mean = times.iter().map(Duration::as_secs_f64).sum::<f64>() / iters as f64;
    let p95 = ((iters * 95 + 99) / 100).saturating_sub(1).min(iters - 1);
    println!("image: {}", path.display());
    println!("device: {:?}", embedder.device().location());
    println!("image_size: {}x{}", embedder.image_size(), embedder.image_size());
    println!("embedding_shape: {:?}", run()?.dims());
    println!("batch_size: {batch_size} iters: {iters} warmup: {warmup}");
    println!(
        "model_only: median={:.2} ms mean={:.2} ms p95={:.2} ms throughput={:.2} images/s",
        times[iters / 2].as_secs_f64() * 1000.0,
        mean * 1000.0,
        times[p95].as_secs_f64() * 1000.0,
        batch_size as f64 / mean
    );

    if profile {
        let (embedding, profile) = embedder.embed_image_tensor_profiled(&image)?;
        embedding
            .sum_all()?
            .to_device(&candle_core::Device::Cpu)?
            .to_scalar::<f32>()?;
        println!("profile_total_ms: {:.2}", profile.total().as_secs_f64() * 1000.0);
        let mut records = profile.records().iter().collect::<Vec<_>>();
        records.sort_unstable_by(|a, b| b.elapsed.cmp(&a.elapsed));
        for record in records.into_iter().take(30) {
            println!(
                "  {:>8.2} ms {:<56} {:?}",
                record.elapsed.as_secs_f64() * 1000.0,
                record.label,
                record.shape
            );
        }
    }
    Ok(())
}

fn benchmark_pooled_image_embeddings(
    model: &witchcraft::mobileclip2::StockMobileClip2,
    path: &Path,
    args: &[String],
) -> Result<()> {
    anyhow::ensure!(args.len() <= 2, "benchclipimage expects [iters] [warmup]");
    let iters = bench_count(args.first().map(String::as_str), "WARP_BENCH_ITERS", 25)?;
    let warmup = bench_count(args.get(1).map(String::as_str), "WARP_BENCH_WARMUP", 5)?;
    anyhow::ensure!(iters > 0, "image benchmark iterations must be greater than zero");

    let image = witchcraft::mobileclip2::load_image(path, model.image_size())?;
    let image = image.unsqueeze(0)?.to_device(model.device())?;
    let run = || -> Result<candle_core::Tensor> {
        let embedding = model.embed_images(&image)?;
        let checksum = embedding
            .sum_all()?
            .to_device(&candle_core::Device::Cpu)?
            .to_scalar::<f32>()?;
        std::hint::black_box(checksum);
        Ok(embedding)
    };
    for _ in 0..warmup {
        run()?;
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let started = Instant::now();
        run()?;
        times.push(started.elapsed());
    }
    times.sort_unstable();
    let mean = times.iter().map(Duration::as_secs_f64).sum::<f64>() / iters as f64;
    let p95 = ((iters * 95 + 99) / 100).saturating_sub(1).min(iters - 1);
    println!("image: {}", path.display());
    println!("device: {:?}", model.device().location());
    println!("image_size: {}x{}", model.image_size(), model.image_size());
    println!("embedding_shape: {:?}", run()?.dims());
    println!("batch_size: 1 iters: {iters} warmup: {warmup}");
    println!(
        "model_only: median={:.2} ms mean={:.2} ms p95={:.2} ms throughput={:.2} images/s",
        times[iters / 2].as_secs_f64() * 1000.0,
        mean * 1000.0,
        times[p95].as_secs_f64() * 1000.0,
        1.0 / mean
    );
    Ok(())
}

fn collect_image_paths(directory: &Path, paths: &mut Vec<PathBuf>) -> Result<()> {
    anyhow::ensure!(directory.is_dir(), "{} is not a directory", directory.display());
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let path = entry.path();
        if file_type.is_dir() {
            collect_image_paths(&path, paths)?;
        } else if file_type.is_file() && super::is_supported_image_path(&path) {
            paths.push(path);
        }
    }
    Ok(())
}

fn image_paths_in_directories(directories: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for directory in directories {
        collect_image_paths(directory, &mut paths)?;
    }
    paths.sort();
    paths.dedup();
    Ok(paths)
}

fn html_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn file_url(path: &Path) -> String {
    let path = witchcraft::normalize_image_path(path);
    let path = path.to_string_lossy();
    let path = path.replace('\\', "/");
    let mut encoded = String::with_capacity(path.len());
    for byte in path.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~' | b'/' | b':') {
            encoded.push(char::from(byte));
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    if encoded.starts_with("//") {
        format!("file:{encoded}")
    } else if encoded.starts_with('/') {
        format!("file://{encoded}")
    } else {
        format!("file:///{encoded}")
    }
}

const THUMBNAIL_SIZE: u32 = 640;
const THUMBNAIL_WORKERS: usize = 4;

fn thumbnail_path_in(directory: &Path, path: &Path) -> PathBuf {
    let path = witchcraft::normalize_image_path(path);
    let hash = Sha256::digest(path.to_string_lossy().as_bytes());
    let name = hash.iter().map(|byte| format!("{byte:02x}")).collect::<String>();
    directory.join(format!("{name}.webp"))
}

fn thumbnail_is_fresh(source: &Path, thumbnail: &Path) -> bool {
    let Ok(thumbnail_metadata) = thumbnail.metadata() else {
        return false;
    };
    match (source.metadata(), thumbnail_metadata.modified()) {
        (Ok(source_metadata), Ok(thumbnail_modified)) => source_metadata
            .modified()
            .map(|source_modified| thumbnail_modified >= source_modified)
            .unwrap_or(true),
        _ => true,
    }
}

fn ensure_thumbnail(path: &Path) -> Result<PathBuf> {
    ensure_thumbnail_in(&env::current_dir()?.join("thumbnails"), path)
}

fn ensure_thumbnail_in(directory: &Path, path: &Path) -> Result<PathBuf> {
    let thumbnail = thumbnail_path_in(directory, path);
    if thumbnail_is_fresh(path, &thumbnail) {
        return Ok(thumbnail);
    }
    std::fs::create_dir_all(directory)?;
    let image = witchcraft::mobileclip2::decode_image_oriented(path)?;
    let image = image.thumbnail(THUMBNAIL_SIZE, THUMBNAIL_SIZE).to_rgb8();
    let output = File::create(&thumbnail)?;
    image::codecs::webp::WebPEncoder::new_lossless(output).encode(
        image.as_raw(),
        image.width(),
        image.height(),
        image::ExtendedColorType::Rgb8,
    )?;
    Ok(thumbnail)
}

fn generate_thumbnails(
    paths: &[PathBuf],
    progress: Option<&witchcraft::progress::Bar>,
) -> (HashMap<PathBuf, PathBuf>, usize) {
    let mut seen = HashSet::new();
    let unique = paths
        .iter()
        .filter(|path| seen.insert((*path).clone()))
        .cloned()
        .collect::<Vec<_>>();
    let mut generated = HashMap::new();
    let mut failed = 0usize;
    for chunk in unique.chunks(THUMBNAIL_WORKERS) {
        let results = std::thread::scope(|scope| {
            chunk
                .iter()
                .map(|path| {
                    let path = path.clone();
                    scope.spawn(move || {
                        let thumbnail = ensure_thumbnail(&path);
                        (path, thumbnail)
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|worker| worker.join().expect("thumbnail worker panicked"))
                .collect::<Vec<_>>()
        });
        for (source, thumbnail) in results {
            match thumbnail {
                Ok(thumbnail) => {
                    generated.insert(source, thumbnail);
                },
                Err(error) => {
                    failed += 1;
                    eprintln!("[WARN] could not create thumbnail for {}: {error}", source.display());
                },
            }
            if let Some(progress) = progress {
                progress.inc(1);
            }
        }
    }
    (generated, failed)
}

fn thumbnail_sources(paths: &[PathBuf]) -> Vec<PathBuf> {
    let (generated, _) = generate_thumbnails(paths, None);
    paths
        .iter()
        .map(|path| generated.get(path).cloned().unwrap_or_else(|| path.clone()))
        .collect()
}

fn create_registered_thumbnails(paths: &[PathBuf]) -> Result<()> {
    let directory = env::current_dir()?.join("thumbnails");
    let mut seen = HashSet::new();
    let unique = paths
        .iter()
        .filter(|path| seen.insert((*path).clone()))
        .cloned()
        .collect::<Vec<_>>();
    let pending = unique
        .iter()
        .filter(|path| !thumbnail_is_fresh(path, &thumbnail_path_in(&directory, path)))
        .cloned()
        .collect::<Vec<_>>();
    let skipped = unique.len() - pending.len();
    if pending.is_empty() {
        println!("all {skipped} registered images already have fresh WebP thumbnails");
        return Ok(());
    }
    let progress = witchcraft::progress::new_with_label(pending.len() as u64, "create thumbnails");
    let (generated, failed) = generate_thumbnails(&pending, Some(&progress));
    progress.finish();
    println!(
        "created {} WebP thumbnails, skipped {skipped} fresh, {failed} failed in {}",
        generated.len(),
        directory.display()
    );
    Ok(())
}

fn image_grid_html_with_links(
    paths: &[PathBuf],
    sources: Option<&[PathBuf]>,
    links: Option<&[String]>,
) -> String {
    let mut tiles = String::new();
    for (index, path) in paths.iter().take(9).enumerate() {
        let url = file_url(path);
        let source_url = sources
            .and_then(|sources| sources.get(index))
            .map(|source| file_url(source))
            .unwrap_or_else(|| url.clone());
        let link = links
            .and_then(|links| links.get(index))
            .map(String::as_str)
            .unwrap_or(&url);
        let label = html_escape(&path.to_string_lossy());
        tiles.push_str(&format!(
            "<figure><a href=\"{link}\"><img src=\"{source_url}\" alt=\"{label}\"></a><figcaption>{label}</figcaption></figure>\n"
        ));
    }
    format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Witchcraft image results</title><style>\
         html,body{{margin:0;min-height:100%;background:#111;color:#ddd;font-family:system-ui,sans-serif}}\
         main{{box-sizing:border-box;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;padding:12px;height:100vh}}\
         figure{{min-width:0;min-height:0;margin:0;display:flex;flex-direction:column;background:#1b1b1b;border-radius:5px;overflow:hidden}}\
         a{{display:flex;min-height:0;flex:1}}img{{display:block;width:100%;height:100%;object-fit:contain}}\
         figcaption{{padding:5px 7px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:12px}}\
         </style></head><body><main>{tiles}</main></body></html>"
    )
}

fn image_cluster_html(
    clusters: &[(String, Vec<PathBuf>)],
    thumbnails: &HashMap<PathBuf, PathBuf>,
) -> String {
    let mut sections = String::new();
    for (label, paths) in clusters {
        let mut tiles = String::new();
        for (index, path) in paths.iter().enumerate() {
            let source = thumbnails.get(path).unwrap_or(path);
            let url = file_url(path);
            let source_url = file_url(source);
            let escaped_path = html_escape(&path.to_string_lossy());
            let class = if index == 0 { " class=\"winner\"" } else { "" };
            let badge = if index == 0 { "<b>MaxSim</b>" } else { "" };
            tiles.push_str(&format!(
                "<figure{class}><a href=\"{url}\"><img loading=\"lazy\" src=\"{source_url}\" alt=\"{escaped_path}\"></a><figcaption>{badge}{escaped_path}</figcaption></figure>\n"
            ));
        }
        sections.push_str(&format!(
            "<section><h2>{} <small>({} images)</small></h2><div class=\"cluster\">{tiles}</div></section>\n",
            html_escape(label),
            paths.len(),
        ));
    }
    format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Witchcraft image clusters</title><style>\
         html,body{{margin:0;min-height:100%;background:#111;color:#ddd;font-family:system-ui,sans-serif}}\
         body{{padding:16px}}section{{margin:0 0 28px}}h2{{font-size:18px;margin:0 0 10px}}\
         h2 small{{color:#999;font-weight:normal}}.cluster{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px}}\
         figure{{min-width:0;margin:0;background:#1b1b1b;border:2px solid transparent;border-radius:5px;overflow:hidden}}\
         figure.winner{{border-color:#66c2ff}}a{{display:block;height:170px}}img{{display:block;width:100%;height:100%;object-fit:contain}}\
         figcaption{{padding:5px 7px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:11px}}\
         figcaption b{{color:#66c2ff;margin-right:7px}}\
         </style></head><body>{sections}</body></html>"
    )
}

#[cfg(test)]
fn image_grid_html(paths: &[PathBuf]) -> String {
    image_grid_html_with_links(paths, None, None)
}

fn open_image_grid(path: &Path) -> std::io::Result<()> {
    #[cfg(target_os = "macos")]
    let mut command = {
        let mut command = Command::new("open");
        command.arg(path);
        command
    };
    #[cfg(target_os = "windows")]
    let mut command = {
        let mut command = Command::new("cmd");
        command.args(["/C", "start", ""]);
        command.arg(path);
        command
    };
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let mut command = {
        let mut command = Command::new("xdg-open");
        command.arg(path);
        command
    };
    command.spawn().map(|_| ())
}

pub fn display_image_grid(paths: &[PathBuf]) {
    if paths.is_empty() || !std::io::stdout().is_terminal() {
        return;
    }
    let sources = thumbnail_sources(paths);
    let preview = env::temp_dir().join("witchcraft-image-preview.html");
    if let Err(error) = std::fs::write(
        &preview,
        image_grid_html_with_links(paths, Some(&sources), None),
    ) {
        eprintln!("warning: could not write image preview: {error}");
        return;
    }
    if let Err(error) = open_image_grid(&preview) {
        eprintln!("warning: could not open image preview: {error}");
    }
}

pub fn display_image_grid_with_neighbors(paths: &[PathBuf], neighbors: &[Vec<PathBuf>]) {
    if paths.is_empty() || !std::io::stdout().is_terminal() {
        return;
    }
    let pages = env::temp_dir().join("witchcraft-image-neighbors");
    if let Err(error) = std::fs::create_dir_all(&pages) {
        eprintln!("warning: could not create image neighbor pages: {error}");
        return;
    }
    let mut all_paths = paths.iter().take(9).cloned().collect::<Vec<_>>();
    for image_neighbors in neighbors.iter().take(paths.len().min(9)) {
        all_paths.extend(image_neighbors.iter().take(9).cloned());
    }
    let all_sources = thumbnail_sources(&all_paths);
    let main_count = paths.len().min(9);
    let main_sources = &all_sources[..main_count];
    let mut source_offset = main_count;
    let mut links = Vec::with_capacity(paths.len().min(9));
    for (index, image_neighbors) in neighbors.iter().take(paths.len().min(9)).enumerate() {
        let page = pages.join(format!("neighbors-{index}.html"));
        let count = image_neighbors.len().min(9);
        let sources = &all_sources[source_offset..source_offset + count];
        source_offset += count;
        if let Err(error) = std::fs::write(
            &page,
            image_grid_html_with_links(image_neighbors, Some(sources), None),
        ) {
            eprintln!("warning: could not write image neighbor page: {error}");
            return;
        }
        links.push(file_url(&page));
    }
    let preview = env::temp_dir().join("witchcraft-image-preview.html");
    if let Err(error) = std::fs::write(
        &preview,
        image_grid_html_with_links(paths, Some(main_sources), Some(&links)),
    ) {
        eprintln!("warning: could not write image preview: {error}");
        return;
    }
    if let Err(error) = open_image_grid(&preview) {
        eprintln!("warning: could not open image preview: {error}");
    }
}

fn cluster_preview_data(results: &[(f32, String, Vec<String>, u32, String)]) -> Vec<(String, Vec<PathBuf>)> {
    results
        .iter()
        .filter_map(|(score, metadata, bodies, sub_idx, _)| {
            if bodies.is_empty() {
                return None;
            }
            let winner = (*sub_idx as usize).min(bodies.len() - 1);
            let mut paths = bodies
                .iter()
                .map(|body| witchcraft::normalize_image_path(Path::new(body)))
                .collect::<Vec<_>>();
            paths.swap(0, winner);
            let key = serde_json::from_str::<serde_json::Value>(metadata)
                .ok()
                .and_then(|metadata| metadata.get("cluster")?.get("key")?.as_str().map(str::to_owned))
                .unwrap_or_else(|| "image cluster".to_owned());
            Some((format!("{score:.4} — {key}"), paths))
        })
        .collect()
}

fn display_image_clusters(clusters: &[(String, Vec<PathBuf>)]) {
    if clusters.is_empty() || !std::io::stdout().is_terminal() {
        return;
    }
    let paths = clusters
        .iter()
        .flat_map(|(_, paths)| paths.iter().cloned())
        .collect::<Vec<_>>();
    let (thumbnails, _) = generate_thumbnails(&paths, None);
    let preview = env::temp_dir().join("witchcraft-image-preview.html");
    if let Err(error) = std::fs::write(&preview, image_cluster_html(clusters, &thumbnails)) {
        eprintln!("warning: could not write image cluster preview: {error}");
        return;
    }
    if let Err(error) = open_image_grid(&preview) {
        eprintln!("warning: could not open image cluster preview: {error}");
    }
}

pub fn run(args: &[String], assets: &Path, db_name: &Path) -> Result<bool> {
    let Some(command) = args.get(1).map(String::as_str) else {
        return Ok(false);
    };
    match command {
        "imagetags" => {
            anyhow::ensure!(args.len() == 3, "imagetags expects an image path");
            let tags = witchcraft::image_tags::extract(Path::new(&args[2]), assets)?;
            println!("{}", serde_json::to_string_pretty(&tags)?);
        },
        "embedimage" => {
            anyhow::ensure!(args.len() == 3 || args.len() == 4, "embedimage expects [rowid] <path>");
            let device = witchcraft::make_device();
            let embedder = witchcraft::Embedder::new(&device, assets)?;
            let mut db = DB::new_fast(db_name.to_path_buf())?;
            let (rowid, count) = if args.len() == 3 {
                witchcraft::cache_image_embeddings_auto(&mut db, &embedder, Path::new(&args[2]))?
            } else {
                let rowid = args[2].parse::<u64>()?;
                let count = witchcraft::cache_image_embeddings(&db, &embedder, rowid, Path::new(&args[3]))?;
                (rowid, count)
            };
            println!("embedded {count} image vectors for rowid {rowid}");
        },
        "embedimages" => {
            anyhow::ensure!(
                args.len() >= 3,
                "embedimages expects [--force] <directory>..."
            );
            let (force, directories) = if args[2] == "--force" {
                anyhow::ensure!(args.len() >= 4, "embedimages expects a directory after --force");
                (true, &args[3..])
            } else {
                (false, &args[2..])
            };
            let directories = directories.iter().map(PathBuf::from).collect::<Vec<_>>();
            let paths = image_paths_in_directories(&directories)?;
            anyhow::ensure!(!paths.is_empty(), "no supported images found under the supplied directories");
            let device = witchcraft::make_device();
            let embedder = witchcraft::Embedder::new(&device, assets)?;
            let mut db = DB::new_fast(db_name.to_path_buf())?;
            let results = witchcraft::cache_image_embeddings_auto_many(
                &mut db,
                &embedder,
                &paths,
                force,
                |path, rowid, count| {
                    println!("rowid {rowid}: {count} vectors for {}", path.display());
                },
            )?;
            let vector_count = results.iter().map(|(_, count)| count).sum::<usize>();
            let skipped = paths.len() - results.len();
            println!(
                "embedded or enriched {} images as {vector_count} vectors, skipped {skipped}; run `{} index` to index them",
                results.len(),
                args[0]
            );
        },
        "clipregister" => {
            anyhow::ensure!(
                args.len() >= 3,
                "clipregister expects <directory>..."
            );
            let directories = args[2..].iter().map(PathBuf::from).collect::<Vec<_>>();
            let paths = image_paths_in_directories(&directories)?;
            anyhow::ensure!(!paths.is_empty(), "no supported images found under the supplied directories");
            super::mobileclip2_ivf::register(&paths)?;
        },
        "clipembed" => {
            anyhow::ensure!(args.len() == 2, "clipembed takes no arguments");
            super::mobileclip2_ivf::embed(assets)?;
        },
        "cliptag" => {
            anyhow::ensure!(args.len() == 2, "cliptag takes no arguments");
            super::mobileclip2_ivf::enrich_tags(assets)?;
        },
        "clipthumbnails" => {
            anyhow::ensure!(args.len() == 2, "clipthumbnails takes no arguments");
            let paths = super::mobileclip2_ivf::registered_paths()?;
            create_registered_thumbnails(&paths)?;
        },
        "clipindex" => {
            anyhow::ensure!(args.len() == 2, "clipindex takes no arguments");
            super::mobileclip2_ivf::update_index()?;
        },
        "clipquery" => {
            let display_neighbors = args[2..].iter().any(|argument| argument == "-D");
            let display_images = display_neighbors || args[2..].iter().any(|argument| argument == "-d");
            let exact = args[2..].iter().any(|argument| argument == "--exact");
            let clustered = args[2..].iter().any(|argument| argument == "--clusters");
            let mut scan_budget = None;
            let mut query_args = Vec::new();
            let mut arguments = args[2..].iter();
            while let Some(argument) = arguments.next() {
                match argument.as_str() {
                    "-d" | "-D" | "--exact" | "--clusters" => {},
                    "--scan-budget" => {
                        let value = arguments
                            .next()
                            .ok_or_else(|| anyhow::anyhow!("--scan-budget expects a positive integer"))?;
                        let value = value.parse::<usize>()?;
                        anyhow::ensure!(value > 0, "--scan-budget must be greater than zero");
                        scan_budget = Some(value);
                    },
                    _ => query_args.push(argument.clone()),
                }
            }
            anyhow::ensure!(
                !query_args.is_empty(),
                "clipquery expects [-d|-D] [--exact] [--clusters] [--scan-budget N] <text>..."
            );
            witchcraft::reset_bucket_io_counters();
            let results = super::mobileclip2_ivf::query(
                assets,
                &query_args.join(" "),
                10,
                exact,
                clustered,
                scan_budget,
            )?;
            witchcraft::log_bucket_io_counters();
            let clusters = clustered.then(|| cluster_preview_data(&results));
            let image_paths = print_search_results(results);
            if display_images && clustered {
                display_image_clusters(clusters.as_deref().unwrap_or_default());
            } else if display_neighbors {
                let neighbors = super::mobileclip2_ivf::nearest_neighbors(&image_paths, 9)?;
                display_image_grid_with_neighbors(&image_paths, &neighbors);
            } else if display_images {
                display_image_grid(&image_paths);
            }
        },
        "image" => {
            anyhow::ensure!(args.len() == 3, "image expects an image path");
            let device = witchcraft::make_device();
            let embedder = witchcraft::Embedder::new(&device, assets)?;
            print_embedding(&embedder.embed_image_path(&args[2])?)?;
        },
        "benchimage" => {
            anyhow::ensure!(args.len() >= 3, "benchimage expects an image path");
            let device = witchcraft::make_device();
            let started = Instant::now();
            let embedder = witchcraft::Embedder::new(&device, assets)?;
            println!("model_load_ms: {:.2}", started.elapsed().as_secs_f64() * 1000.0);
            benchmark_image_embeddings(&embedder, Path::new(&args[2]), &args[3..])?;
        },
        "benchclipimage" => {
            anyhow::ensure!(args.len() >= 3, "benchclipimage expects an image path");
            let fallback_device = witchcraft::make_device();
            let device = witchcraft::make_image_device(&fallback_device);
            let started = Instant::now();
            let model = witchcraft::mobileclip2::StockMobileClip2::load(&device, assets)?;
            println!("model_load_ms: {:.2}", started.elapsed().as_secs_f64() * 1000.0);
            benchmark_pooled_image_embeddings(&model, Path::new(&args[2]), &args[3..])?;
        },
        "scoreimage" => {
            anyhow::ensure!(args.len() >= 4, "scoreimage expects <path> <text>...");
            let device = witchcraft::make_device();
            let embedder = witchcraft::Embedder::new(&device, assets)?;
            let image = embedder.embed_image_path(&args[2])?.squeeze(0)?;
            for text in &args[3..] {
                #[cfg(any(feature = "modernbert", feature = "modernbert-quantized"))]
                {
                    let query = embedder.embed(text)?.0.squeeze(0)?;
                    println!("`{text}': score={}", maxsim_similarity(&query, &image)?);
                }
                #[cfg(not(any(feature = "modernbert", feature = "modernbert-quantized")))]
                {
                    let text_embedding = embedder.embed_text_pooled(text)?.squeeze(0)?.squeeze(0)?;
                    println!(
                        "`{text}': score={}",
                        cosine_similarity(&image.squeeze(0)?, &text_embedding)?
                    );
                }
            }
        },
        "queryimage" => {
            anyhow::ensure!(
                args.len() == 3 || (args.len() == 4 && args[2] == "-d"),
                "queryimage expects [-d] <path>"
            );
            let display_images = args[2] == "-d";
            let image_path = if display_images { &args[3] } else { &args[2] };
            let device = witchcraft::make_device();
            let embedder = witchcraft::Embedder::new(&device, assets)?;
            let db = DB::new_reader(db_name.to_path_buf())?;
            validate_semantic_search(&db, Some(&embedder))?;
            let query = embedder.embed_image_path(image_path)?.squeeze(0)?;
            witchcraft::reset_bucket_io_counters();
            let results = witchcraft::search_embeddings(&db, &query, None, 0.0, 10, None, false)?;
            let image_paths = print_search_results(results);
            witchcraft::log_bucket_io_counters();
            if display_images {
                display_image_grid(&image_paths);
            }
        },
        _ => return Ok(false),
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::{
        cluster_preview_data, ensure_thumbnail_in, file_url, image_cluster_html, image_grid_html,
        image_grid_html_with_links, image_paths_in_directories, thumbnail_path_in,
    };
    use std::collections::HashMap;
    use std::fs;
    use std::path::{Path, PathBuf};

    #[test]
    fn image_preview_encodes_paths_and_escapes_labels() {
        let path = PathBuf::from("/tmp/a photo & <copy>.jpg");
        let html = image_grid_html(&[path]);
        assert!(html.contains("file:///tmp/a%20photo%20%26%20%3Ccopy%3E.jpg"));
        assert!(html.contains("/tmp/a photo &amp; &lt;copy&gt;.jpg"));
    }

    #[test]
    fn image_preview_can_link_tiles_to_neighbor_pages() {
        let path = PathBuf::from("/tmp/photo.jpg");
        let html = image_grid_html_with_links(
            &[path],
            Some(&[PathBuf::from("/tmp/thumb.webp")]),
            Some(&["file:///tmp/neighbors-0.html".into()]),
        );
        assert!(html.contains("href=\"file:///tmp/neighbors-0.html\""));
        assert!(html.contains("src=\"file:///tmp/thumb.webp\""));
    }

    #[test]
    fn cluster_preview_puts_the_maxsim_winner_first() {
        let results = vec![(
            0.42,
            r#"{"cluster":{"key":"event:2020:01:02|DK|Region|Town"}}"#.to_owned(),
            vec!["/tmp/first.jpg".to_owned(), "/tmp/winner.jpg".to_owned()],
            1,
            String::new(),
        )];
        let clusters = cluster_preview_data(&results);
        assert_eq!(clusters[0].1[0], PathBuf::from("/tmp/winner.jpg"));

        let html = image_cluster_html(&clusters, &HashMap::new());
        assert!(html.contains("MaxSim"));
        assert!(html.contains("event:2020:01:02|DK|Region|Town"));
        assert!(html.contains("(2 images)"));
    }

    #[test]
    fn thumbnails_use_a_filename_hash_and_preserve_aspect_ratio() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("original name.jpg");
        let image = image::RgbImage::from_pixel(1280, 640, image::Rgb([20, 40, 60]));
        image.save(&source).unwrap();
        let thumbnail_dir = temporary.path().join("thumbnails");

        let expected = thumbnail_path_in(&thumbnail_dir, &source);
        let thumbnail = ensure_thumbnail_in(&thumbnail_dir, &source).unwrap();

        assert_eq!(thumbnail, expected);
        assert_eq!(thumbnail.extension().unwrap(), "webp");
        assert_eq!(thumbnail.file_stem().unwrap().len(), 64);
        assert_eq!(image::open(thumbnail).unwrap().to_rgb8().dimensions(), (640, 320));
    }

    #[test]
    fn file_urls_support_windows_paths() {
        assert_eq!(
            file_url(Path::new(r"C:\Photos\summer trip.jpg")),
            "file:///C:/Photos/summer%20trip.jpg"
        );
        assert_eq!(
            file_url(Path::new(r"\\?\C:\Photos\summer trip.jpg")),
            "file:///C:/Photos/summer%20trip.jpg"
        );
        assert_eq!(
            file_url(Path::new(r"\\?\UNC\server\Photos\summer trip.jpg")),
            "file://server/Photos/summer%20trip.jpg"
        );
    }

    #[test]
    fn collects_multiple_directories_recursively_without_duplicates() {
        let temporary = tempfile::tempdir().unwrap();
        let first = temporary.path().join("first");
        let nested = first.join("nested");
        let second = temporary.path().join("second");
        fs::create_dir_all(&nested).unwrap();
        fs::create_dir_all(&second).unwrap();
        fs::write(first.join("one.JPG"), []).unwrap();
        fs::write(nested.join("two.heic"), []).unwrap();
        fs::write(nested.join("four.jxl"), []).unwrap();
        fs::write(nested.join("ignored.txt"), []).unwrap();
        fs::write(second.join("three.heif"), []).unwrap();

        let paths = image_paths_in_directories(&[first.clone(), nested, second.clone()]).unwrap();
        assert_eq!(
            paths,
            vec![
                first.join("nested/four.jxl"),
                first.join("nested/two.heic"),
                first.join("one.JPG"),
                second.join("three.heif"),
            ]
        );
    }
}
