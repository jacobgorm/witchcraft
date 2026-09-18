use super::{print_search_results, validate_semantic_search, DB};
use anyhow::Result;
use std::env;
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

fn image_grid_html(paths: &[PathBuf]) -> String {
    let mut tiles = String::new();
    for path in paths.iter().take(9) {
        let url = file_url(path);
        let label = html_escape(&path.to_string_lossy());
        tiles.push_str(&format!(
            "<figure><a href=\"{url}\"><img src=\"{url}\" alt=\"{label}\"></a><figcaption>{label}</figcaption></figure>\n"
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
    let preview = env::temp_dir().join("witchcraft-image-preview.html");
    if let Err(error) = std::fs::write(&preview, image_grid_html(paths)) {
        eprintln!("warning: could not write image preview: {error}");
        return;
    }
    if let Err(error) = open_image_grid(&preview) {
        eprintln!("warning: could not open image preview: {error}");
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
        "clipindex" => {
            anyhow::ensure!(args.len() == 2, "clipindex takes no arguments");
            super::mobileclip2_ivf::update_index()?;
        },
        "clipivfquery" => {
            let display_images = args[2..].iter().any(|argument| argument == "-d");
            let exact = args[2..].iter().any(|argument| argument == "--exact");
            let query_args = args[2..]
                .iter()
                .filter(|argument| !matches!(argument.as_str(), "-d" | "--exact"))
                .cloned()
                .collect::<Vec<_>>();
            anyhow::ensure!(
                !query_args.is_empty(),
                "clipivfquery expects [-d] [--exact] <text>..."
            );
            let results = super::mobileclip2_ivf::query(
                assets,
                &query_args.join(" "),
                10,
                exact,
            )?;
            let image_paths = print_search_results(results);
            if display_images {
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
    use super::{file_url, image_grid_html, image_paths_in_directories};
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
        fs::write(nested.join("ignored.txt"), []).unwrap();
        fs::write(second.join("three.heif"), []).unwrap();

        let paths = image_paths_in_directories(&[first.clone(), nested, second.clone()]).unwrap();
        assert_eq!(
            paths,
            vec![first.join("nested/two.heic"), first.join("one.JPG"), second.join("three.heif")]
        );
    }
}
