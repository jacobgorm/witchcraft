use super::{print_search_results, validate_semantic_search, DB};
use anyhow::Result;
use std::env;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[cfg(unix)]
fn terminal_dimensions() -> Option<(u16, u16)> {
    let mut size = std::mem::MaybeUninit::<libc::winsize>::zeroed();
    let result = unsafe { libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ, size.as_mut_ptr()) };
    if result != 0 {
        return None;
    }
    let size = unsafe { size.assume_init() };
    (size.ws_col > 0 && size.ws_row > 0).then_some((size.ws_col, size.ws_row))
}

#[cfg(not(unix))]
fn terminal_dimensions() -> Option<(u16, u16)> {
    None
}

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

fn image_contact_sheet(paths: &[PathBuf], cell_width: u16, cell_height: u16) -> Result<tempfile::NamedTempFile> {
    const COLUMN_PIXELS: u32 = 8;
    const ROW_PIXELS: u32 = 16;
    const TILE_PADDING: u32 = 4;

    let tile_width = u32::from(cell_width).saturating_mul(COLUMN_PIXELS).max(16);
    let tile_height = u32::from(cell_height).saturating_mul(ROW_PIXELS).max(16);
    let mut sheet = image::RgbaImage::from_pixel(
        tile_width.saturating_mul(3),
        tile_height.saturating_mul(3),
        image::Rgba([0, 0, 0, 255]),
    );
    for (index, path) in paths.iter().take(9).enumerate() {
        let decoded = match witchcraft::mobileclip2::decode_image_oriented(path)
            .map_err(anyhow::Error::from)
        {
            Ok(decoded) => decoded,
            Err(error) => {
                eprintln!("warning: could not display {}: {error}", path.display());
                continue;
            },
        };
        let available_width = tile_width.saturating_sub(TILE_PADDING.saturating_mul(2));
        let available_height = tile_height.saturating_sub(TILE_PADDING.saturating_mul(2));
        let thumbnail = decoded.thumbnail(available_width, available_height).to_rgba8();
        let column = (index % 3) as u32;
        let row = (index / 3) as u32;
        let x = column * tile_width + (tile_width - thumbnail.width()) / 2;
        let y = row * tile_height + (tile_height - thumbnail.height()) / 2;
        image::imageops::overlay(&mut sheet, &thumbnail, i64::from(x), i64::from(y));
    }

    let file = tempfile::Builder::new().suffix(".png").tempfile()?;
    sheet.save(file.path())?;
    Ok(file)
}

pub fn display_image_grid(paths: &[PathBuf]) {
    if paths.is_empty() || !std::io::stdout().is_terminal() {
        return;
    }
    let viu_available = Command::new("viu")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false);
    if !viu_available {
        return;
    }

    let (terminal_width, terminal_height) = terminal_dimensions().unwrap_or((120, 40));
    let horizontal_margin = 2u16;
    let top_margin = 1u16;
    let bottom_margin = 4u16;
    let grid_width = terminal_width
        .saturating_sub(horizontal_margin.saturating_mul(2))
        .max(3);
    let grid_height = terminal_height
        .saturating_sub(top_margin.saturating_add(bottom_margin))
        .max(3);
    let cell_width = (grid_width / 3).max(1);
    let cell_height = (grid_height / 3).max(1);
    let sheet = match image_contact_sheet(paths, cell_width, cell_height) {
        Ok(sheet) => sheet,
        Err(error) => {
            eprintln!("warning: could not build image grid: {error}");
            return;
        },
    };

    print!("\x1b[2J\x1b[H");
    let _ = std::io::stdout().flush();
    let child = Command::new("viu")
        .args([
            "-a",
            "-x",
            &horizontal_margin.to_string(),
            "-y",
            &top_margin.to_string(),
            "-w",
            &grid_width.to_string(),
            "-h",
            &grid_height.to_string(),
            "-s",
        ])
        .arg(sheet.path())
        .spawn();
    let Ok(mut child) = child else {
        return;
    };
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if !status.success() {
                    eprintln!("warning: viu could not display image grid");
                }
                break;
            },
            Ok(None) if started.elapsed() < Duration::from_secs(2) => {
                std::thread::sleep(Duration::from_millis(10));
            },
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                eprintln!("warning: viu timed out; skipping image grid");
                return;
            },
            Err(error) => {
                eprintln!("warning: could not wait for viu: {error}");
                return;
            },
        }
    }
    let cursor_row = top_margin
        .saturating_add(cell_height.saturating_mul(3))
        .saturating_add(1)
        .min(terminal_height.saturating_sub(1).max(1));
    print!("\x1b[{cursor_row};1H");
    let _ = std::io::stdout().flush();
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
        "clipembedimages" => {
            anyhow::ensure!(
                args.len() >= 4,
                "clipembedimages expects <matrix> <directory>..."
            );
            let directories = args[3..].iter().map(PathBuf::from).collect::<Vec<_>>();
            let paths = image_paths_in_directories(&directories)?;
            anyhow::ensure!(!paths.is_empty(), "no supported images found under the supplied directories");
            let device = witchcraft::make_device();
            let model = witchcraft::mobileclip2::StockMobileClip2::load(&device, assets)?;
            super::mobileclip2_matrix::create(&model, &paths, Path::new(&args[2]))?;
        },
        "clipquery" => {
            let display_images = args.get(2).is_some_and(|argument| argument == "-d");
            let matrix_index = if display_images { 3 } else { 2 };
            anyhow::ensure!(
                args.len() > matrix_index + 1,
                "clipquery expects [-d] <matrix> <text>..."
            );
            let text = args[matrix_index + 1..].join(" ");
            let device = witchcraft::make_device();
            let model = witchcraft::mobileclip2::StockMobileClip2::load(&device, assets)?;
            let results = super::mobileclip2_matrix::query(
                &model,
                Path::new(&args[matrix_index]),
                &text,
                10,
            )?;
            let image_paths = results
                .iter()
                .map(|(score, path)| {
                    println!("{score:.8}: {}", path.display());
                    path.clone()
                })
                .collect::<Vec<_>>();
            if display_images {
                display_image_grid(&image_paths);
            }
        },
        "clipivfbuild" => {
            let force = args.get(2).is_some_and(|argument| argument == "--force");
            let matrix_index = if force { 3 } else { 2 };
            anyhow::ensure!(
                args.len() == matrix_index + 1,
                "clipivfbuild expects [--force] <matrix>"
            );
            super::mobileclip2_ivf::build(Path::new(&args[matrix_index]), force)?;
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
    use super::image_paths_in_directories;
    use std::fs;

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
