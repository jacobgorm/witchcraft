use anyhow::Result;
use log::debug;
use log::{Level, LevelFilter, Metadata, Record};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use text_splitter::TextSplitter;
use uuid::Uuid;

mod histogram;

use witchcraft::DB;

struct SimpleLogger;
impl log::Log for SimpleLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Level::Trace
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            println!("[{}] {}", record.level(), record.args());
        }
    }

    fn flush(&self) {}
}

static LOGGER: SimpleLogger = SimpleLogger;

#[derive(Debug, Deserialize)]
struct CSVRecord {
    name: String,
    body: String,
}

#[derive(Serialize, Deserialize)]
struct CorpusMetaData {
    key: String,
}

#[derive(Clone)]
struct SaliencySpan {
    start: usize,
    end: usize,
    score: f32,
    token: String,
}

fn normalized_saliency(score: f32, min_score: f32, max_score: f32) -> f32 {
    let range = max_score - min_score;
    if range.abs() < 1e-6 {
        0.5
    } else {
        ((score - min_score) / range).clamp(0.0, 1.0)
    }
}

fn saliency_bg(norm: f32) -> (u8, u8, u8) {
    let norm = norm.clamp(0.0, 1.0);
    let r = 32.0 + 223.0 * norm;
    let g = 48.0 + 152.0 * norm;
    let b = 72.0 * (1.0 - norm);
    (r.round() as u8, g.round() as u8, b.round() as u8)
}

fn colorize_saliency_text(text: &str, spans: &[SaliencySpan]) -> String {
    if spans.is_empty() {
        return text.to_string();
    }

    let min_score = spans
        .iter()
        .map(|span| span.score)
        .fold(f32::INFINITY, f32::min);
    let max_score = spans
        .iter()
        .map(|span| span.score)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut spans = spans.to_vec();
    spans.sort_unstable_by(|a, b| (a.start, a.end).cmp(&(b.start, b.end)));

    let mut out = String::new();
    let mut cursor = 0usize;
    for span in spans {
        if span.start < cursor || span.start >= span.end || span.end > text.len() {
            continue;
        }
        if !text.is_char_boundary(span.start) || !text.is_char_boundary(span.end) {
            continue;
        }
        out.push_str(&text[cursor..span.start]);
        let norm = normalized_saliency(span.score, min_score, max_score);
        let (r, g, b) = saliency_bg(norm);
        let fg = if norm > 0.58 { "0;0;0" } else { "255;255;255" };
        out.push_str(&format!(
            "\x1b[48;2;{r};{g};{b}m\x1b[38;2;{fg}m{}\x1b[0m",
            &text[span.start..span.end]
        ));
        cursor = span.end;
    }
    out.push_str(&text[cursor..]);
    out
}

fn print_saliency(embedder: &witchcraft::Embedder, text: &str) -> Result<()> {
    let output = embedder.embed_query_with_gate_scores_and_tokens(text)?;
    let gate_scores = output.gate_scores.ok_or_else(|| {
        anyhow::anyhow!("encoder assets do not expose token gate scores")
    })?;
    let spans: Vec<SaliencySpan> = output.offsets
        .into_iter()
        .zip(output.tokens.into_iter())
        .zip(gate_scores.into_iter())
        .filter_map(|(((start, end), token), score)| {
            (start < end).then_some(SaliencySpan { start, end, score, token })
        })
        .collect();
    if spans.is_empty() {
        anyhow::bail!("no visible token spans to display");
    }

    let min_score = spans
        .iter()
        .map(|span| span.score)
        .fold(f32::INFINITY, f32::min);
    let max_score = spans
        .iter()
        .map(|span| span.score)
        .fold(f32::NEG_INFINITY, f32::max);

    println!(
        "gate saliency range: min={min_score:.4} max={max_score:.4} (colors are normalized per input)"
    );
    println!("{}", colorize_saliency_text(text, &spans));
    println!();
    println!("{:>3} {:>10} {:>8} {:>16} text", "#", "gate", "norm", "token");
    for (idx, span) in spans.iter().enumerate() {
        if span.end > text.len()
            || !text.is_char_boundary(span.start)
            || !text.is_char_boundary(span.end)
        {
            continue;
        }
        let text_span = text[span.start..span.end].replace('\n', "\\n");
        let norm = normalized_saliency(span.score, min_score, max_score);
        println!(
            "{idx:>3} {:>10.4} {:>7.1}% {:>16?} {:?}",
            span.score,
            100.0 * norm,
            span.token,
            text_span
        );
    }
    Ok(())
}

#[cfg(feature = "mobileclip2")]
fn print_embedding(embedding: &candle_core::Tensor) -> Result<()> {
    let embedding = embedding
        .squeeze(0)?
        .to_device(&candle_core::Device::Cpu)?;
    let (tokens, _) = embedding.dims2()?;
    if tokens == 1 {
        println!("{}", serde_json::to_string(&embedding.squeeze(0)?.to_vec1::<f32>()?)?);
    } else {
        println!("{}", serde_json::to_string(&embedding.to_vec2::<f32>()?)?);
    }
    Ok(())
}

#[cfg(all(
    feature = "mobileclip2",
    not(any(feature = "modernbert", feature = "modernbert-quantized"))
))]
fn cosine_similarity(lhs: &candle_core::Tensor, rhs: &candle_core::Tensor) -> Result<f32> {
    Ok((lhs * rhs)?.sum_all()?.to_scalar::<f32>()?)
}

#[cfg(all(
    feature = "mobileclip2",
    any(feature = "modernbert", feature = "modernbert-quantized")
))]
fn maxsim_similarity(query: &candle_core::Tensor, image: &candle_core::Tensor) -> Result<f32> {
    let image = image.to_device(query.device())?;
    let similarities = query.matmul(&image.transpose(0, 1)?)?;
    Ok(similarities.max(1)?.mean_all()?.to_scalar::<f32>()?)
}

#[cfg(feature = "mobileclip2")]
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

#[cfg(feature = "mobileclip2")]
fn benchmark_image_embeddings(
    embedder: &witchcraft::Embedder,
    path: &std::path::Path,
    args: &[String],
) -> Result<()> {
    let profile = args
        .iter()
        .any(|arg| matches!(arg.as_str(), "profile" | "--profile"));
    let counts = args
        .iter()
        .filter(|arg| !matches!(arg.as_str(), "profile" | "--profile"))
        .map(String::as_str)
        .collect::<Vec<_>>();
    anyhow::ensure!(counts.len() <= 2, "benchimage expects [iters] [warmup] [profile]");
    let iters = bench_count(counts.first().copied(), "WARP_BENCH_ITERS", 25)?;
    let warmup = bench_count(counts.get(1).copied(), "WARP_BENCH_WARMUP", 5)?;
    anyhow::ensure!(iters > 0, "image benchmark iterations must be greater than zero");

    let image = witchcraft::mobileclip2::load_image(path, embedder.image_size())?
        .to_device(embedder.device())?
        .unsqueeze(0)?;
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
        let started = std::time::Instant::now();
        run()?;
        times.push(started.elapsed());
    }
    times.sort_unstable();
    let mean = times.iter().map(std::time::Duration::as_secs_f64).sum::<f64>() / iters as f64;
    let p95 = ((iters * 95 + 99) / 100).saturating_sub(1).min(iters - 1);
    println!("image: {}", path.display());
    println!("device: {:?}", embedder.device().location());
    println!("image_size: {}x{}", embedder.image_size(), embedder.image_size());
    println!("embedding_shape: {:?}", run()?.dims());
    println!("iters: {iters} warmup: {warmup}");
    println!(
        "model_only: median={:.2} ms mean={:.2} ms p95={:.2} ms throughput={:.2} images/s",
        times[iters / 2].as_secs_f64() * 1000.0,
        mean * 1000.0,
        times[p95].as_secs_f64() * 1000.0,
        1.0 / mean
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

fn split_doc(body: String) -> Vec<String> {
    let max_characters = 300;
    let splitter = TextSplitter::new(max_characters);
    splitter
        .chunks(&body)
        .map(|body| format!("{body}\n").to_string())
        .collect()
}

pub fn read_csv(db: &mut DB, csvname: std::path::PathBuf) -> Result<()> {
    println!("register documents from CSV...");

    let file = File::open(csvname)?;
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(false)
        .from_reader(file);

    for result in rdr.deserialize() {
        let record: CSVRecord = result?;
        let metadata = CorpusMetaData { key: record.name };
        let metadata = serde_json::to_string(&metadata)?;
        let body = record.body;

        let bodies = split_doc(body.clone());
        let lens = bodies.iter().map(|b| b.chars().count()).collect();
        let body = bodies.join("");
        let uuid = Uuid::new_v5(&Uuid::NAMESPACE_OID, body.as_bytes());
        db.add_doc(None, &uuid, None, &metadata, &body, Some(lens))
            .unwrap();
    }

    Ok(())
}

pub fn bulk_search(
    db: &DB,
    embedder: Option<&witchcraft::Embedder>,
    csvname: std::path::PathBuf,
    outputname: std::path::PathBuf,
    use_fulltext: bool,
) -> Result<()> {
    validate_semantic_search(db, embedder)?;

    let file = File::open(csvname)?;
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(false)
        .from_reader(file);

    let file = File::create(outputname).unwrap();
    let mut writer = BufWriter::new(file);

    let mut metadata_query = db.query("SELECT metadata FROM document WHERE rowid = ?1")?;
    let mut histogram = histogram::Histogram::new(10000);
    let mut embedder_histogram = histogram::Histogram::new(10000);
    witchcraft::reset_bucket_io_counters();

    for result in rdr.deserialize() {
        let record: (String, String) = result?;
        let key = record.0;
        let question = record.1;
        let top_k = 100;

        debug!("searching for: {}", question);
        let now = std::time::Instant::now();
        let fts_start = std::time::Instant::now();
        let fts_matches = if use_fulltext {
            witchcraft::fulltext_search(db, &question, top_k, None)?
        } else {
            vec![]
        };
        if use_fulltext {
            debug!(
                "fulltext search took {} ms.",
                fts_start.elapsed().as_millis()
            );
        }

        let sem_matches = if let Some(embedder) = embedder {
            let now = std::time::Instant::now();
            let qe = witchcraft::embed_query_for_search(embedder, &question)?;
            let embedder_latency_ms = now.elapsed().as_millis() as u32;
            embedder_histogram.record(embedder_latency_ms);

            let match_start = std::time::Instant::now();
            let matches = witchcraft::match_centroids_with_query_weights(
                db,
                &qe.embeddings,
                qe.weights.as_deref(),
                0.0,
                top_k,
                None,
            )?;
            debug!(
                "match_centroids call took {} ms.",
                match_start.elapsed().as_millis()
            );
            matches
        } else {
            vec![]
        };
        let sem_idxs: Vec<witchcraft::DocPtr> = sem_matches
            .iter()
            .map(|&(_, idx, sub_idx)| (idx, sub_idx))
            .collect();

        let fusion_start = std::time::Instant::now();
        let mut fused = if use_fulltext {
            let fts_idxs: Vec<witchcraft::DocPtr> = fts_matches
                .iter()
                .map(|&(_, idx, sub_idx)| (idx, sub_idx))
                .collect();
            witchcraft::hybrid_reciprocal_rank_fusion(&fts_idxs, &sem_idxs, 60.0)
        } else {
            sem_idxs
        };
        fused.truncate(top_k);
        debug!(
            "rank fusion took {} ms.",
            fusion_start.elapsed().as_millis()
        );

        let metadata_start = std::time::Instant::now();
        let mut metadatas = vec![];
        for (idx, _sub_idx) in &fused {
            let metadata = metadata_query.query_row((*idx,), |row| row.get::<_, String>(0))?;
            metadatas.push(metadata);
        }
        debug!(
            "fetching {} metadata took {} ms.",
            metadatas.len(),
            metadata_start.elapsed().as_millis()
        );
        let total_ms = now.elapsed().as_millis();
        histogram.record(total_ms.try_into().unwrap());
        debug!("search took {} ms in total", now.elapsed().as_millis());

        write!(writer, "{}\t", key).unwrap();
        for metadata in &metadatas {
            let data: CorpusMetaData = serde_json::from_str(metadata)?;
            write!(writer, "{},", data.key).unwrap();
        }
        writeln!(writer).unwrap();
        writer.flush().unwrap();
    }
    if embedder.is_some() {
        println!("p95 embedder latency = {} ms", embedder_histogram.p95());
    }
    println!("p95 total search latency = {} ms", histogram.p95());
    witchcraft::log_bucket_io_counters();
    Ok(())
}

pub fn bulk_exact_search(
    db: &DB,
    embedder: &witchcraft::Embedder,
    csvname: std::path::PathBuf,
    outputname: std::path::PathBuf,
) -> Result<()> {
    let file = File::open(csvname)?;
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(false)
        .from_reader(file);

    let mut records = Vec::new();
    let mut queries = Vec::new();
    let mut embedder_histogram = histogram::Histogram::new(10000);
    for result in rdr.deserialize() {
        let record: (String, String) = result?;
        let now = std::time::Instant::now();
        let (qe, _offsets) = embedder.embed_query(&record.1)?;
        let qe = qe.get(0)?;
        embedder_histogram.record(now.elapsed().as_millis() as u32);
        records.push(record);
        queries.push(qe);
    }

    let now = std::time::Instant::now();
    let results = witchcraft::exact_match_centroids_bulk(db, &queries, 100)?;
    println!("exact search took {} ms.", now.elapsed().as_millis());

    let file = File::create(outputname).unwrap();
    let mut writer = BufWriter::new(file);
    let mut metadata_query = db.query("SELECT metadata FROM document WHERE rowid = ?1")?;
    for ((key, _question), matches) in records.iter().zip(results.iter()) {
        write!(writer, "{}\t", key).unwrap();
        for (_score, idx, _sub_idx) in matches {
            let metadata = metadata_query.query_row((*idx,), |row| row.get::<_, String>(0))?;
            let data: CorpusMetaData = serde_json::from_str(&metadata)?;
            write!(writer, "{},", data.key).unwrap();
        }
        writeln!(writer).unwrap();
    }
    writer.flush().unwrap();
    println!("p95 embedder latency = {} ms", embedder_histogram.p95());
    Ok(())
}

fn validate_semantic_search(db: &DB, embedder: Option<&witchcraft::Embedder>) -> Result<()> {
    if embedder.is_none() {
        return Ok(());
    }
    if let Some(reason) = witchcraft::semantic_index_unavailable_reason(db)? {
        anyhow::bail!("semantic index is not ready: {reason}; run warp-cli index or warp-cli reindex");
    }
    Ok(())
}

#[cfg(feature = "mobileclip2")]
fn image_paths_in_directory(directory: &std::path::Path) -> Result<Vec<std::path::PathBuf>> {
    anyhow::ensure!(directory.is_dir(), "{} is not a directory", directory.display());
    let mut paths = std::fs::read_dir(directory)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .map(|extension| {
                        matches!(
                            extension.to_ascii_lowercase().as_str(),
                            "jpg" | "jpeg" | "png" | "webp" | "bmp" | "gif" | "tif" | "tiff"
                        )
                    })
                    .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths)
}

fn main() -> Result<()> {
    let _ = log::set_logger(&LOGGER).map(|()| log::set_max_level(LevelFilter::Info));

    let args: Vec<String> = env::args().collect();
    let assets = std::path::PathBuf::from("assets");
    let db_name = std::path::PathBuf::from("mydb.sqlite");

    if args.len() == 3 && args[1] == "readcsv" {
        let mut db = DB::new_fast(db_name).unwrap();
        let csvname = &args[2];
        read_csv(&mut db, csvname.into()).unwrap();
    } else if args.len() == 2 && &args[1] == "embed" {
        let device = witchcraft::make_device();
        let embedder = witchcraft::Embedder::new(&device, &assets).unwrap();
        let db = DB::new_fast(db_name).unwrap();
        let _got = witchcraft::embed_chunks(&db, &embedder, None).unwrap();
    } else if args.len() == 2 && &args[1] == "index" {
        let device = witchcraft::make_device();
        let embedder = witchcraft::Embedder::new(&device, &assets).unwrap();
        let db = DB::new_fast(db_name).unwrap();
        witchcraft::index_chunks_with_options(
            &db,
            Some(&embedder),
            false,
            witchcraft::IndexOptions::default().force_flush(),
        )
        .unwrap();
    } else if args.len() == 2 && &args[1] == "reindex" {
        println!("reindex: opening {}", db_name.display());
        let db = DB::new_fast(db_name).unwrap();
        println!("reindex: rebuilding semantic index from cached embeddings");
        witchcraft::index_chunks_with_options(
            &db,
            None,
            true,
            witchcraft::IndexOptions::default().force_flush(),
        )
        .unwrap();
        println!("reindex: done");
    } else if (args.len() == 3 || args.len() == 4) && &args[1] == "embedimage" {
        #[cfg(feature = "mobileclip2")]
        {
            let device = witchcraft::make_device();
            let embedder = witchcraft::Embedder::new(&device, &assets)?;
            let mut db = DB::new_fast(db_name)?;
            let (rowid, count) = if args.len() == 3 {
                witchcraft::cache_image_embeddings_auto(
                    &mut db,
                    &embedder,
                    std::path::Path::new(&args[2]),
                )?
            } else {
                let rowid = args[2].parse::<u64>()?;
                let count = witchcraft::cache_image_embeddings(
                    &db,
                    &embedder,
                    rowid,
                    std::path::Path::new(&args[3]),
                )?;
                (rowid, count)
            };
            println!("embedded {count} image vectors for rowid {rowid}");
        }
        #[cfg(not(feature = "mobileclip2"))]
        anyhow::bail!("image embedding requires building with --features mobileclip2");
    } else if (args.len() == 3 || args.len() == 4) && &args[1] == "embedimages" {
        #[cfg(feature = "mobileclip2")]
        {
            let (force, directory) = if args.len() == 4 {
                anyhow::ensure!(
                    args[2] == "--force",
                    "expected --force before the image directory"
                );
                (true, &args[3])
            } else {
                (false, &args[2])
            };
            let paths = image_paths_in_directory(std::path::Path::new(directory))?;
            anyhow::ensure!(
                !paths.is_empty(),
                "no supported images found in {}",
                directory
            );
            let device = witchcraft::make_device();
            let embedder = witchcraft::Embedder::new(&device, &assets)?;
            let mut db = DB::new_fast(db_name)?;
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
                "embedded {} images as {vector_count} vectors, skipped {skipped}; run `{} index` to index them",
                results.len(),
                args[0]
            );
        }
        #[cfg(not(feature = "mobileclip2"))]
        anyhow::bail!("image embedding requires building with --features mobileclip2");
    } else if args.len() >= 3 && &args[1] == "saliency" {
        let device = witchcraft::make_device();
        let embedder = witchcraft::Embedder::new(&device, &assets).unwrap();
        let text = args[2..].join(" ");
        print_saliency(&embedder, &text)?;
    } else if args.len() == 3 && &args[1] == "image" {
        #[cfg(feature = "mobileclip2")]
        {
            let device = witchcraft::make_device();
            let embedder = witchcraft::Embedder::new(&device, &assets)?;
            print_embedding(&embedder.embed_image_path(&args[2])?)?;
        }
        #[cfg(not(feature = "mobileclip2"))]
        anyhow::bail!("image embeddings require building with --features mobileclip2");
    } else if args.len() >= 3 && &args[1] == "benchimage" {
        #[cfg(feature = "mobileclip2")]
        {
            let device = witchcraft::make_device();
            let started = std::time::Instant::now();
            let embedder = witchcraft::Embedder::new(&device, &assets)?;
            println!("model_load_ms: {:.2}", started.elapsed().as_secs_f64() * 1000.0);
            benchmark_image_embeddings(
                &embedder,
                std::path::Path::new(&args[2]),
                &args[3..],
            )?;
        }
        #[cfg(not(feature = "mobileclip2"))]
        anyhow::bail!("image benchmarks require building with --features mobileclip2");
    } else if args.len() >= 4 && &args[1] == "scoreimage" {
        #[cfg(feature = "mobileclip2")]
        {
            let device = witchcraft::make_device();
            let embedder = witchcraft::Embedder::new(&device, &assets)?;
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
        }
        #[cfg(not(feature = "mobileclip2"))]
        anyhow::bail!("image/text scoring requires building with --features mobileclip2");
    } else if args.len() >= 3 && (args[1] == "query" || args[1] == "hybrid" || args[1] == "fulltext") {
        let embedder = if args[1] != "fulltext" {
            let device = witchcraft::make_device();
            Some(witchcraft::Embedder::new(&device, &assets).unwrap())
        } else {
            None
        };
        let mut cache = witchcraft::EmbeddingsCache::new(1);
        let db = DB::new_reader(db_name).unwrap();
        validate_semantic_search(&db, embedder.as_ref())?;
        let q = &args[2..].join(" ");
        let use_fulltext = args[1] == "hybrid" || args[1] == "fulltext";
        witchcraft::reset_bucket_io_counters();
        let results =
            witchcraft::search(&db, embedder.as_ref(), &mut cache, q, 0.0, 10, use_fulltext, None).unwrap();
        for (score, _metadata, bodies, sub_idx, _date) in results {
            let idx = (sub_idx as usize).min(bodies.len().saturating_sub(1));
            let body = &bodies[idx];
            println!("{score}: {body} @ {sub_idx}");
            println!("=============================================");
        }
        witchcraft::log_bucket_io_counters();
    } else if args.len() >= 4
        && (args[1] == "querycsv" || args[1] == "hybridcsv" || args[1] == "fulltextcsv" || args[1] == "exactcsv")
    {
        let use_fulltext = args[1] == "hybridcsv" || args[1] == "fulltextcsv";
        let embedder = if args[1] != "fulltextcsv" {
            let device = witchcraft::make_device();
            Some(witchcraft::Embedder::new(&device, &assets).unwrap())
        } else {
            None
        };
        let db = DB::new_reader(db_name).unwrap();
        let csvname = &args[2];
        let outputname = &args[3];
        if args[1] == "exactcsv" {
            bulk_exact_search(&db, embedder.as_ref().unwrap(), csvname.into(), outputname.into())?;
        } else {
            bulk_search(
                &db,
                embedder.as_ref(),
                csvname.into(),
                outputname.into(),
                use_fulltext,
            )?;
        }
    } else if args.len() >= 4 && &args[1] == "score" {
        let device = witchcraft::make_device();
        let embedder = witchcraft::Embedder::new(&device, &assets).unwrap();
        let mut cache = witchcraft::EmbeddingsCache::new(1);
        let sentences: Vec<String> = std::env::args().skip(3).collect();
        let scores =
            witchcraft::score_query_sentences(&embedder, &mut cache, &args[2], &sentences).unwrap();
        for (i, score) in scores.iter().enumerate() {
            println!("`{}': score={}", args[3 + i], *score);
        }
    } else if args.len() == 2 && &args[1] == "clear" {
        let mut db = DB::new_fast(db_name).unwrap();
        db.clear();
    } else {
        eprintln!("\n*** Usage: {} clear | readcsv <file> | embed | embedimage [rowid] <path> | embedimages [--force] <directory> | index | reindex | saliency <text> | image <path> | benchimage <path> [iters] [warmup] [profile] | scoreimage <path> <text>... | query <text> | hybrid <text> | querycsv|exactcsv <file> <results-file> ***\n", args[0]);
    };
    Ok(())
}
