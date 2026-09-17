use anyhow::Result;
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const DIMENSIONS: usize = 512;
const BATCH_SIZE: usize = 8;

#[derive(Serialize, Deserialize)]
struct MatrixMetadata {
    version: usize,
    model: String,
    dimensions: usize,
    image_size: usize,
    normalized: bool,
    paths: Vec<String>,
}

fn sidecar(prefix: &Path, suffix: &str) -> PathBuf {
    let mut path = prefix.as_os_str().to_owned();
    path.push(suffix);
    PathBuf::from(path)
}

fn flush_batch(
    model: &witchcraft::mobileclip2::StockMobileClip2,
    images: &mut Vec<Tensor>,
    paths: &mut Vec<PathBuf>,
    output: &mut BufWriter<File>,
    embedded_paths: &mut Vec<String>,
) -> Result<()> {
    if images.is_empty() {
        return Ok(());
    }
    let batch = Tensor::stack(images, 0)?;
    let rows = model
        .embed_images(&batch)?
        .to_device(&Device::Cpu)?
        .to_vec2::<f32>()?;
    anyhow::ensure!(rows.len() == paths.len(), "MobileCLIP2 batch size changed");
    for (path, row) in paths.drain(..).zip(rows) {
        anyhow::ensure!(row.len() == DIMENSIONS, "expected a {DIMENSIONS}d image vector");
        for value in row {
            output.write_all(&value.to_le_bytes())?;
        }
        println!("embedded {}", path.display());
        embedded_paths.push(path.to_string_lossy().into_owned());
    }
    images.clear();
    Ok(())
}

pub fn create(
    model: &witchcraft::mobileclip2::StockMobileClip2,
    paths: &[PathBuf],
    prefix: &Path,
) -> Result<usize> {
    let matrix_path = sidecar(prefix, ".f32");
    let metadata_path = sidecar(prefix, ".json");
    let mut output = BufWriter::new(File::create(&matrix_path)?);
    let mut images = Vec::with_capacity(BATCH_SIZE);
    let mut batch_paths = Vec::with_capacity(BATCH_SIZE);
    let mut embedded_paths = Vec::with_capacity(paths.len());

    for path in paths {
        match witchcraft::mobileclip2::load_image(path, model.image_size()) {
            Ok(image) => {
                images.push(image);
                batch_paths.push(path.clone());
            },
            Err(error) => {
                eprintln!("[WARN] skipping {}: {error}", path.display());
                continue;
            },
        }
        if images.len() == BATCH_SIZE {
            flush_batch(
                model,
                &mut images,
                &mut batch_paths,
                &mut output,
                &mut embedded_paths,
            )?;
        }
    }
    flush_batch(
        model,
        &mut images,
        &mut batch_paths,
        &mut output,
        &mut embedded_paths,
    )?;
    output.flush()?;

    let metadata = MatrixMetadata {
        version: 1,
        model: "timm/MobileCLIP2-S0-OpenCLIP".to_string(),
        dimensions: DIMENSIONS,
        image_size: model.image_size(),
        normalized: true,
        paths: embedded_paths,
    };
    serde_json::to_writer_pretty(File::create(&metadata_path)?, &metadata)?;
    println!(
        "wrote {} rows to {} and {}",
        metadata.paths.len(),
        matrix_path.display(),
        metadata_path.display()
    );
    Ok(metadata.paths.len())
}

pub fn query(
    model: &witchcraft::mobileclip2::StockMobileClip2,
    prefix: &Path,
    text: &str,
    limit: usize,
) -> Result<Vec<(f32, PathBuf)>> {
    let metadata_path = sidecar(prefix, ".json");
    let matrix_path = sidecar(prefix, ".f32");
    let metadata: MatrixMetadata = serde_json::from_reader(File::open(&metadata_path)?)?;
    anyhow::ensure!(metadata.version == 1, "unsupported matrix version {}", metadata.version);
    anyhow::ensure!(metadata.dimensions == DIMENSIONS, "expected a {DIMENSIONS}d matrix");
    anyhow::ensure!(metadata.normalized, "matrix rows are not normalized");

    let query = model
        .embed_text(text)?
        .to_device(&Device::Cpu)?
        .squeeze(0)?
        .to_vec1::<f32>()?;
    anyhow::ensure!(query.len() == DIMENSIONS, "expected a {DIMENSIONS}d text vector");

    let mut bytes = Vec::new();
    File::open(&matrix_path)?.read_to_end(&mut bytes)?;
    let expected = metadata.paths.len() * DIMENSIONS * std::mem::size_of::<f32>();
    anyhow::ensure!(
        bytes.len() == expected,
        "matrix has {} bytes, expected {expected} for {} rows",
        bytes.len(),
        metadata.paths.len()
    );
    score_rows(&bytes, &query, metadata.paths, limit)
}

fn score_rows(
    bytes: &[u8],
    query: &[f32],
    paths: Vec<String>,
    limit: usize,
) -> Result<Vec<(f32, PathBuf)>> {
    anyhow::ensure!(query.len() == DIMENSIONS, "expected a {DIMENSIONS}d query");
    anyhow::ensure!(bytes.len() == paths.len() * DIMENSIONS * 4, "matrix size does not match paths");
    let mut scored = Vec::with_capacity(paths.len());
    for (index, path) in paths.into_iter().enumerate() {
        let row = &bytes[index * DIMENSIONS * 4..(index + 1) * DIMENSIONS * 4];
        let score: f32 = row
            .chunks_exact(4)
            .zip(query)
            .map(|(value, query)| {
                f32::from_le_bytes(value.try_into().expect("four-byte float")) * query
            })
            .sum();
        scored.push((score, PathBuf::from(path)));
    }
    scored.sort_unstable_by(|left, right| right.0.total_cmp(&left.0));
    scored.truncate(limit.min(scored.len()));
    Ok(scored)
}

#[cfg(test)]
mod tests {
    use super::{score_rows, DIMENSIONS};

    #[test]
    fn exhaustive_scores_are_sorted_by_dot_product() {
        let mut rows = vec![0.0f32; DIMENSIONS * 2];
        rows[0] = 1.0;
        rows[DIMENSIONS + 1] = 1.0;
        let bytes = rows
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let mut query = vec![0.0; DIMENSIONS];
        query[1] = 1.0;
        let results = score_rows(
            &bytes,
            &query,
            vec!["first.jpg".to_string(), "second.jpg".to_string()],
            10,
        )
        .unwrap();
        assert_eq!(results[0].1.to_str(), Some("second.jpg"));
        assert_eq!(results[0].0, 1.0);
        assert_eq!(results[1].0, 0.0);
    }
}
