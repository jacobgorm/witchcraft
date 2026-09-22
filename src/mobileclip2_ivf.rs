use anyhow::Result;
use candle_core::Tensor;
use std::path::{Path, PathBuf};
use witchcraft::{DB, IndexOptions};

const DATABASE: &str = "mobileclip2-ivf.sqlite";
const BATCH_SIZE: usize = 8;

pub fn register(paths: &[PathBuf]) -> Result<()> {
    let mut db = DB::new_fast(PathBuf::from(DATABASE))?;
    let (added, reused, skipped) = register_paths(&mut db, paths)?;
    db.checkpoint();
    println!(
        "registered {added} new images, reused {reused}, skipped {skipped}; run `warp-cli clipembed` next"
    );
    Ok(())
}

fn register_paths(db: &mut DB, paths: &[PathBuf]) -> Result<(usize, usize, usize)> {
    let mut added = 0usize;
    let mut reused = 0usize;
    let mut skipped = 0usize;
    for path in paths {
        if let Err(error) = witchcraft::mobileclip2::validate_image_header(path) {
            skipped += 1;
            eprintln!("[WARN] skipping {}: {error}", path.display());
            continue;
        }
        let (rowid, inserted) = witchcraft::register_image_document_auto(db, path)?;
        let display_path = witchcraft::normalize_image_path(path);
        if inserted {
            added += 1;
            println!("registered rowid {rowid}: {}", display_path.display());
        } else {
            reused += 1;
            println!("already registered rowid {rowid}: {}", display_path.display());
        }
    }
    Ok((added, reused, skipped))
}

fn missing_embeddings(db: &DB) -> Result<Vec<(u64, PathBuf)>> {
    let mut query = db.query(
        "SELECT document.rowid, document.body
         FROM document LEFT JOIN chunk ON document.hash = chunk.hash
         WHERE chunk.hash IS NULL OR chunk.embedding_count = 0
         ORDER BY document.rowid",
    )?;
    let rows = query.query_map((), |row| {
        Ok((
            row.get::<_, i64>(0)?,
            PathBuf::from(row.get::<_, String>(1)?),
        ))
    })?;
    rows.map(|row| {
        let (rowid, path) = row?;
        Ok((rowid.try_into()?, path))
    })
    .collect()
}

pub fn embed(assets: &Path) -> Result<()> {
    let db = DB::new_fast(PathBuf::from(DATABASE))?;
    let missing = missing_embeddings(&db)?;
    if missing.is_empty() {
        println!("all registered images already have pooled MobileCLIP2 embeddings");
        return Ok(());
    }

    let fallback_device = witchcraft::make_device();
    let device = witchcraft::make_image_device(&fallback_device);
    let model = witchcraft::mobileclip2::StockMobileClip2::load(&device, assets)?;
    log::info!("pooled MobileCLIP2 image device: {:?}", model.device().location());
    let progress = witchcraft::progress::new_with_label(missing.len() as u64, "embed images");
    let mut embedded = 0usize;
    let mut failed = 0usize;
    // Keep two prepared batches queued so image reads and decoding overlap the
    // current GPU forward pass without allowing decoded tensors to grow unbounded.
    let (prepared_tx, prepared_rx) = std::sync::mpsc::sync_channel(2);
    let image_size = model.image_size();
    std::thread::scope(|scope| -> Result<()> {
        scope.spawn(move || {
            for documents in missing.chunks(BATCH_SIZE) {
                let prepared = std::thread::scope(|decode_scope| {
                    let workers = documents
                        .iter()
                        .map(|(rowid, path)| {
                            let rowid = *rowid;
                            let path = path.clone();
                            decode_scope.spawn(move || {
                                let image = witchcraft::mobileclip2::load_image(&path, image_size)
                                    .map_err(anyhow::Error::from);
                                (rowid, path, image)
                            })
                        })
                        .collect::<Vec<_>>();
                    workers
                        .into_iter()
                        .map(|worker| worker.join().expect("image preparation worker panicked"))
                        .collect::<Vec<_>>()
                });
                if prepared_tx.send(prepared).is_err() {
                    break;
                }
            }
        });

        let mut pipeline_error = None;
        while let Ok(prepared) = prepared_rx.recv() {
            let batch_result = (|| -> Result<()> {
                let mut images = Vec::with_capacity(prepared.len());
                let mut valid = Vec::with_capacity(prepared.len());
                for (rowid, path, image) in prepared {
                    match image {
                        Ok(image) => {
                            images.push(image);
                            valid.push((rowid, path));
                        },
                        Err(error) => {
                            failed += 1;
                            eprintln!(
                                "[WARN] skipping rowid {rowid} ({}): {error}",
                                path.display()
                            );
                            progress.inc(1);
                        },
                    }
                }
                if images.is_empty() {
                    return Ok(());
                }

                let batch = Tensor::stack(&images, 0)?;
                let outputs = model.embed_images(&batch)?;
                anyhow::ensure!(outputs.dim(0)? == valid.len(), "MobileCLIP2 batch size changed");
                for (index, (rowid, _path)) in valid.into_iter().enumerate() {
                    let output = outputs.get(index)?.unsqueeze(0)?.unsqueeze(0)?;
                    witchcraft::cache_precomputed_l2_int8_image_embeddings(&db, rowid, output)?;
                    embedded += 1;
                    progress.inc(1);
                }
                Ok(())
            })();
            if let Err(error) = batch_result {
                pipeline_error = Some(error);
                break;
            }
        }
        drop(prepared_rx);
        match pipeline_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    })?;
    progress.finish();
    db.checkpoint();
    println!(
        "embedded {embedded} images, {failed} failed; run `warp-cli clipindex` to update the index"
    );
    Ok(())
}

pub fn update_index() -> Result<()> {
    let db = DB::new_fast(PathBuf::from(DATABASE))?;
    let pending = witchcraft::count_unindexed_embeddings(&db)?;
    let unavailable = witchcraft::semantic_index_unavailable_reason(&db)?;
    if pending == 0 && unavailable.is_none() {
        println!("pooled MobileCLIP2 IVF index is already up to date");
        return Ok(());
    }
    if pending > 0 {
        println!("adding {pending} pooled embeddings to {DATABASE}");
    } else {
        println!("building pooled MobileCLIP2 IVF index for {DATABASE}");
    }
    witchcraft::index_chunks_with_options(
        &db,
        None,
        false,
        IndexOptions::new(4)?.force_flush(),
    )?;
    println!("pooled MobileCLIP2 IVF index is ready");
    Ok(())
}

pub fn query(
    assets: &Path,
    text: &str,
    limit: usize,
    exact: bool,
) -> Result<Vec<(f32, String, Vec<String>, u32, String)>> {
    let db = DB::new_reader(PathBuf::from(DATABASE))?;
    if !exact {
        if let Some(reason) = witchcraft::semantic_index_unavailable_reason(&db)? {
            anyhow::bail!("pooled MobileCLIP2 IVF index is not ready: {reason}; run clipindex");
        }
        if witchcraft::count_unindexed_embeddings(&db)? > 0 {
            anyhow::bail!("pooled MobileCLIP2 embeddings are not indexed; run clipindex");
        }
    }
    let device = witchcraft::make_device();
    let model = witchcraft::mobileclip2::StockMobileClip2::load(&device, assets)?;
    let query = model.embed_text(text)?;
    if exact {
        witchcraft::exact_search_embeddings(&db, &query, limit)
    } else {
        witchcraft::search_embeddings(&db, &query, None, 0.0, limit, None, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_embedding_query_only_returns_uncached_documents() {
        let temporary = tempfile::tempdir().unwrap();
        let db = DB::new_fast(temporary.path().join("images.sqlite")).unwrap();
        let image = temporary.path().join("image.jpg");
        std::fs::write(&image, []).unwrap();
        let mut db = db;
        let (rowid, inserted) = witchcraft::register_image_document_auto(&mut db, &image).unwrap();
        assert!(inserted);
        let (same_rowid, inserted) =
            witchcraft::register_image_document_auto(&mut db, &image).unwrap();
        assert_eq!(same_rowid, rowid);
        assert!(!inserted);
        assert_eq!(missing_embeddings(&db).unwrap(), vec![(rowid, image.canonicalize().unwrap())]);

        let output = Tensor::ones((1, 1, 512), candle_core::DType::F32, &candle_core::Device::Cpu)
            .unwrap();
        witchcraft::cache_precomputed_l2_int8_image_embeddings(&db, rowid, output).unwrap();
        assert!(missing_embeddings(&db).unwrap().is_empty());
    }

    #[test]
    fn experiment_uses_an_isolated_database() {
        assert_ne!(DATABASE, "mydb.sqlite");
        assert!(DATABASE.contains("mobileclip2"));
    }

    #[test]
    fn registration_skips_bad_headers_before_allocating_rowids() {
        let temporary = tempfile::tempdir().unwrap();
        let invalid = temporary.path().join("invalid.png");
        std::fs::write(&invalid, b"not an image").unwrap();
        let gif_with_jpeg_extension = temporary.path().join("animation.jpg");
        std::fs::write(
            &gif_with_jpeg_extension,
            b"GIF89a\x01\x00\x01\x00\x00\x00\x00",
        )
        .unwrap();
        let valid = temporary.path().join("valid.png");
        let mut jpeg = Vec::new();
        image::codecs::jpeg::JpegEncoder::new(&mut jpeg)
            .encode(&[255, 0, 0], 1, 1, image::ExtendedColorType::Rgb8)
            .unwrap();
        std::fs::write(&valid, jpeg).unwrap();
        let mut db = DB::new_fast(temporary.path().join("images.sqlite")).unwrap();

        let counts = register_paths(
            &mut db,
            &[invalid, gif_with_jpeg_extension, valid.clone()],
        )
        .unwrap();

        assert_eq!(counts, (1, 0, 2));
        let mut query = db.query("SELECT rowid, body FROM document").unwrap();
        let row = query
            .query_row((), |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)))
            .unwrap();
        assert_eq!(row.0, 1);
        assert_eq!(PathBuf::from(row.1), valid.canonicalize().unwrap());
    }
}
