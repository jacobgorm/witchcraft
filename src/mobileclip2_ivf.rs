use anyhow::Result;
use candle_core::{Device, Tensor};
use std::path::{Path, PathBuf};
use witchcraft::{DB, IndexOptions};

const DATABASE: &str = "mobileclip2-ivf.sqlite";
const DIMENSIONS: usize = 512;

pub fn build(matrix: &Path, force: bool) -> Result<()> {
    let mut db = DB::new_fast(PathBuf::from(DATABASE))?;
    let mut imported = 0usize;
    let mut skipped = 0usize;
    let total = super::mobileclip2_matrix::for_each_row(matrix, |path, row| {
        if !force {
            if let Some(rowid) = witchcraft::cached_image_rowid(&db, &path)? {
                skipped += 1;
                println!("skipping rowid {rowid}: {}", path.display());
                return Ok(());
            }
        }
        let output = Tensor::from_vec(row, (1, 1, DIMENSIONS), &Device::Cpu)?;
        let (rowid, vectors) = witchcraft::cache_precomputed_image_embeddings_auto(
            &mut db,
            &path,
            output,
        )?;
        println!("rowid {rowid}: {vectors} pooled 512d vector for {}", path.display());
        imported += 1;
        Ok(())
    })?;
    println!("read {total} matrix rows; imported {imported}, skipped {skipped}");

    println!("rebuilding {DATABASE} with 4-bit polar residuals");
    witchcraft::index_chunks_with_options(
        &db,
        None,
        true,
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
            anyhow::bail!("pooled MobileCLIP2 IVF index is not ready: {reason}; run clipivfbuild");
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
    fn experiment_uses_an_isolated_database() {
        assert_ne!(DATABASE, "mydb.sqlite");
        assert!(DATABASE.contains("mobileclip2"));
    }
}
