use anyhow::Result;
use candle_core::Tensor;
use chrono::{Duration, NaiveDateTime};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use witchcraft::{DB, IndexOptions};

const DATABASE: &str = "mobileclip2-ivf.sqlite";
const BATCH_SIZE: usize = 8;
const RESIDUAL_QUANT_BITS: u8 = 4;
const DEFAULT_SCAN_BUDGET: usize = 5_000;
type SearchResult = (f32, String, Vec<String>, u32, String);

pub fn registered_paths() -> Result<Vec<PathBuf>> {
    let db = DB::new_reader(PathBuf::from(DATABASE))?;
    let mut query = db.query("SELECT body FROM document ORDER BY rowid")?;
    let rows = query.query_map((), |row| row.get::<_, String>(0))?;
    rows.map(|row| Ok(PathBuf::from(row?))).collect()
}

pub fn enrich_tags(assets: &Path) -> Result<()> {
    let db = DB::new_fast(PathBuf::from(DATABASE))?;
    let pending = {
        let mut query = db.query(
            "SELECT rowid, body FROM document
             WHERE COALESCE(json_extract(metadata, '$.exif_scanned'), 0) = 0
             ORDER BY rowid",
        )?;
        let rows = query.query_map((), |row| {
            Ok((row.get::<_, i64>(0)?, PathBuf::from(row.get::<_, String>(1)?)))
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()?
    };
    if pending.is_empty() {
        println!("all registered images already have EXIF tags");
        return Ok(());
    }

    let progress = witchcraft::progress::new_with_label(pending.len() as u64, "extract EXIF tags");
    let mut tagged = 0usize;
    let mut failed = 0usize;
    for (rowid, path) in pending {
        match witchcraft::image_tags::extract(&path, assets) {
            Ok(tags) => {
                witchcraft::update_registered_image_tags(&db, rowid.try_into()?, &path, &tags)?;
                tagged += 1;
            },
            Err(error) => {
                failed += 1;
                eprintln!("[WARN] could not extract EXIF tags from {}: {error}", path.display());
            },
        }
        progress.inc(1);
    }
    progress.finish();
    db.checkpoint();
    println!("tagged {tagged} registered images, {failed} failed");
    Ok(())
}

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
        IndexOptions::new(RESIDUAL_QUANT_BITS)?.force_flush(),
    )?;
    println!("pooled MobileCLIP2 IVF index is ready");
    Ok(())
}

const CLUSTER_MAX_GAP_MINUTES: i64 = 30;
const CLUSTER_STATIONARY_MAX_GAP_MINUTES: i64 = 60;
const CLUSTER_MAX_SPAN_MINUTES: i64 = 120;
const CLUSTER_STATIONARY_DISTANCE_KM: f64 = 1.0;
const CLUSTER_MAX_DISTANCE_KM: f64 = 10.0;

#[derive(Debug)]
struct ClusterMember {
    rowid: u64,
    body: String,
    metadata: String,
    captured_at: Option<NaiveDateTime>,
    latitude: Option<f64>,
    longitude: Option<f64>,
    country: String,
    region: String,
    town: String,
}

fn cluster_member(rowid: u64, metadata: String, body: String) -> ClusterMember {
    let parsed = serde_json::from_str::<serde_json::Value>(&metadata).unwrap_or_default();
    let image = parsed.get("image").unwrap_or(&serde_json::Value::Null);
    let string_field = |name| {
        image
            .get(name)
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .to_owned()
    };
    let captured_at = image
        .get("captured_at")
        .and_then(serde_json::Value::as_str)
        .and_then(|value| NaiveDateTime::parse_from_str(value, "%Y:%m:%d %H:%M:%S").ok());
    ClusterMember {
        rowid,
        body,
        metadata,
        captured_at,
        latitude: image.get("latitude").and_then(serde_json::Value::as_f64),
        longitude: image.get("longitude").and_then(serde_json::Value::as_f64),
        country: string_field("country_code"),
        region: string_field("region"),
        town: string_field("nearest_town"),
    }
}

fn distance_km(left: &ClusterMember, right: &ClusterMember) -> Option<f64> {
    let (lat1, lon1, lat2, lon2) = (
        left.latitude?,
        left.longitude?,
        right.latitude?,
        right.longitude?,
    );
    let delta_lat = (lat2 - lat1).to_radians();
    let delta_lon = (lon2 - lon1).to_radians();
    let lat1 = lat1.to_radians();
    let lat2 = lat2.to_radians();
    let haversine = (delta_lat / 2.0).sin().powi(2)
        + lat1.cos() * lat2.cos() * (delta_lon / 2.0).sin().powi(2);
    Some(6_371.0 * 2.0 * haversine.sqrt().asin())
}

fn same_event(previous: &ClusterMember, anchor: &ClusterMember, current: &ClusterMember) -> bool {
    let (Some(previous_time), Some(anchor_time), Some(current_time)) = (
        previous.captured_at,
        anchor.captured_at,
        current.captured_at,
    ) else {
        return false;
    };
    let distance = distance_km(anchor, current);
    let max_gap_minutes = if distance.is_some_and(|distance| {
        distance <= CLUSTER_STATIONARY_DISTANCE_KM
    }) {
        CLUSTER_STATIONARY_MAX_GAP_MINUTES
    } else {
        CLUSTER_MAX_GAP_MINUTES
    };
    if previous_time.date() != current_time.date()
        || current_time.signed_duration_since(previous_time)
            > Duration::minutes(max_gap_minutes)
        || current_time.signed_duration_since(anchor_time)
            > Duration::minutes(CLUSTER_MAX_SPAN_MINUTES)
        || current_time < previous_time
    {
        return false;
    }
    if let Some(distance) = distance {
        return distance <= CLUSTER_MAX_DISTANCE_KM;
    }
    if !anchor.town.is_empty() && !current.town.is_empty() {
        return anchor.country == current.country
            && anchor.region == current.region
            && anchor.town == current.town;
    }
    anchor_time.date() == current_time.date()
}

type ClusterMembers = HashMap<String, Vec<(u64, String, String)>>;

fn sessionize_members(mut records: Vec<ClusterMember>) -> (HashMap<String, String>, ClusterMembers) {
    records.sort_unstable_by_key(|record| (record.captured_at, record.rowid));
    let mut sessions = Vec::<Vec<ClusterMember>>::new();
    for record in records {
        let joins_last = sessions.last().is_some_and(|session| {
            same_event(session.last().unwrap(), &session[0], &record)
        });
        if joins_last {
            sessions.last_mut().unwrap().push(record);
        } else {
            sessions.push(vec![record]);
        }
    }

    let mut body_keys = HashMap::new();
    let mut members = HashMap::new();
    let mut event_number_by_date = HashMap::<String, usize>::new();
    for session in sessions {
        let first = &session[0];
        let key = if let Some(captured_at) = first.captured_at {
            let date = captured_at.format("%Y:%m:%d").to_string();
            let event_number = event_number_by_date.entry(date.clone()).or_default();
            *event_number += 1;
            format!(
                "event:{date}:{event_number:03}|{}|{}|{}",
                first.country, first.region, first.town
            )
        } else {
            format!("image:{}", first.body)
        };
        let mut cluster = Vec::with_capacity(session.len());
        for record in session {
            body_keys.insert(record.body.clone(), key.clone());
            cluster.push((record.rowid, record.body, record.metadata));
        }
        members.insert(key, cluster);
    }
    (body_keys, members)
}

fn cluster_results(
    db: &DB,
    query_embedding: &Tensor,
    results: Vec<SearchResult>,
    limit: usize,
) -> Result<Vec<SearchResult>> {
    let (body_keys, mut members) = {
        let mut records = Vec::new();
        let mut query = db.query("SELECT rowid, metadata, body FROM document ORDER BY rowid")?;
        let rows = query.query_map((), |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;
        for row in rows {
            let (rowid, metadata, body) = row?;
            records.push(cluster_member(rowid.try_into()?, metadata, body));
        }
        sessionize_members(records)
    };

    let mut winners = HashMap::<String, SearchResult>::new();
    let mut known_scores = HashMap::<String, f32>::new();
    for result in results {
        let index = (result.3 as usize).min(result.2.len().saturating_sub(1));
        let body = result.2.get(index).cloned().unwrap_or_default();
        known_scores.insert(body.clone(), result.0);
        let key = body_keys
            .get(&body)
            .cloned()
            .unwrap_or_else(|| format!("image:{body}"));
        let replace = winners.get(&key).is_none_or(|winner| result.0 > winner.0);
        if replace {
            winners.insert(key, result);
        }
    }

    let mut winners = winners.into_iter().collect::<Vec<_>>();
    winners.sort_unstable_by(|(_, left), (_, right)| right.0.total_cmp(&left.0));
    let query_values = query_embedding
        .flatten_all()?
        .to_device(&candle_core::Device::Cpu)?
        .to_vec1::<f32>()?;
    let mut clustered = winners
        .into_iter()
        .map(|(key, mut result)| {
            let mut cluster_members = members.remove(&key).unwrap_or_default();
            let mut scored_members = cluster_members
                .drain(..)
                .map(|(rowid, body, metadata)| {
                    let score = if let Some(score) = known_scores.get(&body) {
                        *score
                    } else {
                        let embedding = witchcraft::cached_document_embedding_tensor(db, rowid)?
                            .ok_or_else(|| anyhow::anyhow!("missing pooled MobileCLIP2 embedding for {body}"))?;
                        let values = embedding.flatten_all()?.to_vec1::<f32>()?;
                        anyhow::ensure!(
                            values.len() == query_values.len(),
                            "image {body} has {} embedding values, expected {}",
                            values.len(),
                            query_values.len()
                        );
                        query_values
                            .iter()
                            .zip(values)
                            .map(|(query, image)| query * image)
                            .sum()
                    };
                    Ok((score, body, metadata))
                })
                .collect::<Result<Vec<_>>>()?;
            scored_members.sort_unstable_by(|left, right| right.0.total_cmp(&left.0));
            let member_scores = scored_members
                .iter()
                .map(|(score, _, _)| *score)
                .collect::<Vec<_>>();
            let winning_metadata = scored_members
                .first()
                .map(|(_, _, metadata)| metadata.as_str())
                .unwrap_or(&result.1);
            let cluster_members = scored_members
                .iter()
                .map(|(_, body, _)| body.clone())
                .collect::<Vec<_>>();
            let mut metadata = serde_json::from_str::<serde_json::Value>(winning_metadata)
                .unwrap_or_else(|_| serde_json::json!({}));
            metadata["cluster"] = serde_json::json!({
                "key": key,
                "size": cluster_members.len(),
                "member_scores": member_scores,
            });
            result.1 = metadata.to_string();
            result.2 = cluster_members;
            result.3 = 0;
            if let Some(score) = member_scores.first() {
                result.0 = *score;
            }
            Ok(result)
        })
        .collect::<Result<Vec<_>>>()?;
    clustered.sort_unstable_by(|left, right| right.0.total_cmp(&left.0));
    clustered.truncate(limit);
    Ok(clustered)
}

pub fn query(
    assets: &Path,
    text: &str,
    limit: usize,
    exact: bool,
    clustered: bool,
    scan_budget: Option<usize>,
) -> Result<Vec<SearchResult>> {
    let db = DB::new_reader(PathBuf::from(DATABASE))?;
    if !exact {
        if let Some(reason) = witchcraft::semantic_index_unavailable_reason(&db)? {
            anyhow::bail!("pooled MobileCLIP2 IVF index is not ready: {reason}; run clipindex");
        }
        if witchcraft::count_unindexed_embeddings(&db)? > 0 {
            anyhow::bail!("pooled MobileCLIP2 embeddings are not indexed; run clipindex");
        }
    }
    if clustered {
        let missing: i64 = db
            .query(
                "SELECT count(*) FROM document
                 WHERE COALESCE(json_extract(metadata, '$.exif_scanned'), 0) = 0",
            )?
            .query_row((), |row| row.get(0))?;
        anyhow::ensure!(missing == 0, "{missing} images lack EXIF tags; run cliptag first");
    }
    let device = witchcraft::make_device();
    let model = witchcraft::mobileclip2::StockMobileClip2::load(&device, assets)?;
    let query = model.embed_text(text)?;
    let candidate_limit = if clustered && exact {
        db.query("SELECT count(*) FROM document")?
            .query_row((), |row| row.get::<_, i64>(0))?
            .try_into()?
    } else if clustered {
        (limit * 100).max(1_000)
    } else {
        limit
    };
    let effective_scan_budget = (!exact).then_some(scan_budget.unwrap_or(DEFAULT_SCAN_BUDGET));
    let results = witchcraft::with_vector_scan_budget(effective_scan_budget, || {
        if exact {
            witchcraft::exact_search_embeddings(&db, &query, candidate_limit)
        } else {
            witchcraft::search_embeddings(&db, &query, None, 0.0, candidate_limit, None, false)
        }
    })?;
    if clustered {
        cluster_results(&db, &query, results, limit)
    } else {
        Ok(results)
    }
}

pub fn nearest_neighbors(paths: &[PathBuf], limit: usize) -> Result<Vec<Vec<PathBuf>>> {
    let db = DB::new_reader(PathBuf::from(DATABASE))?;
    if let Some(reason) = witchcraft::semantic_index_unavailable_reason(&db)? {
        anyhow::bail!("pooled MobileCLIP2 IVF index is not ready: {reason}; run clipindex");
    }
    let device = witchcraft::make_device();
    // The preceding text query may have populated the process-wide centroid
    // cache using a different Candle Device handle for the same GPU.
    witchcraft::clear_generations_cache();
    let mut all_neighbors = Vec::with_capacity(paths.len());
    for path in paths {
        let rowid = witchcraft::cached_image_rowid(&db, path)?
            .ok_or_else(|| anyhow::anyhow!("no pooled MobileCLIP2 embedding for {}", path.display()))?;
        let query = witchcraft::cached_document_embedding_tensor(&db, rowid)?
            .ok_or_else(|| anyhow::anyhow!("missing pooled MobileCLIP2 embedding for {}", path.display()))?
            .to_device(&device)?;
        let source = witchcraft::normalize_image_path(path);
        let results = witchcraft::with_vector_scan_budget(Some(DEFAULT_SCAN_BUDGET), || {
            witchcraft::search_embeddings(&db, &query, None, 0.0, limit + 1, None, false)
        })?;
        let mut neighbors = Vec::with_capacity(limit);
        for (_, _, bodies, sub_idx, _) in results {
            let index = (sub_idx as usize).min(bodies.len().saturating_sub(1));
            let candidate = witchcraft::normalize_image_path(Path::new(&bodies[index]));
            if candidate != source && !neighbors.contains(&candidate) {
                neighbors.push(candidate);
            }
            if neighbors.len() == limit {
                break;
            }
        }
        all_neighbors.push(neighbors);
    }
    Ok(all_neighbors)
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
    fn exif_clusters_split_on_time_gaps() {
        let record = |rowid, body: &str, captured_at: &str| {
            cluster_member(
                rowid,
                format!(r#"{{"image":{{"captured_at":"{captured_at}"}}}}"#),
                body.to_owned(),
            )
        };
        let (keys, _) = sessionize_members(vec![
            record(1, "one", "2023:06:14 07:00:00"),
            record(2, "two", "2023:06:14 07:25:00"),
            record(3, "three", "2023:06:14 08:00:00"),
        ]);

        assert_eq!(keys["one"], keys["two"]);
        assert_ne!(keys["two"], keys["three"]);
    }

    #[test]
    fn exif_clusters_limit_total_session_span_despite_short_gaps() {
        let record = |rowid, body: &str, captured_at: &str| {
            cluster_member(
                rowid,
                format!(r#"{{"image":{{"captured_at":"{captured_at}"}}}}"#),
                body.to_owned(),
            )
        };
        let (keys, _) = sessionize_members(vec![
            record(1, "start", "2023:06:14 07:00:00"),
            record(2, "middle-one", "2023:06:14 07:30:00"),
            record(3, "middle-two", "2023:06:14 08:00:00"),
            record(4, "middle-three", "2023:06:14 08:30:00"),
            record(5, "edge", "2023:06:14 09:00:00"),
            record(6, "past-edge", "2023:06:14 09:30:00"),
        ]);

        assert_eq!(keys["start"], keys["edge"]);
        assert_ne!(keys["edge"], keys["past-edge"]);
    }

    #[test]
    fn exif_clusters_split_on_distance_and_keep_undated_images_singleton() {
        let record = |rowid, body: &str, captured_at: Option<&str>, latitude, longitude| {
            let captured_at = captured_at
                .map(|value| format!(r#""captured_at":"{value}","#))
                .unwrap_or_default();
            cluster_member(
                rowid,
                format!(
                    r#"{{"image":{{{captured_at}"latitude":{latitude},"longitude":{longitude}}}}}"#
                ),
                body.to_owned(),
            )
        };
        let (keys, _) = sessionize_members(vec![
            record(1, "one", Some("2023:06:14 07:00:00"), 56.46, 10.04),
            record(2, "two", Some("2023:06:14 07:10:00"), 56.16, 10.20),
            record(3, "undated-one", None, 56.16, 10.20),
            record(4, "undated-two", None, 56.16, 10.20),
        ]);

        assert_ne!(keys["one"], keys["two"]);
        assert_ne!(keys["undated-one"], keys["undated-two"]);
    }

    #[test]
    fn exif_clusters_allow_longer_gaps_only_while_stationary() {
        let record = |rowid, body: &str, captured_at: &str, latitude| {
            cluster_member(
                rowid,
                format!(
                    r#"{{"image":{{"captured_at":"{captured_at}","latitude":{latitude},"longitude":10.0}}}}"#
                ),
                body.to_owned(),
            )
        };
        let (keys, _) = sessionize_members(vec![
            record(1, "anchor", "2023:06:14 07:00:00", 56.0),
            record(2, "stationary", "2023:06:14 07:50:00", 56.005),
            record(3, "moved", "2023:06:14 08:35:00", 56.02),
        ]);

        assert_eq!(keys["anchor"], keys["stationary"]);
        assert_ne!(keys["stationary"], keys["moved"]);
    }

    #[test]
    fn cluster_members_are_sorted_by_clip_score() {
        let temporary = tempfile::tempdir().unwrap();
        let mut db = DB::new_fast(temporary.path().join("images.sqlite")).unwrap();
        let first = temporary.path().join("first.jpg");
        let winner = temporary.path().join("winner.jpg");
        std::fs::write(&first, []).unwrap();
        std::fs::write(&winner, []).unwrap();
        let (first_rowid, _) = witchcraft::register_image_document_auto(&mut db, &first).unwrap();
        let (winner_rowid, _) = witchcraft::register_image_document_auto(&mut db, &winner).unwrap();
        let first_metadata = r#"{"name":"first","image":{"captured_at":"2020:01:02 10:00:00"}}"#;
        let winner_metadata = r#"{"name":"winner","image":{"captured_at":"2020:01:02 10:20:00"}}"#;
        db.query("UPDATE document SET metadata = ?1 WHERE rowid = ?2")
            .unwrap()
            .execute((first_metadata, i64::try_from(first_rowid).unwrap()))
            .unwrap();
        db.query("UPDATE document SET metadata = ?1 WHERE rowid = ?2")
            .unwrap()
            .execute((winner_metadata, i64::try_from(winner_rowid).unwrap()))
            .unwrap();

        let mut first_values = vec![0.0f32; 512];
        first_values[0] = 1.0;
        let mut winner_values = vec![0.0f32; 512];
        winner_values[1] = 1.0;
        let first_embedding = Tensor::from_vec(
            first_values,
            (1, 1, 512),
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let winner_embedding = Tensor::from_vec(
            winner_values.clone(),
            (1, 1, 512),
            &candle_core::Device::Cpu,
        )
        .unwrap();
        witchcraft::cache_precomputed_l2_int8_image_embeddings(&db, first_rowid, first_embedding)
            .unwrap();
        witchcraft::cache_precomputed_l2_int8_image_embeddings(&db, winner_rowid, winner_embedding)
            .unwrap();

        let query = Tensor::from_vec(winner_values, (1, 512), &candle_core::Device::Cpu).unwrap();
        let results = vec![(
            0.0,
            first_metadata.to_owned(),
            vec![first.canonicalize().unwrap().to_string_lossy().into_owned()],
            0,
            String::new(),
        )];
        let clustered = cluster_results(&db, &query, results, 10).unwrap();

        assert_eq!(clustered[0].2[0], winner.canonicalize().unwrap().to_string_lossy());
        assert_eq!(clustered[0].3, 0);
        assert!(clustered[0].0 > 0.99);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&clustered[0].1).unwrap()["name"],
            "winner"
        );
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
