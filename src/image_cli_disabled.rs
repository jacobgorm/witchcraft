use anyhow::Result;
use std::path::{Path, PathBuf};

const IMAGE_COMMANDS: &[&str] = &[
    "imagetags",
    "embedimage",
    "embedimages",
    "clipthumbnails",
    "image",
    "benchimage",
    "scoreimage",
    "queryimage",
];

pub fn run(args: &[String], _assets: &Path, _db_name: &Path) -> Result<bool> {
    let Some(command) = args.get(1).map(String::as_str) else {
        return Ok(false);
    };
    if IMAGE_COMMANDS.contains(&command) {
        anyhow::bail!("{command} requires building with --features mobileclip2");
    }
    Ok(false)
}

pub fn display_image_grid(_paths: &[PathBuf]) {}
