//! MobileCLIP2-S0 image and text encoder using Candle.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};
use candle_transformers::models::{fastvit, openclip::text_model};
use serde::Deserialize;
use std::io::Error;
use std::path::Path;
#[cfg(not(feature = "embed-assets"))]
use std::path::PathBuf;
use tokenizers::Tokenizer;

use crate::profiled_fastvit::ProfiledFastVit;
pub use crate::profiled_fastvit::{ImageProfile, ProfileRecord};

#[cfg(feature = "embed-assets")]
use crate::embed_asset;

#[cfg(feature = "embed-assets")]
embed_asset!(pub CONFIG, "mobileclip2-config.json");
#[cfg(feature = "embed-assets")]
embed_asset!(pub TOKENIZER, "mobileclip2-tokenizer.json");
#[cfg(feature = "embed-assets")]
embed_asset!(pub MODEL, "mobileclip2.safetensors");

const DEFAULT_CONTEXT_LENGTH: usize = 77;
const DEFAULT_EMBED_DIM: usize = 512;
const DEFAULT_TEXT_WIDTH: usize = 512;
const DEFAULT_TEXT_HEADS: usize = 8;
const DEFAULT_TEXT_LAYERS: usize = 12;
const DEFAULT_VOCAB_SIZE: usize = 49408;

#[cfg(not(feature = "embed-assets"))]
const CONFIG_ENV: &str = "WARP_MOBILECLIP2_CONFIG";
#[cfg(not(feature = "embed-assets"))]
const TOKENIZER_ENV: &str = "WARP_MOBILECLIP2_TOKENIZER";
#[cfg(not(feature = "embed-assets"))]
const WEIGHTS_ENV: &str = "WARP_MOBILECLIP2_WEIGHTS";

#[cfg(not(feature = "embed-assets"))]
const CONFIG_CANDIDATES: &[&str] = &["mobileclip2-config.json", "open_clip_config.json"];
#[cfg(not(feature = "embed-assets"))]
const TOKENIZER_CANDIDATES: &[&str] = &["mobileclip2-tokenizer.json", "open_clip_tokenizer.json"];
#[cfg(not(feature = "embed-assets"))]
const SAFETENSOR_CANDIDATES: &[&str] = &["mobileclip2.safetensors", "open_clip_model.safetensors"];
#[cfg(not(feature = "embed-assets"))]
const PTH_CANDIDATES: &[&str] = &["mobileclip2_s0.pt", "open_clip_pytorch_model.bin"];

#[derive(Debug, Clone, Deserialize)]
struct Config {
    #[serde(default = "default_embed_dim")]
    embed_dim: usize,
    #[serde(default)]
    vision_cfg: VisionConfig,
    text_cfg: TextConfig,
}

#[derive(Debug, Clone, Deserialize)]
struct VisionConfig {
    #[serde(default = "default_vision_model_name")]
    timm_model_name: String,
    #[serde(default = "default_image_size")]
    image_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct TextConfig {
    #[serde(default = "default_context_length")]
    context_length: usize,
    #[serde(default = "default_vocab_size")]
    vocab_size: usize,
    #[serde(default = "default_text_width")]
    width: usize,
    #[serde(default = "default_text_heads")]
    heads: usize,
    #[serde(default = "default_text_layers")]
    layers: usize,
}

fn default_context_length() -> usize {
    DEFAULT_CONTEXT_LENGTH
}

fn default_embed_dim() -> usize {
    DEFAULT_EMBED_DIM
}

fn default_image_size() -> usize {
    256
}

fn default_text_width() -> usize {
    DEFAULT_TEXT_WIDTH
}

fn default_text_heads() -> usize {
    DEFAULT_TEXT_HEADS
}

fn default_text_layers() -> usize {
    DEFAULT_TEXT_LAYERS
}

fn default_vocab_size() -> usize {
    DEFAULT_VOCAB_SIZE
}

fn default_vision_model_name() -> String {
    "fastvit_mci0".to_string()
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            timm_model_name: default_vision_model_name(),
            image_size: default_image_size(),
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            embed_dim: DEFAULT_EMBED_DIM,
            vision_cfg: VisionConfig::default(),
            text_cfg: TextConfig {
                context_length: DEFAULT_CONTEXT_LENGTH,
                vocab_size: DEFAULT_VOCAB_SIZE,
                width: DEFAULT_TEXT_WIDTH,
                heads: DEFAULT_TEXT_HEADS,
                layers: DEFAULT_TEXT_LAYERS,
            },
        }
    }
}

impl Config {
    fn from_json(bytes: &[u8]) -> candle_core::Result<Self> {
        let value: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|e| Error::other(format!("failed to parse MobileCLIP2 config: {e}")))?;
        let model_cfg = value.get("model_cfg").unwrap_or(&value);
        serde_json::from_value(model_cfg.clone())
            .map_err(|e| Error::other(format!("failed to parse MobileCLIP2 model_cfg: {e}")).into())
    }

    fn text_model_config(&self) -> text_model::Config {
        let width = self.text_cfg.width;
        text_model::Config {
            vocab_size: self.text_cfg.vocab_size,
            embed_dim: width,
            intermediate_size: width * 4,
            max_position_embeddings: self.text_cfg.context_length,
            pad_with: None,
            num_hidden_layers: self.text_cfg.layers,
            num_attention_heads: self.text_cfg.heads,
            projection_dim: self.embed_dim,
        }
    }

    fn vision_model_config(&self) -> candle_core::Result<fastvit::Config> {
        match self.vision_cfg.timm_model_name.as_str() {
            "fastvit_mci0" => Ok(fastvit::Config::mci0()),
            "fastvit_mci1" => Ok(fastvit::Config::mci1()),
            "fastvit_mci2" => Ok(fastvit::Config::mci2()),
            name => {
                Err(Error::other(format!("unsupported MobileCLIP2 vision model `{name}`")).into())
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct T5EncoderModel {
    text_model: text_model::OpenClipTextTransformer,
    vision_model: ProfiledFastVit,
    projection: Linear,
    context_length: usize,
    image_size: usize,
    device: Device,
}

impl T5EncoderModel {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let text_config = cfg.text_model_config();
        let text_model = text_model::OpenClipTextTransformer::new(vb.pp("text"), &text_config)?;
        let vision_config = cfg.vision_model_config()?;
        let vision_model =
            ProfiledFastVit::new(&vision_config, cfg.embed_dim, vb.pp("visual.trunk"))?;
        let text_projection = vb.get(
            (text_config.embed_dim, text_config.projection_dim),
            "text.text_projection",
        )?;
        let projection = Linear::new(text_projection.t()?, None);
        Ok(Self {
            text_model,
            vision_model,
            projection,
            context_length: text_config.max_position_embeddings,
            image_size: cfg.vision_cfg.image_size,
            device: vb.device().clone(),
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_len = input_ids.dim(candle_core::D::Minus1)?;
        if seq_len > self.context_length {
            candle_core::bail!(
                "MobileCLIP2-S0 input has {seq_len} tokens, context length is {}",
                self.context_length
            );
        }
        let hidden = self.text_model.forward(input_ids)?;
        self.projection.forward(&hidden)
    }

    pub fn forward_text_pooled(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_len = input_ids.dim(candle_core::D::Minus1)?;
        if seq_len > self.context_length {
            candle_core::bail!(
                "MobileCLIP2-S0 input has {seq_len} tokens, context length is {}",
                self.context_length
            );
        }
        let pooled = input_ids.apply(&self.text_model)?;
        self.projection.forward(&pooled)
    }

    pub fn forward_image(&self, pixel_values: &Tensor) -> Result<Tensor> {
        self.vision_model.forward(pixel_values)
    }

    pub fn forward_image_profiled(&self, pixel_values: &Tensor) -> Result<(Tensor, ImageProfile)> {
        self.vision_model.forward_profiled(pixel_values)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn max_seq_len(&self) -> usize {
        self.context_length
    }

    pub fn image_size(&self) -> usize {
        self.image_size
    }
}

pub struct T5ModelBuilder {
    config: Config,
}

impl T5ModelBuilder {
    pub fn load(assets: &Path) -> candle_core::Result<(Self, Tokenizer)> {
        #[cfg(feature = "embed-assets")]
        let config = Config::from_json(
            CONFIG
                .bytes(assets)
                .map_err(|_| Error::other("failed to read embedded mobileclip2-config.json"))?,
        )?;

        #[cfg(not(feature = "embed-assets"))]
        let config = match find_asset_path(CONFIG_ENV, CONFIG_CANDIDATES, assets) {
            Some(path) => Config::from_json(&std::fs::read(&path).map_err(|e| {
                Error::other(format!(
                    "failed to read MobileCLIP2 config {}: {e}",
                    path.display()
                ))
            })?)?,
            None => Config::default(),
        };

        #[cfg(feature = "embed-assets")]
        let tokenizer = Tokenizer::from_bytes(
            TOKENIZER
                .bytes(assets)
                .map_err(|_| Error::other("failed to read embedded mobileclip2-tokenizer.json"))?,
        )
        .map_err(|e| {
            Error::other(format!(
                "failed to parse embedded MobileCLIP2 tokenizer: {e}"
            ))
        })?;

        #[cfg(not(feature = "embed-assets"))]
        let tokenizer_path = find_asset_path(TOKENIZER_ENV, TOKENIZER_CANDIDATES, assets)
            .ok_or_else(|| {
                Error::other(format!(
                    "missing MobileCLIP2 tokenizer; expected one of {} under {} or set {TOKENIZER_ENV}",
                    TOKENIZER_CANDIDATES.join(", "),
                    assets.display()
                ))
            })?;

        #[cfg(not(feature = "embed-assets"))]
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            Error::other(format!(
                "failed to parse MobileCLIP2 tokenizer {}: {e}",
                tokenizer_path.display()
            ))
        })?;

        Ok((Self { config }, tokenizer))
    }

    pub fn build_encoder(
        &self,
        device: &Device,
        assets: &Path,
    ) -> candle_core::Result<T5EncoderModel> {
        let vb = load_weights(device, assets)?;
        T5EncoderModel::load(vb, &self.config).map_err(|e| {
            Error::other(format!("failed to load MobileCLIP2-S0 text encoder: {e}")).into()
        })
    }
}

#[cfg(not(feature = "embed-assets"))]
fn find_asset_path(env_var: &str, candidates: &[&str], assets: &Path) -> Option<PathBuf> {
    if let Some(path) = find_env_path(env_var) {
        return Some(path);
    }
    find_candidate_path(candidates, assets)
}

#[cfg(not(feature = "embed-assets"))]
fn find_env_path(env_var: &str) -> Option<PathBuf> {
    std::env::var_os(env_var)
        .map(PathBuf::from)
        .filter(|path| path.is_file())
}

#[cfg(not(feature = "embed-assets"))]
fn find_candidate_path(candidates: &[&str], assets: &Path) -> Option<PathBuf> {
    candidates
        .iter()
        .map(|name| assets.join(name))
        .find(|path| path.is_file())
}

#[cfg(feature = "embed-assets")]
fn load_weights(device: &Device, assets: &Path) -> candle_core::Result<VarBuilder<'static>> {
    let model_bytes = MODEL
        .bytes(assets)
        .map_err(|_| Error::other("failed to read embedded mobileclip2.safetensors"))?;
    VarBuilder::from_slice_safetensors(model_bytes, DType::F32, device)
}

#[cfg(not(feature = "embed-assets"))]
fn load_weights(device: &Device, assets: &Path) -> candle_core::Result<VarBuilder<'static>> {
    if let Some(path) = find_env_path(WEIGHTS_ENV) {
        return load_weight_path(path, device);
    }
    if let Some(path) = find_candidate_path(SAFETENSOR_CANDIDATES, assets) {
        return load_safetensors_path(path, device);
    }
    if let Some(path) = find_candidate_path(PTH_CANDIDATES, assets) {
        return load_pth_path(path, device);
    }

    Err(Error::other(format!(
        "missing MobileCLIP2 weights; expected one of {}, {} under {} or set {WEIGHTS_ENV}",
        SAFETENSOR_CANDIDATES.join(", "),
        PTH_CANDIDATES.join(", "),
        assets.display()
    ))
    .into())
}

#[cfg(not(feature = "embed-assets"))]
fn load_weight_path(path: PathBuf, device: &Device) -> candle_core::Result<VarBuilder<'static>> {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("safetensors") => load_safetensors_path(path, device),
        Some("pt" | "pth" | "bin") => load_pth_path(path, device),
        _ => Err(Error::other(format!(
            "unsupported MobileCLIP2 weight file extension for {}",
            path.display()
        ))
        .into()),
    }
}

#[cfg(not(feature = "embed-assets"))]
fn load_safetensors_path(
    path: PathBuf,
    device: &Device,
) -> candle_core::Result<VarBuilder<'static>> {
    unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device) }
}

#[cfg(not(feature = "embed-assets"))]
fn load_pth_path(path: PathBuf, device: &Device) -> candle_core::Result<VarBuilder<'static>> {
    VarBuilder::from_pth(path, DType::F32, device)
}

pub fn load_image<P: AsRef<Path>>(path: P, image_size: usize) -> candle_core::Result<Tensor> {
    let image = image::ImageReader::open(path.as_ref())
        .map_err(candle_core::Error::wrap)?
        .decode()
        .map_err(candle_core::Error::wrap)?
        .resize_to_fill(
            image_size as u32,
            image_size as u32,
            image::imageops::FilterType::Triangle,
        )
        .to_rgb8();
    let data = image.into_raw();
    let tensor =
        Tensor::from_vec(data, (image_size, image_size, 3), &Device::Cpu)?.permute((2, 0, 1))?;
    tensor.to_dtype(DType::F32)? / 255.0
}
