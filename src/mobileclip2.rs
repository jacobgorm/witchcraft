//! MobileCLIP2-S0 image and text encoder using Candle.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};
#[cfg(any(feature = "modernbert", feature = "modernbert-quantized"))]
use candle_nn::{layer_norm, linear, linear_no_bias, LayerNorm};
use candle_transformers::models::{fastvit, openclip::text_model};
use serde::Deserialize;
use std::io::Error;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

use crate::profiled_fastvit::ProfiledFastVit;
pub use crate::profiled_fastvit::{ImageProfile, ProfileRecord};

#[cfg(any(
    feature = "embed-assets",
    feature = "modernbert",
    feature = "modernbert-quantized"
))]
use crate::embed_asset;

#[cfg(feature = "embed-assets")]
embed_asset!(pub CONFIG, "mobileclip2-config.json");
#[cfg(feature = "embed-assets")]
embed_asset!(pub TOKENIZER, "mobileclip2-tokenizer.json");
#[cfg(feature = "embed-assets")]
embed_asset!(pub MODEL, "mobileclip2.safetensors");
#[cfg(any(feature = "modernbert", feature = "modernbert-quantized"))]
embed_asset!(pub XTR_ADAPTER, "mobileclip2-xtr.safetensors");

const DEFAULT_CONTEXT_LENGTH: usize = 77;
const DEFAULT_EMBED_DIM: usize = 512;
#[cfg(any(feature = "modernbert", feature = "modernbert-quantized"))]
const XTR_IMAGE_SIZE: usize = 512;
#[cfg(any(feature = "modernbert", feature = "modernbert-quantized"))]
const GATED_XTR_IMAGE_TOKENS: usize = 128;
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

#[cfg(any(feature = "modernbert", feature = "modernbert-quantized"))]
#[derive(Clone, Debug)]
struct MlpImageAdapter {
    norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

#[cfg(any(feature = "modernbert", feature = "modernbert-quantized"))]
impl MlpImageAdapter {
    fn load(vb: VarBuilder, hidden: usize) -> Result<Self> {
        Ok(Self {
            norm: layer_norm(512, 1e-5, vb.pp(0))?,
            fc1: linear(512, hidden, vb.pp(1))?,
            fc2: linear_no_bias(hidden, 96, vb.pp(3))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.fc2
            .forward(&self.fc1.forward(&self.norm.forward(xs)?)?.gelu_erf()?)
    }
}

#[cfg(any(feature = "modernbert", feature = "modernbert-quantized"))]
#[derive(Clone, Debug)]
enum XtrImageAdapter {
    Gated {
        mlp: MlpImageAdapter,
        token_gate_norm: LayerNorm,
        token_gate: Linear,
        keep_tokens: usize,
    },
    Residual {
        base: MlpImageAdapter,
        residual: MlpImageAdapter,
        scale: f64,
    },
    ResidualGated {
        base: MlpImageAdapter,
        residual: MlpImageAdapter,
        scale: f64,
        token_gate_norm: LayerNorm,
        token_gate: Linear,
        keep_tokens: usize,
        output_scale: Option<Tensor>,
        output_bias: Option<Tensor>,
    },
}

#[cfg(any(feature = "modernbert", feature = "modernbert-quantized"))]
fn normalize_xtr_vectors(xs: &Tensor) -> Result<Tensor> {
    xs.broadcast_div(&xs.sqr()?.sum_keepdim(2)?.sqrt()?)
}

#[cfg(any(feature = "modernbert", feature = "modernbert-quantized"))]
impl XtrImageAdapter {
    fn load(
        vb: VarBuilder,
        adapter_type: &str,
        residual_scale: Option<f64>,
        image_tokens: Option<usize>,
        has_output_scale: bool,
        has_output_bias: bool,
    ) -> Result<Self> {
        match adapter_type {
            "gated_mlp" => Ok(Self::Gated {
                mlp: MlpImageAdapter::load(vb.pp("net"), 256)?,
                token_gate_norm: layer_norm(512, 1e-5, vb.pp("token_gate_norm"))?,
                token_gate: linear(512, 1, vb.pp("token_gate"))?,
                keep_tokens: image_tokens.unwrap_or(GATED_XTR_IMAGE_TOKENS),
            }),
            "residual_mlp" => Ok(Self::Residual {
                base: MlpImageAdapter::load(vb.pp("base.net"), 256)?,
                residual: MlpImageAdapter::load(vb.pp("residual"), 128)?,
                scale: residual_scale.ok_or_else(|| {
                    Error::other("residual MobileCLIP2 XTR asset lacks residual_scale metadata")
                })?,
            }),
            "residual_gated_mlp" => Ok(Self::ResidualGated {
                base: MlpImageAdapter::load(vb.pp("base.net"), 256)?,
                residual: MlpImageAdapter::load(vb.pp("residual"), 128)?,
                scale: residual_scale.ok_or_else(|| {
                    Error::other("residual MobileCLIP2 XTR asset lacks residual_scale metadata")
                })?,
                token_gate_norm: layer_norm(512, 1e-5, vb.pp("token_gate_norm"))?,
                token_gate: linear(512, 1, vb.pp("token_gate"))?,
                keep_tokens: image_tokens.ok_or_else(|| {
                    Error::other("gated MobileCLIP2 XTR asset lacks image_tokens metadata")
                })?,
                output_scale: if has_output_scale {
                    Some(vb.get(96, "output_scale")?)
                } else {
                    None
                },
                output_bias: if has_output_bias {
                    Some(vb.get(96, "output_bias")?)
                } else {
                    None
                },
            }),
            other => Err(Error::other(format!(
                "unsupported MobileCLIP2 XTR adapter type `{other}`"
            ))
            .into()),
        }
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Gated {
                mlp,
                token_gate_norm,
                token_gate,
                keep_tokens,
            } => {
                let xs = select_gated_tokens(
                    xs,
                    token_gate_norm,
                    token_gate,
                    *keep_tokens,
                )?;
                mlp.forward(&xs)
            }
            Self::Residual {
                base,
                residual,
                scale,
            } => {
                let base = normalize_xtr_vectors(&base.forward(xs)?)?;
                base.broadcast_add(&(residual.forward(xs)? * *scale)?)
            }
            Self::ResidualGated {
                base,
                residual,
                scale,
                token_gate_norm,
                token_gate,
                keep_tokens,
                output_scale,
                output_bias,
            } => {
                let xs = select_gated_tokens(
                    xs,
                    token_gate_norm,
                    token_gate,
                    *keep_tokens,
                )?;
                let base = normalize_xtr_vectors(&base.forward(&xs)?)?;
                let mut output = base.broadcast_add(&(residual.forward(&xs)? * *scale)?)?;
                if let Some(scale) = output_scale {
                    output = output.broadcast_mul(scale)?;
                }
                match output_bias {
                    Some(bias) => output.broadcast_add(bias),
                    None => Ok(output),
                }
            }
        }
    }
}

#[cfg(any(feature = "modernbert", feature = "modernbert-quantized"))]
fn select_gated_tokens(
    xs: &Tensor,
    token_gate_norm: &LayerNorm,
    token_gate: &Linear,
    keep_tokens: usize,
) -> Result<Tensor> {
    let xs = xs.contiguous()?;
    let (batch, tokens, width) = xs.dims3()?;
    let keep = keep_tokens.min(tokens);
    let gate_logits = token_gate
        .forward(&token_gate_norm.forward(&xs)?)?
        .squeeze(2)?;
    let indices = gate_logits
        .contiguous()?
        .arg_sort_last_dim(false)?
        .narrow(1, 0, keep)?
        .unsqueeze(2)?
        .broadcast_as((batch, keep, width))?
        .contiguous()?;
    xs.gather(&indices, 1)
}

/// MobileCLIP2 vision tower followed by the trained ModernBERT-XTR adapter.
#[cfg(any(feature = "modernbert", feature = "modernbert-quantized"))]
#[derive(Clone, Debug)]
pub struct XtrImageEncoder {
    vision_model: ProfiledFastVit,
    adapter: XtrImageAdapter,
    image_size: usize,
    device: Device,
}

#[cfg(any(feature = "modernbert", feature = "modernbert-quantized"))]
impl XtrImageEncoder {
    pub fn load(device: &Device, assets: &Path) -> Result<Self> {
        let (builder, _) = T5ModelBuilder::load(assets)?;
        let vision_config = builder.config.vision_model_config()?;
        let mobileclip_vb = load_weights(device, assets)?;
        let vision_model = ProfiledFastVit::new(
            &vision_config,
            builder.config.embed_dim,
            mobileclip_vb.pp("visual.trunk"),
        )?;
        let adapter_bytes = XTR_ADAPTER
            .bytes(assets)
            .map_err(|_| Error::other("failed to read mobileclip2-xtr.safetensors"))?;
        let tensors = safetensors::SafeTensors::deserialize(&adapter_bytes)
            .map_err(|e| Error::other(format!("invalid MobileCLIP2 XTR asset: {e}")))?;
        let (_, header) = safetensors::SafeTensors::read_metadata(&adapter_bytes)
            .map_err(|e| Error::other(format!("invalid MobileCLIP2 XTR metadata: {e}")))?;
        let metadata = header.metadata().as_ref();
        let adapter_type = metadata
            .and_then(|values| values.get("adapter_type"))
            .map(String::as_str)
            .unwrap_or_else(|| {
                if tensors.names().contains(&"base.net.0.weight") {
                    "residual_mlp"
                } else {
                    "gated_mlp"
                }
            });
        let residual_scale = metadata
            .and_then(|values| values.get("residual_scale"))
            .map(|value| value.parse::<f64>())
            .transpose()
            .map_err(|e| Error::other(format!("invalid residual_scale metadata: {e}")))?;
        let image_tokens = metadata
            .and_then(|values| values.get("image_tokens"))
            .map(|value| value.parse::<usize>())
            .transpose()
            .map_err(|e| Error::other(format!("invalid image_tokens metadata: {e}")))?;
        let has_output_scale = tensors.names().contains(&"output_scale");
        let has_output_bias = tensors.names().contains(&"output_bias");
        let adapter_vb = VarBuilder::from_buffered_safetensors(
            adapter_bytes.to_vec(),
            DType::F32,
            device,
        )?;
        let adapter = XtrImageAdapter::load(
            adapter_vb,
            adapter_type,
            residual_scale,
            image_tokens,
            has_output_scale,
            has_output_bias,
        )?;
        Ok(Self {
            vision_model,
            adapter,
            image_size: XTR_IMAGE_SIZE,
            device: device.clone(),
        })
    }

    /// Return normalized-ready 96d patch vectors for a 512x512 image.
    /// Gated assets keep the token count recorded in their metadata.
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        self.adapter
            .forward(&self.vision_model.forward_patches(pixel_values)?)
    }

    pub fn forward_profiled(&self, pixel_values: &Tensor) -> Result<(Tensor, ImageProfile)> {
        let (patches, mut profile) = self.vision_model.forward_patches_profiled(pixel_values)?;
        self.device.synchronize()?;
        let started = std::time::Instant::now();
        let output = self.adapter.forward(&patches)?;
        self.device.synchronize()?;
        profile.record("xtr_adapter", started.elapsed(), output.dims().to_vec());
        Ok((output, profile))
    }

    pub fn image_size(&self) -> usize {
        self.image_size
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
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
    let path = path.as_ref();
    let image = image::ImageReader::open(path)
        .map_err(candle_core::Error::wrap)?
        .decode()
        .map_err(candle_core::Error::wrap)?;
    let image = preprocess_image(image, image_size)?;
    let data = image.into_raw();
    let tensor =
        Tensor::from_vec(data, (image_size, image_size, 3), &Device::Cpu)?.permute((2, 0, 1))?;
    tensor.to_dtype(DType::F32)? / 255.0
}

fn preprocess_image(
    image: image::DynamicImage,
    image_size: usize,
) -> candle_core::Result<image::RgbImage> {
    let image_size = u32::try_from(image_size)
        .map_err(|_| Error::other(format!("image size {image_size} does not fit in u32")))?;
    if image_size == 0 {
        return Err(Error::other("image size must be greater than zero").into());
    }

    let (width, height) = (image.width(), image.height());
    if width == 0 || height == 0 {
        return Err(Error::other("input image has zero width or height").into());
    }

    // resize_to_fill scales uniformly until the shorter source dimension fills
    // the square. Before shrinking, remove frequencies above the destination
    // Nyquist limit. Assuming roughly half-pixel source reconstruction blur,
    // variances add under convolution, giving this scale-aware extra sigma.
    let downsample = (width as f32 / image_size as f32)
        .min(height as f32 / image_size as f32);
    let image = if downsample > 1.0 {
        let sigma = 0.5 * (downsample * downsample - 1.0).sqrt();
        image.blur(sigma)
    } else {
        image
    };

    Ok(image
        .resize_to_fill(
            image_size,
            image_size,
            image::imageops::FilterType::Lanczos3,
        )
        .to_rgb8())
}

#[cfg(test)]
mod image_preprocessing_tests {
    use super::*;

    #[test]
    fn downsampling_checkerboard_is_low_pass_filtered() -> candle_core::Result<()> {
        let mut source = image::RgbImage::new(1024, 1024);
        for (x, y, pixel) in source.enumerate_pixels_mut() {
            let value = if (x + y) % 2 == 0 { 0 } else { 255 };
            *pixel = image::Rgb([value, value, value]);
        }

        let resized = preprocess_image(image::DynamicImage::ImageRgb8(source), 64)?;
        assert_eq!(resized.dimensions(), (64, 64));
        let (min, max) = resized
            .pixels()
            .map(|pixel| pixel[0])
            .fold((u8::MAX, u8::MIN), |(min, max), value| {
                (min.min(value), max.max(value))
            });
        assert!(
            max - min <= 12,
            "checkerboard retained high frequencies: {min}..={max}"
        );
        Ok(())
    }

    #[test]
    fn preprocessing_preserves_cover_crop_dimensions() -> candle_core::Result<()> {
        let source = image::DynamicImage::ImageRgb8(image::RgbImage::new(640, 320));
        let resized = preprocess_image(source, 256)?;
        assert_eq!(resized.dimensions(), (256, 256));
        Ok(())
    }

}
