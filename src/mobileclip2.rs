//! MobileCLIP2-S0 image and text encoder using Candle.

use std::io::{BufReader, Error, Read, Seek, SeekFrom};
use std::path::Path;
#[cfg(not(feature = "embed-assets"))]
use std::path::PathBuf;

#[cfg(feature = "modernbert-quantized")]
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Module, Result, Tensor};
#[cfg(any(feature = "modernbert", feature = "modernbert-quantized"))]
use candle_nn::{layer_norm, linear, linear_no_bias, LayerNorm};
use candle_nn::{Linear, VarBuilder};
use candle_transformers::models::fastvit;
use candle_transformers::models::openclip::text_model;
use image::GenericImageView;
use serde::Deserialize;
use tokenizers::Tokenizer;

#[cfg(any(
    feature = "embed-assets",
    feature = "modernbert",
    feature = "modernbert-quantized"
))]
use crate::embed_asset;
use crate::profiled_fastvit::ProfiledFastVit;
pub use crate::profiled_fastvit::{ImageProfile, ProfileRecord};

#[cfg(feature = "embed-assets")]
embed_asset!(pub CONFIG, "mobileclip2-config.json");
#[cfg(all(
    feature = "embed-assets",
    not(any(feature = "modernbert", feature = "modernbert-quantized"))
))]
embed_asset!(pub TOKENIZER, "mobileclip2-tokenizer.json");
#[cfg(all(feature = "embed-assets", not(feature = "modernbert-quantized")))]
embed_asset!(pub MODEL, "mobileclip2.safetensors");
#[cfg(feature = "modernbert")]
embed_asset!(pub XTR_ADAPTER, "mobileclip2-xtr.safetensors");
#[cfg(feature = "modernbert-quantized")]
embed_asset!(pub QUANTIZED_MODEL, "mobileclip2.gguf");
#[cfg(feature = "modernbert-quantized")]
embed_asset!(pub QUANTIZED_XTR_ADAPTER, "mobileclip2-xtr.gguf");

const DEFAULT_CONTEXT_LENGTH: usize = 77;
const DEFAULT_EMBED_DIM: usize = 512;
#[cfg(any(feature = "modernbert", feature = "modernbert-quantized"))]
const XTR_IMAGE_SIZE: usize = 512;
#[cfg(any(feature = "modernbert", feature = "modernbert-quantized"))]
const GATED_XTR_IMAGE_TOKENS: usize = 64;
const DEFAULT_TEXT_WIDTH: usize = 512;
const DEFAULT_TEXT_HEADS: usize = 8;
const DEFAULT_TEXT_LAYERS: usize = 12;
const DEFAULT_VOCAB_SIZE: usize = 49408;
const STOCK_WEIGHTS: &str = "mobileclip2-stock.safetensors";
const STOCK_TOKENIZER: &str = "mobileclip2-tokenizer.json";

#[cfg(not(feature = "embed-assets"))]
const CONFIG_ENV: &str = "WARP_MOBILECLIP2_CONFIG";
#[cfg(all(
    not(feature = "embed-assets"),
    not(any(feature = "modernbert", feature = "modernbert-quantized"))
))]
const TOKENIZER_ENV: &str = "WARP_MOBILECLIP2_TOKENIZER";
#[cfg(all(not(feature = "embed-assets"), not(feature = "modernbert-quantized")))]
const WEIGHTS_ENV: &str = "WARP_MOBILECLIP2_WEIGHTS";

#[cfg(not(feature = "embed-assets"))]
const CONFIG_CANDIDATES: &[&str] = &["mobileclip2-config.json", "open_clip_config.json"];
#[cfg(all(
    not(feature = "embed-assets"),
    not(any(feature = "modernbert", feature = "modernbert-quantized"))
))]
const TOKENIZER_CANDIDATES: &[&str] = &["mobileclip2-tokenizer.json", "open_clip_tokenizer.json"];
#[cfg(all(not(feature = "embed-assets"), not(feature = "modernbert-quantized")))]
const SAFETENSOR_CANDIDATES: &[&str] = &["mobileclip2.safetensors", "open_clip_model.safetensors"];
#[cfg(all(not(feature = "embed-assets"), not(feature = "modernbert-quantized")))]
const PTH_CANDIDATES: &[&str] = &["mobileclip2_s0.pt", "open_clip_pytorch_model.bin"];

#[derive(Debug, Clone, Deserialize)]
struct Config {
    #[serde(default = "default_embed_dim")]
    embed_dim: usize,
    #[serde(default)]
    vision_cfg: VisionConfig,
    #[cfg_attr(any(feature = "modernbert", feature = "modernbert-quantized"), allow(dead_code))]
    text_cfg: TextConfig,
}

#[derive(Debug, Clone, Deserialize)]
struct VisionConfig {
    #[serde(default = "default_vision_model_name")]
    timm_model_name: String,
    #[serde(default = "default_image_size")]
    #[cfg_attr(any(feature = "modernbert", feature = "modernbert-quantized"), allow(dead_code))]
    image_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[cfg_attr(any(feature = "modernbert", feature = "modernbert-quantized"), allow(dead_code))]
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

    #[cfg_attr(any(feature = "modernbert", feature = "modernbert-quantized"), allow(dead_code))]
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
    adapter_device: Device,
    image_size: usize,
    device: Device,
}

#[cfg(any(feature = "modernbert", feature = "modernbert-quantized"))]
impl XtrImageEncoder {
    pub fn load(device: &Device, assets: &Path) -> Result<Self> {
        let config = load_config(assets)?;
        let vision_config = config.vision_model_config()?;
        let mobileclip_vb = load_xtr_weights(device, assets)?;
        let vision_model = ProfiledFastVit::new(
            &vision_config,
            config.embed_dim,
            mobileclip_vb.pp("visual.trunk"),
        )?;
        #[cfg(any(
            all(feature = "neso-d3d12", target_os = "windows"),
            all(feature = "neso-metal", target_os = "macos", target_arch = "x86_64")
        ))]
        let adapter_device = if matches!(device, Device::D3D12(_) | Device::Metal(_)) {
            Device::Cpu
        } else {
            device.clone()
        };
        #[cfg(not(any(
            all(feature = "neso-d3d12", target_os = "windows"),
            all(feature = "neso-metal", target_os = "macos", target_arch = "x86_64")
        )))]
        let adapter_device = device.clone();

        #[cfg(feature = "modernbert")]
        let (
            adapter_vb,
            adapter_type,
            residual_scale,
            image_tokens,
            has_output_scale,
            has_output_bias,
        ) = {
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
            let adapter_vb =
                VarBuilder::from_buffered_safetensors(
                    adapter_bytes.to_vec(),
                    DType::F32,
                    &adapter_device,
                )?;
            (
                adapter_vb,
                adapter_type.to_string(),
                residual_scale,
                image_tokens,
                has_output_scale,
                has_output_bias,
            )
        };

        #[cfg(feature = "modernbert-quantized")]
        let (
            adapter_vb,
            adapter_type,
            residual_scale,
            image_tokens,
            has_output_scale,
            has_output_bias,
        ) = {
            let adapter_bytes = QUANTIZED_XTR_ADAPTER
                .bytes(assets)
                .map_err(|_| Error::other("failed to read mobileclip2-xtr.gguf"))?;
            let (adapter_vb, metadata) = load_gguf_weights(adapter_bytes, &adapter_device)?;
            let adapter_type = gguf_string(&metadata, "adapter_type").unwrap_or_else(|| {
                if adapter_vb.contains_tensor("base.net.0.weight") {
                    "residual_mlp"
                } else {
                    "gated_mlp"
                }
            });
            let residual_scale = gguf_string(&metadata, "residual_scale")
                .map(str::parse::<f64>)
                .transpose()
                .map_err(|e| Error::other(format!("invalid residual_scale metadata: {e}")))?;
            let image_tokens = gguf_string(&metadata, "image_tokens")
                .map(str::parse::<usize>)
                .transpose()
                .map_err(|e| Error::other(format!("invalid image_tokens metadata: {e}")))?;
            let has_output_scale = adapter_vb.contains_tensor("output_scale");
            let has_output_bias = adapter_vb.contains_tensor("output_bias");
            (
                adapter_vb,
                adapter_type.to_string(),
                residual_scale,
                image_tokens,
                has_output_scale,
                has_output_bias,
            )
        };

        let adapter = XtrImageAdapter::load(
            adapter_vb,
            &adapter_type,
            residual_scale,
            image_tokens,
            has_output_scale,
            has_output_bias,
        )?;
        Ok(Self {
            vision_model,
            adapter,
            adapter_device,
            image_size: XTR_IMAGE_SIZE,
            device: device.clone(),
        })
    }

    /// Return normalized-ready 96d patch vectors for a 512x512 image.
    /// Gated assets keep the token count recorded in their metadata.
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let patches = self
            .vision_model
            .forward_patches(pixel_values)?
            .to_device(&self.adapter_device)?;
        self.adapter.forward(&patches)
    }

    pub fn forward_profiled(&self, pixel_values: &Tensor) -> Result<(Tensor, ImageProfile)> {
        let (patches, mut profile) = self.vision_model.forward_patches_profiled(pixel_values)?;
        self.device.synchronize()?;
        let started = std::time::Instant::now();
        let output = self
            .adapter
            .forward(&patches.to_device(&self.adapter_device)?)?;
        self.adapter_device.synchronize()?;
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

/// Unmodified MobileCLIP2 image/text model used as an exhaustive 512d baseline.
pub struct StockMobileClip2 {
    tokenizer: Tokenizer,
    model: T5EncoderModel,
}

impl StockMobileClip2 {
    pub fn load(device: &Device, assets: &Path) -> Result<Self> {
        let config = load_config(assets)?;
        let tokenizer_path = assets.join(STOCK_TOKENIZER);
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|error| {
            Error::other(format!(
                "failed to load stock MobileCLIP2 tokenizer {}: {error}",
                tokenizer_path.display()
            ))
        })?;
        let weights_path = assets.join(STOCK_WEIGHTS);
        if !weights_path.is_file() {
            return Err(Error::other(format!(
                "missing stock MobileCLIP2 weights {}; run `make mobileclip2-stock-assets`",
                weights_path.display()
            ))
            .into());
        }
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)?
        };
        let model = T5EncoderModel::load(vb, &config)?;
        Ok(Self { tokenizer, model })
    }

    pub fn embed_text(&self, text: &str) -> Result<Tensor> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|error| Error::other(format!("tokenize MobileCLIP2 query: {error}")))?;
        let context_length = self.model.max_seq_len();
        // MobileCLIP2-S0 has bidirectional text attention (`no_causal_mask`),
        // so its EOS representation depends on the same full-context padding
        // used by the reference OpenCLIP tokenizer.
        let padded = padded_text_ids(encoding.get_ids(), context_length)?;
        let input = Tensor::new(padded.as_slice(), self.model.device())?.unsqueeze(0)?;
        normalize_rows(&self.model.forward_text_pooled(&input)?)
    }

    pub fn embed_images(&self, images: &Tensor) -> Result<Tensor> {
        let images = images.to_device(self.model.device())?;
        normalize_rows(&self.model.forward_image(&images)?)
    }

    pub fn image_size(&self) -> usize {
        self.model.image_size()
    }

    pub fn device(&self) -> &Device {
        self.model.device()
    }
}

fn normalize_rows(values: &Tensor) -> Result<Tensor> {
    values.broadcast_div(&values.sqr()?.sum_keepdim(1)?.sqrt()?)
}

fn padded_text_ids(ids: &[u32], context_length: usize) -> Result<Vec<u32>> {
    if ids.len() > context_length {
        return Err(Error::other(format!(
            "MobileCLIP2 query has {} tokens, context length is {context_length}",
            ids.len()
        ))
        .into());
    }
    let mut padded = vec![0; context_length];
    padded[..ids.len()].copy_from_slice(ids);
    Ok(padded)
}

#[cfg(not(any(feature = "modernbert", feature = "modernbert-quantized")))]
pub struct T5ModelBuilder {
    config: Config,
}

fn load_config(assets: &Path) -> candle_core::Result<Config> {
    #[cfg(feature = "embed-assets")]
    return Config::from_json(
        CONFIG
            .bytes(assets)
            .map_err(|_| Error::other("failed to read embedded mobileclip2-config.json"))?,
    );

    #[cfg(not(feature = "embed-assets"))]
    match find_asset_path(CONFIG_ENV, CONFIG_CANDIDATES, assets) {
        Some(path) => Config::from_json(&std::fs::read(&path).map_err(|e| {
            Error::other(format!(
                "failed to read MobileCLIP2 config {}: {e}",
                path.display()
            ))
        })?),
        None => Ok(Config::default()),
    }
}

#[cfg(not(any(feature = "modernbert", feature = "modernbert-quantized")))]
impl T5ModelBuilder {
    pub fn load(assets: &Path) -> candle_core::Result<(Self, Tokenizer)> {
        let config = load_config(assets)?;

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

#[cfg(feature = "modernbert-quantized")]
fn gguf_string<'a>(
    metadata: &'a std::collections::HashMap<String, gguf_file::Value>,
    key: &str,
) -> Option<&'a str> {
    match metadata.get(key) {
        Some(gguf_file::Value::String(value)) => Some(value),
        _ => None,
    }
}

#[cfg(feature = "modernbert-quantized")]
fn load_gguf_weights(
    bytes: &[u8],
    device: &Device,
) -> candle_core::Result<(
    VarBuilder<'static>,
    std::collections::HashMap<String, gguf_file::Value>,
)> {
    let mut cursor = std::io::Cursor::new(bytes);
    let content = gguf_file::Content::read(&mut cursor)?;
    let names = content.tensor_infos.keys().cloned().collect::<Vec<_>>();
    let mut tensors = std::collections::HashMap::with_capacity(names.len());
    for name in names {
        // GGUF quantized storage is not implemented for D3D12. Decode on the
        // CPU once at model load, then upload the dequantized fp32 tensor.
        let quantized_device = if matches!(device, Device::D3D12(_)) {
            &Device::Cpu
        } else {
            device
        };
        let tensor = content
            .tensor(&mut cursor, &name, quantized_device)?
            .dequantize(device)?;
        tensors.insert(name, tensor);
    }
    Ok((
        VarBuilder::from_tensors(tensors, DType::F32, device),
        content.metadata,
    ))
}

#[cfg(feature = "modernbert-quantized")]
fn load_xtr_weights(device: &Device, assets: &Path) -> candle_core::Result<VarBuilder<'static>> {
    let bytes = QUANTIZED_MODEL
        .bytes(assets)
        .map_err(|_| Error::other("failed to read mobileclip2.gguf"))?;
    load_gguf_weights(bytes, device).map(|(vb, _)| vb)
}

#[cfg(feature = "modernbert")]
fn load_xtr_weights(device: &Device, assets: &Path) -> candle_core::Result<VarBuilder<'static>> {
    load_weights(device, assets)
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

#[cfg(all(feature = "embed-assets", not(feature = "modernbert-quantized")))]
fn load_weights(device: &Device, assets: &Path) -> candle_core::Result<VarBuilder<'static>> {
    let model_bytes = MODEL
        .bytes(assets)
        .map_err(|_| Error::other("failed to read embedded mobileclip2.safetensors"))?;
    VarBuilder::from_slice_safetensors(model_bytes, DType::F32, device)
}

#[cfg(all(not(feature = "embed-assets"), not(feature = "modernbert-quantized")))]
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

#[cfg(all(not(feature = "embed-assets"), not(feature = "modernbert-quantized")))]
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

#[cfg(all(not(feature = "embed-assets"), not(feature = "modernbert-quantized")))]
fn load_safetensors_path(
    path: PathBuf,
    device: &Device,
) -> candle_core::Result<VarBuilder<'static>> {
    unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device) }
}

#[cfg(all(not(feature = "embed-assets"), not(feature = "modernbert-quantized")))]
fn load_pth_path(path: PathBuf, device: &Device) -> candle_core::Result<VarBuilder<'static>> {
    VarBuilder::from_pth(path, DType::F32, device)
}

fn is_heif_header(header: &[u8]) -> bool {
    header.len() >= 12
        && &header[4..8] == b"ftyp"
        && matches!(
        &header[8..12],
        b"heic"
            | b"heix"
            | b"heim"
            | b"heis"
            | b"hevc"
            | b"hevx"
            | b"hevm"
            | b"hevs"
            | b"mif1"
            | b"msf1"
    )
}

fn is_jxl_header(header: &[u8]) -> bool {
    header.starts_with(&[0xff, 0x0a])
        || header.starts_with(&[
            0x00, 0x00, 0x00, 0x0c, b'J', b'X', b'L', b' ', 0x0d, 0x0a, 0x87, 0x0a,
        ])
}

enum ImageSource {
    Heif,
    Jxl(BufReader<std::fs::File>),
    Standard(image::ImageReader<BufReader<std::fs::File>>),
}

fn open_image_source(path: &Path) -> candle_core::Result<ImageSource> {
    let mut file = std::fs::File::open(path).map_err(candle_core::Error::wrap)?;
    let mut header = [0u8; 12];
    let read = file.read(&mut header).map_err(candle_core::Error::wrap)?;
    if is_heif_header(&header[..read]) {
        return Ok(ImageSource::Heif);
    }
    file.seek(SeekFrom::Start(0))
        .map_err(candle_core::Error::wrap)?;
    if is_jxl_header(&header[..read]) {
        return Ok(ImageSource::Jxl(BufReader::new(file)));
    }
    let reader = image::ImageReader::new(BufReader::new(file))
        .with_guessed_format()
        .map_err(candle_core::Error::wrap)?;
    match reader.format() {
        Some(image::ImageFormat::Jpeg | image::ImageFormat::Png | image::ImageFormat::WebP) => {},
        Some(format) => {
            return Err(Error::other(format!(
                "unsupported image format {format:?} in {}",
                path.display()
            ))
            .into());
        },
        None => {
            return Err(Error::other(format!(
                "unrecognized image header in {}",
                path.display()
            ))
            .into());
        },
    }
    Ok(ImageSource::Standard(reader))
}

fn decode_heif(path: &Path) -> candle_core::Result<image::DynamicImage> {
    let decoded = heif_oxide::decode_file(path)
        .map_err(|error| Error::other(format!("failed to decode {}: {error}", path.display())))?;
    let image = image::RgbaImage::from_raw(decoded.width, decoded.height, decoded.to_rgba8())
        .ok_or_else(|| Error::other(format!("invalid decoded image size for {}", path.display())))?;
    Ok(image::DynamicImage::ImageRgba8(image))
}

fn decode_jxl(
    path: &Path,
    reader: BufReader<std::fs::File>,
) -> candle_core::Result<image::DynamicImage> {
    let decoder = jxl_oxide::integration::JxlDecoder::new(reader)
        .map_err(|error| Error::other(format!("failed to decode {}: {error}", path.display())))?;
    image::DynamicImage::from_decoder(decoder).map_err(candle_core::Error::wrap)
}

pub fn validate_image_header<P: AsRef<Path>>(path: P) -> candle_core::Result<()> {
    open_image_source(path.as_ref()).map(|_| ())
}

pub fn decode_image<P: AsRef<Path>>(path: P) -> candle_core::Result<image::DynamicImage> {
    let path = path.as_ref();
    match open_image_source(path)? {
        ImageSource::Heif => decode_heif(path),
        ImageSource::Jxl(reader) => decode_jxl(path, reader),
        ImageSource::Standard(reader) => reader.decode().map_err(candle_core::Error::wrap),
    }
}

pub fn decode_image_oriented<P: AsRef<Path>>(path: P) -> candle_core::Result<image::DynamicImage> {
    use image::ImageDecoder;

    let path = path.as_ref();
    match open_image_source(path)? {
        ImageSource::Heif => decode_heif(path),
        // jxl-oxide applies the JPEG XL orientation while rendering.
        ImageSource::Jxl(reader) => decode_jxl(path, reader),
        ImageSource::Standard(reader) => {
            let mut decoder = reader.into_decoder().map_err(candle_core::Error::wrap)?;
            let orientation = decoder.orientation().map_err(candle_core::Error::wrap)?;
            let mut image =
                image::DynamicImage::from_decoder(decoder).map_err(candle_core::Error::wrap)?;
            image.apply_orientation(orientation);
            Ok(image)
        },
    }
}

pub fn load_image<P: AsRef<Path>>(path: P, image_size: usize) -> candle_core::Result<Tensor> {
    let image = decode_image(path)?;
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

    let side = width.min(height);
    let image = image.into_rgb8();
    let source = image.view((width - side) / 2, (height - side) / 2, side, side);
    // Resize directly from the centered source view: Lanczos samples only the
    // retained region and widens its support according to the downsampling ratio.
    let resized = image::imageops::resize(
        &*source,
        image_size,
        image_size,
        image::imageops::FilterType::Lanczos3,
    );
    Ok(resized)
}

#[cfg(test)]
mod image_preprocessing_tests {
    use super::*;

    #[test]
    fn mobileclip2_text_ids_are_zero_padded_to_full_context() {
        let padded = padded_text_ids(&[49406, 320, 49407], 6).unwrap();
        assert_eq!(padded, [49406, 320, 49407, 0, 0, 0]);
    }

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

    #[test]
    fn preprocessing_resamples_only_the_center_crop() -> candle_core::Result<()> {
        let mut source = image::RgbImage::from_pixel(640, 320, image::Rgb([0, 255, 0]));
        for y in 0..320 {
            for x in 160..480 {
                source.put_pixel(x, y, image::Rgb([255, 0, 0]));
            }
        }

        let resized = preprocess_image(image::DynamicImage::ImageRgb8(source), 64)?;
        assert!(resized.pixels().all(|pixel| *pixel == image::Rgb([255, 0, 0])));
        Ok(())
    }

    #[test]
    fn decoding_uses_file_signature_before_extension() -> candle_core::Result<()> {
        let mut jpeg = Vec::new();
        image::codecs::jpeg::JpegEncoder::new(&mut jpeg)
            .encode(&[255, 0, 0, 0, 0, 255], 2, 1, image::ExtendedColorType::Rgb8)
            .map_err(candle_core::Error::wrap)?;
        let file = tempfile::Builder::new()
            .suffix(".png")
            .tempfile()
            .map_err(candle_core::Error::wrap)?;
        std::fs::write(file.path(), jpeg).map_err(candle_core::Error::wrap)?;

        let decoded = decode_image_oriented(file.path())?;

        assert_eq!((decoded.width(), decoded.height()), (2, 1));
        Ok(())
    }

    #[test]
    fn decoding_supports_jpeg_xl_by_signature() -> candle_core::Result<()> {
        // Small valid codestream from the jxl-oxide decoder documentation.
        let jxl = [
            0xff, 0x0a, 0x30, 0x54, 0x10, 0x09, 0x08, 0x06, 0x01, 0x00, 0x78, 0x00, 0x4b,
            0x38, 0x41, 0x3c, 0xb6, 0x3a, 0x51, 0xfe, 0x00, 0x47, 0x1e, 0xa0, 0x85, 0xb8,
            0x27, 0x1a, 0x48, 0x45, 0x84, 0x1b, 0x71, 0x4f, 0xa8, 0x3e, 0x8e, 0x30, 0x03,
            0x92, 0x84, 0x01,
        ];
        let file = tempfile::Builder::new()
            .suffix(".png")
            .tempfile()
            .map_err(candle_core::Error::wrap)?;
        std::fs::write(file.path(), jxl).map_err(candle_core::Error::wrap)?;

        let decoded = decode_image_oriented(file.path())?;

        assert!(decoded.width() > 0);
        assert!(decoded.height() > 0);
        Ok(())
    }

    #[test]
    fn recognizes_both_jpeg_xl_signatures() {
        assert!(is_jxl_header(&[0xff, 0x0a]));
        assert!(is_jxl_header(&[
            0x00, 0x00, 0x00, 0x0c, b'J', b'X', b'L', b' ', 0x0d, 0x0a, 0x87, 0x0a,
        ]));
    }

    #[test]
    fn decoding_applies_exif_orientation() -> candle_core::Result<()> {
        let mut jpeg = Vec::new();
        image::codecs::jpeg::JpegEncoder::new(&mut jpeg)
            .encode(&[255, 0, 0, 0, 0, 255], 2, 1, image::ExtendedColorType::Rgb8)
            .map_err(candle_core::Error::wrap)?;
        let exif_orientation_6 = [
            0xff, 0xe1, 0x00, 0x22, b'E', b'x', b'i', b'f', 0, 0, b'I', b'I', 0x2a, 0, 8, 0, 0,
            0, 1, 0, 0x12, 1, 3, 0, 1, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0,
        ];
        jpeg.splice(2..2, exif_orientation_6);
        let file = tempfile::Builder::new()
            .suffix(".jpg")
            .tempfile()
            .map_err(candle_core::Error::wrap)?;
        std::fs::write(file.path(), jpeg).map_err(candle_core::Error::wrap)?;

        let decoded = decode_image_oriented(file.path())?;

        assert_eq!((decoded.width(), decoded.height()), (1, 2));
        Ok(())
    }
}
