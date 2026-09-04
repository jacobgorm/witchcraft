use candle_core::{Context, DType, Device, Result, Tensor, D};
use candle_nn::{
    batch_norm, conv2d, conv2d_no_bias, linear, linear_no_bias, ops::sigmoid, ops::softmax,
    BatchNorm, Conv2d, Conv2dConfig, Linear, VarBuilder,
};
use candle_transformers::models::fastvit::Config;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct ProfileRecord {
    pub label: String,
    pub elapsed: Duration,
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug, Default)]
pub struct ImageProfile {
    records: Vec<ProfileRecord>,
}

impl ImageProfile {
    pub fn records(&self) -> &[ProfileRecord] {
        &self.records
    }

    pub fn total(&self) -> Duration {
        self.records
            .iter()
            .fold(Duration::ZERO, |acc, record| acc + record.elapsed)
    }

    pub fn record(&mut self, label: impl Into<String>, elapsed: Duration, shape: Vec<usize>) {
        self.records.push(ProfileRecord {
            label: label.into(),
            elapsed,
            shape,
        });
    }

    pub fn time<F>(&mut self, device: &Device, label: impl Into<String>, f: F) -> Result<Tensor>
    where
        F: FnOnce() -> Result<Tensor>,
    {
        let label = label.into();
        device.synchronize()?;
        let started = Instant::now();
        let output = f()?;
        device.synchronize()?;
        self.records.push(ProfileRecord {
            label,
            elapsed: started.elapsed(),
            shape: output.dims().to_vec(),
        });
        Ok(output)
    }
}

#[derive(Clone, Debug)]
struct ConvNorm {
    conv: Conv2d,
}

impl ConvNorm {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv2d_cfg = Conv2dConfig {
            stride,
            padding: kernel / 2,
            groups: in_channels,
            ..Default::default()
        };
        let bn = batch_norm(out_channels, 1e-5, vb.pp("bn"))?;
        let conv = conv2d_no_bias(in_channels, out_channels, kernel, conv2d_cfg, vb.pp("conv"))?;
        let conv = conv.absorb_bn(&bn)?;
        Ok(Self { conv })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.conv)
    }
}

#[derive(Clone, Debug)]
struct SqueezeExcitation {
    fc1: Conv2d,
    fc2: Conv2d,
}

impl SqueezeExcitation {
    fn new(in_channels: usize, squeeze_channels: usize, vb: VarBuilder) -> Result<Self> {
        let conv2d_cfg = Conv2dConfig {
            ..Default::default()
        };
        let fc1 = conv2d(in_channels, squeeze_channels, 1, conv2d_cfg, vb.pp("fc1"))?;
        let fc2 = conv2d(squeeze_channels, in_channels, 1, conv2d_cfg, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = xs.mean_keepdim(D::Minus2)?.mean_keepdim(D::Minus1)?;
        let xs = sigmoid(&xs.apply(&self.fc1)?.relu()?.apply(&self.fc2)?)?;
        residual.broadcast_mul(&xs)
    }
}

#[derive(Clone, Debug)]
struct MobileOneBlock {
    reparam_conv: Conv2d,
    se: Option<SqueezeExcitation>,
    use_act: bool,
}

impl MobileOneBlock {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        group_size: usize,
        use_act: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let groups = in_channels.checked_div(group_size).unwrap_or(1);
        let padding = kernel / 2;
        let conv2d_cfg = Conv2dConfig {
            stride,
            groups,
            padding,
            ..Default::default()
        };

        let mut w = Tensor::zeros(
            (out_channels, in_channels / groups, kernel, kernel),
            DType::F32,
            vb.device(),
        )?;
        let mut b = Tensor::zeros(out_channels, DType::F32, vb.device())?;

        let conv_kxk_bn = batch_norm(out_channels, 1e-5, vb.pp("conv_kxk.0.bn"));
        let conv_kxk = conv2d_no_bias(
            in_channels,
            out_channels,
            kernel,
            conv2d_cfg,
            vb.pp("conv_kxk.0.conv"),
        );
        if let (Ok(conv), Ok(bn)) = (conv_kxk, conv_kxk_bn) {
            let (wk, bk) = fuse_conv_bn(conv.weight(), bn)?;
            w = (w + wk)?;
            b = (b + bk)?;
        }

        let conv_scale_bn = batch_norm(out_channels, 1e-5, vb.pp("conv_scale.bn"));
        let conv_scale = conv2d_no_bias(
            in_channels,
            out_channels,
            1,
            conv2d_cfg,
            vb.pp("conv_scale.conv"),
        );
        if let (Ok(conv), Ok(bn)) = (conv_scale, conv_scale_bn) {
            let (ws, bs) = fuse_conv_bn(conv.weight(), bn)?;
            let ws = ws
                .pad_with_zeros(D::Minus1, 1, 1)?
                .pad_with_zeros(D::Minus2, 1, 1)?;
            w = (w + ws)?;
            b = (b + bs)?;
        }

        let identity_bn = batch_norm(out_channels, 1e-5, vb.pp("identity"));
        if let Ok(id_bn) = identity_bn {
            let mut weights: Vec<f32> = vec![0.0; w.elem_count()];
            let id = in_channels / groups;
            for i in 0..in_channels {
                if kernel > 1 {
                    weights[i * kernel * kernel + 4] = 1.0;
                } else {
                    weights[i * (id + 1)] = 1.0;
                }
            }
            let weights = &Tensor::from_vec(weights, w.shape(), w.device())?;
            let (wi, bi) = fuse_conv_bn(weights, id_bn)?;
            w = (w + wi)?;
            b = (b + bi)?;
        }

        let se = SqueezeExcitation::new(out_channels, out_channels / 16, vb.pp("se")).ok();
        Ok(Self {
            reparam_conv: Conv2d::new(w, Some(b), conv2d_cfg),
            se,
            use_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.apply(&self.reparam_conv)?;
        if let Some(se) = &self.se {
            xs = se.forward(&xs)?;
        }
        if self.use_act {
            xs = xs.gelu_erf()?;
        }
        Ok(xs)
    }

    fn forward_profiled(
        &self,
        xs: &Tensor,
        label: &str,
        profile: &mut ImageProfile,
        device: &Device,
    ) -> Result<Tensor> {
        let mut xs = profile.time(device, format!("{label}.conv"), || {
            xs.apply(&self.reparam_conv)
        })?;
        if let Some(se) = &self.se {
            xs = profile.time(device, format!("{label}.se"), || se.forward(&xs))?;
        }
        if self.use_act {
            xs = profile.time(device, format!("{label}.gelu"), || xs.gelu_erf())?;
        }
        Ok(xs)
    }
}

#[derive(Clone, Debug)]
struct Repmixer {
    gamma: Tensor,
    norm: MobileOneBlock,
    mixer: MobileOneBlock,
}

impl Repmixer {
    fn new(dim: usize, kernel: usize, vb: VarBuilder) -> Result<Self> {
        let gamma = vb.get((dim, 1, 1), "layer_scale.gamma")?;
        let norm = MobileOneBlock::new(dim, dim, kernel, 1, 1, false, vb.pp("norm"))?;
        let mixer = MobileOneBlock::new(dim, dim, kernel, 1, 1, false, vb.pp("mixer"))?;
        Ok(Self { gamma, norm, mixer })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = (self.mixer.forward(xs)? - self.norm.forward(xs)?)?;
        let xs = xs.broadcast_mul(&self.gamma.reshape((1, (), 1, 1))?)?;
        xs + residual
    }

    fn forward_profiled(
        &self,
        xs: &Tensor,
        label: &str,
        profile: &mut ImageProfile,
        device: &Device,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let mixer = self
            .mixer
            .forward_profiled(xs, &format!("{label}.mixer"), profile, device)?;
        let norm = self
            .norm
            .forward_profiled(xs, &format!("{label}.norm"), profile, device)?;
        profile.time(device, format!("{label}.scale_residual"), || {
            let xs = (mixer - norm)?;
            let xs = xs.broadcast_mul(&self.gamma.reshape((1, (), 1, 1))?)?;
            xs + residual
        })
    }
}

#[derive(Clone, Debug)]
struct ConvMlp {
    conv: ConvNorm,
    fc1: Conv2d,
    fc2: Conv2d,
}

impl ConvMlp {
    fn new(dim: usize, exp_ratio: usize, vb: VarBuilder) -> Result<Self> {
        let conv2d_cfg = Conv2dConfig {
            ..Default::default()
        };
        let conv = ConvNorm::new(dim, dim, 7, 1, vb.pp("conv"))?;
        let fc1 = conv2d(dim, dim * exp_ratio, 1, conv2d_cfg, vb.pp("fc1"))?;
        let fc2 = conv2d(dim * exp_ratio, dim, 1, conv2d_cfg, vb.pp("fc2"))?;
        Ok(Self { conv, fc1, fc2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.conv
            .forward(xs)?
            .apply(&self.fc1)?
            .gelu_erf()?
            .apply(&self.fc2)
    }

    fn forward_profiled(
        &self,
        xs: &Tensor,
        label: &str,
        profile: &mut ImageProfile,
        device: &Device,
    ) -> Result<Tensor> {
        let xs = profile.time(device, format!("{label}.conv"), || self.conv.forward(xs))?;
        let xs = profile.time(device, format!("{label}.fc1"), || xs.apply(&self.fc1))?;
        let xs = profile.time(device, format!("{label}.gelu"), || xs.gelu_erf())?;
        profile.time(device, format!("{label}.fc2"), || xs.apply(&self.fc2))
    }
}

#[derive(Clone, Debug)]
struct PositionalEncoding {
    conv: Conv2d,
}

impl PositionalEncoding {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let conv2d_cfg = Conv2dConfig {
            stride: 1,
            padding: 3,
            groups: dim,
            ..Default::default()
        };
        let conv = conv2d(dim, dim, 7, conv2d_cfg, vb.pp("pos_enc"))?;
        Ok(Self { conv })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs + xs.apply(&self.conv)?
    }

    fn forward_profiled(
        &self,
        xs: &Tensor,
        label: &str,
        profile: &mut ImageProfile,
        device: &Device,
    ) -> Result<Tensor> {
        profile.time(device, label, || self.forward(xs))
    }
}

#[derive(Clone, Debug)]
struct Attention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let qkv = linear_no_bias(dim, dim * 3, vb.pp("qkv"))?;
        let proj = linear(dim, dim, vb.pp("proj"))?;
        let head_dim = 32;
        let num_heads = dim / head_dim;
        let scale = (head_dim as f64).powf(-0.5);
        Ok(Self {
            qkv,
            proj,
            num_heads,
            head_dim,
            scale,
        })
    }

    fn qkv(&self, xs: &Tensor) -> Result<(Tensor, usize, usize, usize, usize)> {
        let (b, c, h, w) = xs.dims4()?;
        let n = h * w;
        let xs = xs.flatten_from(2)?.transpose(D::Minus1, D::Minus2)?;
        let qkv = xs
            .apply(&self.qkv)?
            .reshape((b, n, 3, self.num_heads, self.head_dim))?
            .permute((2, 0, 3, 1, 4))?;
        Ok((qkv, b, c, h, w))
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (qkv, b, c, h, w) = self.qkv(xs)?;
        let q = (qkv.get(0)? * self.scale)?;
        let k = qkv.get(1)?;
        let v = qkv.get(2)?;
        let att = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let att = softmax(&att, D::Minus1)?;
        let xs = att.matmul(&v)?;
        let xs = xs.transpose(1, 2)?.reshape((b, h * w, c))?;
        let xs = xs.apply(&self.proj)?;
        xs.transpose(D::Minus1, D::Minus2)?.reshape((b, c, h, w))
    }

    fn forward_profiled(
        &self,
        xs: &Tensor,
        label: &str,
        profile: &mut ImageProfile,
        device: &Device,
    ) -> Result<Tensor> {
        let (b, c, h, w) = xs.dims4()?;
        let n = h * w;
        let qkv = profile.time(device, format!("{label}.qkv"), || {
            xs.flatten_from(2)?
                .transpose(D::Minus1, D::Minus2)?
                .apply(&self.qkv)?
                .reshape((b, n, 3, self.num_heads, self.head_dim))?
                .permute((2, 0, 3, 1, 4))
        })?;
        let q = profile.time(device, format!("{label}.q_scale"), || {
            qkv.get(0)? * self.scale
        })?;
        let k = qkv.get(1)?;
        let v = qkv.get(2)?;
        let att = profile.time(device, format!("{label}.qk_matmul"), || {
            q.matmul(&k.transpose(D::Minus2, D::Minus1)?)
        })?;
        let att = profile.time(device, format!("{label}.softmax"), || {
            softmax(&att, D::Minus1)
        })?;
        let xs = profile.time(device, format!("{label}.av_matmul"), || att.matmul(&v))?;
        let xs = xs.transpose(1, 2)?.reshape((b, h * w, c))?;
        let xs = profile.time(device, format!("{label}.proj"), || xs.apply(&self.proj))?;
        profile.time(device, format!("{label}.reshape_out"), || {
            xs.transpose(D::Minus1, D::Minus2)?.reshape((b, c, h, w))
        })
    }
}

#[derive(Clone, Debug)]
enum FastVitBlock {
    Repmixer {
        token_mixer: Repmixer,
        gamma: Tensor,
        mlp: ConvMlp,
    },
    Attention {
        gamma1: Tensor,
        gamma2: Tensor,
        norm: BatchNorm,
        token_mixer: Attention,
        mlp: ConvMlp,
    },
}

impl FastVitBlock {
    fn repmixer(dim: usize, exp_ratio: usize, vb: VarBuilder) -> Result<Self> {
        let gamma = vb.get((dim, 1, 1), "layer_scale.gamma")?;
        let token_mixer = Repmixer::new(dim, 3, vb.pp("token_mixer"))?;
        let mlp = ConvMlp::new(dim, exp_ratio, vb.pp("mlp"))?;
        Ok(Self::Repmixer {
            token_mixer,
            gamma,
            mlp,
        })
    }

    fn attention(dim: usize, exp_ratio: usize, vb: VarBuilder) -> Result<Self> {
        let gamma1 = vb.get((dim, 1, 1), "layer_scale_1.gamma")?;
        let gamma2 = vb.get((dim, 1, 1), "layer_scale_2.gamma")?;
        let norm = batch_norm(dim, 1e-5, vb.pp("norm"))?;
        let token_mixer = Attention::new(dim, vb.pp("token_mixer"))?;
        let mlp = ConvMlp::new(dim, exp_ratio, vb.pp("mlp"))?;
        Ok(Self::Attention {
            gamma1,
            gamma2,
            norm,
            token_mixer,
            mlp,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Repmixer {
                token_mixer,
                gamma,
                mlp,
            } => {
                let residual = token_mixer.forward(xs)?;
                let xs = mlp
                    .forward(&residual)?
                    .broadcast_mul(&gamma.reshape((1, (), 1, 1))?)?;
                xs + residual
            }
            Self::Attention {
                gamma1,
                gamma2,
                norm,
                token_mixer,
                mlp,
            } => {
                let xs = xs.clone();
                let xs = (&xs
                    + &token_mixer
                        .forward(&xs.apply_t(norm, false)?)?
                        .broadcast_mul(&gamma1.reshape((1, (), 1, 1))?)?)?;
                let xs = (&xs
                    + &mlp
                        .forward(&xs)?
                        .broadcast_mul(&gamma2.reshape((1, (), 1, 1))?)?)?;
                Ok(xs)
            }
        }
    }

    fn forward_profiled(
        &self,
        xs: &Tensor,
        label: &str,
        profile: &mut ImageProfile,
        device: &Device,
    ) -> Result<Tensor> {
        match self {
            Self::Repmixer {
                token_mixer,
                gamma,
                mlp,
            } => {
                let residual = token_mixer.forward_profiled(
                    xs,
                    &format!("{label}.token_mixer"),
                    profile,
                    device,
                )?;
                let mlp_out =
                    mlp.forward_profiled(&residual, &format!("{label}.mlp"), profile, device)?;
                profile.time(device, format!("{label}.mlp_residual"), || {
                    let xs = mlp_out.broadcast_mul(&gamma.reshape((1, (), 1, 1))?)?;
                    xs + residual
                })
            }
            Self::Attention {
                gamma1,
                gamma2,
                norm,
                token_mixer,
                mlp,
            } => {
                let residual = xs.clone();
                let xs = profile.time(device, format!("{label}.norm"), || {
                    residual.apply_t(norm, false)
                })?;
                let mixed = token_mixer.forward_profiled(
                    &xs,
                    &format!("{label}.token_mixer"),
                    profile,
                    device,
                )?;
                let xs = profile.time(device, format!("{label}.attn_residual"), || {
                    &residual + &mixed.broadcast_mul(&gamma1.reshape((1, (), 1, 1))?)?
                })?;
                let mlp_out =
                    mlp.forward_profiled(&xs, &format!("{label}.mlp"), profile, device)?;
                profile.time(device, format!("{label}.mlp_residual"), || {
                    &xs + &mlp_out.broadcast_mul(&gamma2.reshape((1, (), 1, 1))?)?
                })
            }
        }
    }
}

#[derive(Clone, Debug)]
struct PatchEmbed {
    lk: ConvNorm,
    sk: ConvNorm,
    se: Option<SqueezeExcitation>,
    use_act: bool,
    mb: MobileOneBlock,
}

impl PatchEmbed {
    fn new(in_channels: usize, out_channels: usize, use_act: bool, vb: VarBuilder) -> Result<Self> {
        let lk = ConvNorm::new(in_channels, out_channels, 7, 2, vb.pp("proj.0.large_conv"))?;
        let sk = ConvNorm::new(in_channels, out_channels, 3, 2, vb.pp("proj.0.small_conv"))?;
        let se = SqueezeExcitation::new(out_channels, out_channels / 4, vb.pp("proj.0.se")).ok();
        let mb = MobileOneBlock::new(out_channels, out_channels, 1, 1, 0, true, vb.pp("proj.1"))?;
        Ok(Self {
            lk,
            sk,
            se,
            use_act,
            mb,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = (self.lk.forward(xs)? + self.sk.forward(xs)?)?;
        if let Some(se) = &self.se {
            xs = se.forward(&xs)?;
        }
        if self.use_act {
            xs = xs.gelu_erf()?;
        }
        self.mb.forward(&xs)
    }

    fn forward_profiled(
        &self,
        xs: &Tensor,
        label: &str,
        profile: &mut ImageProfile,
        device: &Device,
    ) -> Result<Tensor> {
        let lk = profile.time(device, format!("{label}.large_conv"), || {
            self.lk.forward(xs)
        })?;
        let sk = profile.time(device, format!("{label}.small_conv"), || {
            self.sk.forward(xs)
        })?;
        let mut xs = profile.time(device, format!("{label}.merge"), || lk + sk)?;
        if let Some(se) = &self.se {
            xs = profile.time(device, format!("{label}.se"), || se.forward(&xs))?;
        }
        if self.use_act {
            xs = profile.time(device, format!("{label}.gelu"), || xs.gelu_erf())?;
        }
        self.mb
            .forward_profiled(&xs, &format!("{label}.mobileone"), profile, device)
    }
}

#[derive(Clone, Debug)]
struct Stem {
    mb0: MobileOneBlock,
    mb1: MobileOneBlock,
    mb2: MobileOneBlock,
}

impl Stem {
    fn new(in_channels: usize, out_channels: usize, vb: VarBuilder) -> Result<Self> {
        let mb0 = MobileOneBlock::new(in_channels, out_channels, 3, 2, 0, true, vb.pp(0))?;
        let mb1 = MobileOneBlock::new(out_channels, out_channels, 3, 2, 1, true, vb.pp(1))?;
        let mb2 = MobileOneBlock::new(out_channels, out_channels, 1, 1, 0, true, vb.pp(2))?;
        Ok(Self { mb0, mb1, mb2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.mb2.forward(&self.mb1.forward(&self.mb0.forward(xs)?)?)
    }

    fn forward_profiled(
        &self,
        xs: &Tensor,
        profile: &mut ImageProfile,
        device: &Device,
    ) -> Result<Tensor> {
        let xs = self.mb0.forward_profiled(xs, "stem.0", profile, device)?;
        let xs = self.mb1.forward_profiled(&xs, "stem.1", profile, device)?;
        self.mb2.forward_profiled(&xs, "stem.2", profile, device)
    }
}

#[derive(Clone, Debug)]
struct Stage {
    downsample: Option<PatchEmbed>,
    pos_emb: Option<PositionalEncoding>,
    blocks: Vec<FastVitBlock>,
}

impl Stage {
    fn new(cfg: &Config, idx: usize, vb: VarBuilder) -> Result<Self> {
        let nblocks = cfg.blocks[idx];
        let dim = cfg.in_channels << idx;
        let downsample = PatchEmbed::new(dim / 2, dim, cfg.lkc_use_act, vb.pp("downsample")).ok();
        let pos_emb = PositionalEncoding::new(dim, vb.pp("pos_emb")).ok();
        let mut blocks = Vec::with_capacity(nblocks);
        for block_idx in 0..nblocks {
            let block = if cfg.attn && idx == 3 {
                FastVitBlock::attention(dim, cfg.exp_ratio, vb.pp(format!("blocks.{block_idx}")))?
            } else {
                FastVitBlock::repmixer(dim, cfg.exp_ratio, vb.pp(format!("blocks.{block_idx}")))?
            };
            blocks.push(block);
        }
        Ok(Self {
            downsample,
            pos_emb,
            blocks,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        if let Some(downsample) = &self.downsample {
            xs = downsample.forward(&xs)?;
        }
        if let Some(pos_emb) = &self.pos_emb {
            xs = pos_emb.forward(&xs)?;
        }
        for block in &self.blocks {
            xs = block.forward(&xs)?;
        }
        Ok(xs)
    }

    fn forward_profiled(
        &self,
        xs: &Tensor,
        idx: usize,
        profile: &mut ImageProfile,
        device: &Device,
    ) -> Result<Tensor> {
        let label = format!("stages.{idx}");
        let mut xs = xs.clone();
        if let Some(downsample) = &self.downsample {
            xs = downsample.forward_profiled(
                &xs,
                &format!("{label}.downsample"),
                profile,
                device,
            )?;
        }
        if let Some(pos_emb) = &self.pos_emb {
            xs = pos_emb.forward_profiled(&xs, &format!("{label}.pos_emb"), profile, device)?;
        }
        for (block_idx, block) in self.blocks.iter().enumerate() {
            xs = block.forward_profiled(
                &xs,
                &format!("{label}.blocks.{block_idx}"),
                profile,
                device,
            )?;
        }
        Ok(xs)
    }
}

#[derive(Clone, Debug)]
pub struct ProfiledFastVit {
    stem: Stem,
    stages: Vec<Stage>,
    final_conv: MobileOneBlock,
    cls: Linear,
    device: Device,
}

impl ProfiledFastVit {
    pub fn new(cfg: &Config, nclasses: usize, vb: VarBuilder) -> Result<Self> {
        let stem = Stem::new(3, cfg.in_channels, vb.pp("stem"))?;
        let stages_vb = vb.pp("stages");
        let stages = vec![
            Stage::new(cfg, 0, stages_vb.pp(0))?,
            Stage::new(cfg, 1, stages_vb.pp(1))?,
            Stage::new(cfg, 2, stages_vb.pp(2))?,
            Stage::new(cfg, 3, stages_vb.pp(3))?,
        ];
        let final_conv = MobileOneBlock::new(
            cfg.in_channels * 8,
            cfg.in_channels * 16,
            3,
            1,
            1,
            true,
            vb.pp("final_conv"),
        )?;
        let cls = linear(cfg.in_channels * 16, nclasses, vb.pp("head.fc"))?;
        Ok(Self {
            stem,
            stages,
            final_conv,
            cls,
            device: vb.device().clone(),
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = self.stem.forward(xs)?;
        for stage in &self.stages {
            xs = stage.forward(&xs)?;
        }
        xs = self.final_conv.forward(&xs)?;
        xs.mean(D::Minus2)?.mean(D::Minus1)?.apply(&self.cls)
    }

    pub fn forward_profiled(&self, xs: &Tensor) -> Result<(Tensor, ImageProfile)> {
        let mut profile = ImageProfile::default();
        let mut xs = self.stem.forward_profiled(xs, &mut profile, &self.device)?;
        for (idx, stage) in self.stages.iter().enumerate() {
            xs = stage.forward_profiled(&xs, idx, &mut profile, &self.device)?;
        }
        xs = self
            .final_conv
            .forward_profiled(&xs, "final_conv", &mut profile, &self.device)?;
        let xs = profile.time(&self.device, "head.pool", || {
            xs.mean(D::Minus2)?.mean(D::Minus1)
        })?;
        let xs = profile.time(&self.device, "head.fc", || xs.apply(&self.cls))?;
        Ok((xs, profile))
    }
}

fn fuse_conv_bn(weights: &Tensor, bn: BatchNorm) -> Result<(Tensor, Tensor)> {
    let (gamma, beta) = bn.weight_and_bias().context("no weight-bias")?;
    let mu = bn.running_mean();
    let sigma = (bn.running_var() + bn.eps())?.sqrt();
    let gps = (gamma / sigma)?;
    let bias = (beta - mu * &gps)?;
    let weights = weights.broadcast_mul(&gps.reshape(((), 1, 1, 1))?)?;
    Ok((weights, bias))
}
