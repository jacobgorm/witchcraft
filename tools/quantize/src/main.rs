use std::env;
use std::path::PathBuf;

use candle_core::Device;
use candle_core::quantized::{GgmlDType, QTensor, gguf_file};

fn run_quantize_safetensors(
    in_file: PathBuf,
    out_path: PathBuf,
    prefix: Option<&str>,
) -> anyhow::Result<()> {
    let bytes = std::fs::read(&in_file)?;
    let (_, header) = safetensors::SafeTensors::read_metadata(&bytes)?;
    let metadata = header
        .metadata()
        .as_ref()
        .into_iter()
        .flat_map(|metadata| metadata.iter())
        .map(|(key, value)| (key.clone(), gguf_file::Value::String(value.clone())))
        .collect::<Vec<_>>();
    let tensors = candle_core::safetensors::load_buffer(&bytes, &Device::Cpu)?;
    let tensors = tensors
        .into_iter()
        .filter(|(name, tensor)| {
            tensor.rank() > 0 && prefix.is_none_or(|prefix| name.starts_with(prefix))
        })
        .collect::<Vec<_>>();
    println!("tensors: {}", tensors.len());

    let mut qtensors = Vec::with_capacity(tensors.len());
    for (name, tensor) in tensors {
        let qdtype = if name.contains("token_gate") {
            GgmlDType::F32
        } else if tensor.rank() == 2 {
            let last_dim = tensor.dim(1)?;
            if last_dim % GgmlDType::Q4K.block_size() == 0 {
                GgmlDType::Q4K
            } else if last_dim % GgmlDType::Q4_1.block_size() == 0 {
                GgmlDType::Q4_1
            } else {
                GgmlDType::F32
            }
        } else {
            GgmlDType::F32
        };
        qtensors.push((name, QTensor::quantize(&tensor, qdtype)?));
    }
    let qtensors = qtensors
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect::<Vec<_>>();
    let metadata = metadata
        .iter()
        .map(|(key, value)| (key.as_str(), value))
        .collect::<Vec<_>>();

    let mut out = std::fs::File::create(&out_path)?;
    gguf_file::write(&mut out, &metadata, &qtensors)?;
    out.sync_all()?;
    println!("wrote {}", out_path.display());
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = env::args().collect::<Vec<_>>();
    let (prefix, in_file, out_file) = match args.as_slice() {
        [_, in_file, out_file] => (None, in_file, out_file),
        [_, flag, prefix, in_file, out_file] if flag == "--prefix" => {
            (Some(prefix.as_str()), in_file, out_file)
        }
        _ => anyhow::bail!(
            "usage: {} [--prefix TENSOR_PREFIX] IN.safetensors OUT.gguf",
            args[0]
        ),
    };
    if prefix.is_some_and(str::is_empty) {
        anyhow::bail!("tensor prefix cannot be empty");
    }
    run_quantize_safetensors(PathBuf::from(in_file), PathBuf::from(out_file), prefix)?;
    Ok(())
}
