use crate::d3d12::{BufferBinding, Gpu, GpuBuffer, Pipelines};
use crate::err::D3D12KernelError;
use crate::source::Source;
use crate::utils::linear_groups;

/// Contiguous affine: output[i] = input[i] * mul + add
pub fn call_affine(
    gpu: &Gpu,
    pipelines: &Pipelines,
    count: u32,
    input: &GpuBuffer,
    output: &GpuBuffer,
    mul: f32,
    add: f32,
) -> Result<(), D3D12KernelError> {
    let pso = pipelines.load_pipeline(gpu, Source::Affine, "affine_f32")?;
    let constants = [count, mul.to_bits(), add.to_bits()];
    let srvs = [BufferBinding::structured_f32(input, count)];
    let uav = BufferBinding::structured_f32(output, count);
    gpu.dispatch(&pso, &constants, &srvs, &uav, linear_groups(count))
}

/// Strided affine: output[i] = input[strided(i)] * mul + add
/// `meta_buffer` contains packed dims[] then strides[] as u32 arrays.
pub fn call_affine_strided(
    gpu: &Gpu,
    pipelines: &Pipelines,
    count: u32,
    num_dims: u32,
    input: &GpuBuffer,
    input_count: u32,
    meta_buffer: &GpuBuffer,
    output: &GpuBuffer,
    mul: f32,
    add: f32,
) -> Result<(), D3D12KernelError> {
    let pso = pipelines.load_pipeline(gpu, Source::Affine, "affine_f32_strided")?;
    let constants = [count, mul.to_bits(), add.to_bits(), num_dims];
    let srvs = [
        BufferBinding::structured_f32(input, input_count),
        BufferBinding::raw(meta_buffer),
    ];
    let uav = BufferBinding::structured_f32(output, count);
    gpu.dispatch(&pso, &constants, &srvs, &uav, linear_groups(count))
}
