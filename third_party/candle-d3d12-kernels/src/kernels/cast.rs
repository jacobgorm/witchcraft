use crate::d3d12::{BufferBinding, Gpu, GpuBuffer, Pipelines};
use crate::err::D3D12KernelError;
use crate::source::Source;
use crate::utils::linear_groups;

/// Cast f32 -> u32
pub fn call_cast_f32_to_u32(
    gpu: &Gpu,
    pipelines: &Pipelines,
    count: u32,
    input: &GpuBuffer,
    output: &GpuBuffer,
) -> Result<(), D3D12KernelError> {
    let pso = pipelines.load_pipeline(gpu, Source::Cast, "cast_f32_to_u32")?;
    let constants = [count];
    let srvs = [BufferBinding::structured_f32(input, count)];
    let uav = BufferBinding::structured_u32(output, count);
    gpu.dispatch(&pso, &constants, &srvs, &uav, linear_groups(count))
}

/// Cast u32 -> f32
pub fn call_cast_u32_to_f32(
    gpu: &Gpu,
    pipelines: &Pipelines,
    count: u32,
    input: &GpuBuffer,
    output: &GpuBuffer,
) -> Result<(), D3D12KernelError> {
    let pso = pipelines.load_pipeline(gpu, Source::Cast, "cast_u32_to_f32")?;
    let constants = [count];
    let srvs = [BufferBinding::structured_u32(input, count)];
    let uav = BufferBinding::structured_f32(output, count);
    gpu.dispatch(&pso, &constants, &srvs, &uav, linear_groups(count))
}

/// Cast f64 -> f32 (input is raw ByteAddressBuffer for 8-byte elements)
pub fn call_cast_f64_to_f32(
    gpu: &Gpu,
    pipelines: &Pipelines,
    count: u32,
    input: &GpuBuffer,
    output: &GpuBuffer,
) -> Result<(), D3D12KernelError> {
    let pso = pipelines.load_pipeline(gpu, Source::Cast, "cast_f64_to_f32")?;
    let constants = [count];
    let srvs = [BufferBinding::raw(input)];
    let uav = BufferBinding::structured_f32(output, count);
    gpu.dispatch(&pso, &constants, &srvs, &uav, linear_groups(count))
}
