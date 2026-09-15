use crate::d3d12::{BufferBinding, Gpu, GpuBuffer, Pipelines};
use crate::err::D3D12KernelError;
use crate::source::Source;
use crate::utils::linear_groups;

pub fn call_fill_f32(
    gpu: &Gpu,
    pipelines: &Pipelines,
    count: u32,
    value: f32,
    output: &GpuBuffer,
) -> Result<(), D3D12KernelError> {
    let pso = pipelines.load_pipeline(gpu, Source::Fill, "fill_f32")?;
    let constants = [count, value.to_bits()];
    let uav = BufferBinding::structured_f32(output, count);
    gpu.dispatch(&pso, &constants, &[], &uav, linear_groups(count))
}

pub fn call_fill_u32(
    gpu: &Gpu,
    pipelines: &Pipelines,
    count: u32,
    value: f32,
    output: &GpuBuffer,
) -> Result<(), D3D12KernelError> {
    let pso = pipelines.load_pipeline(gpu, Source::Fill, "fill_u32")?;
    let constants = [count, value.to_bits()];
    let uav = BufferBinding::structured_u32(output, count);
    gpu.dispatch(&pso, &constants, &[], &uav, linear_groups(count))
}

pub fn call_fill_u8(
    gpu: &Gpu,
    pipelines: &Pipelines,
    count: u32,
    value: f32,
    output: &GpuBuffer,
) -> Result<(), D3D12KernelError> {
    let pso = pipelines.load_pipeline(gpu, Source::Fill, "fill_u8")?;
    let constants = [count, value.to_bits()];
    let uav = BufferBinding::raw(output);
    gpu.dispatch(&pso, &constants, &[], &uav, linear_groups(count))
}
