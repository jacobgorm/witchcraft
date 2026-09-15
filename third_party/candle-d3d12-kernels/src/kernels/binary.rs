use crate::d3d12::{BufferBinding, Gpu, GpuBuffer, Pipelines};
use crate::err::D3D12KernelError;
use crate::source::Source;
use crate::utils::linear_groups;

/// Dispatch a contiguous binary kernel (add_f32, sub_f32, mul_f32, div_f32, etc.).
pub fn call_binary_contiguous(
    gpu: &Gpu,
    pipelines: &Pipelines,
    entry_point: &str,
    count: u32,
    lhs: &GpuBuffer,
    rhs: &GpuBuffer,
    output: &GpuBuffer,
) -> Result<(), D3D12KernelError> {
    let pso = pipelines.load_pipeline(gpu, Source::Binary, entry_point)?;
    let constants = [count];
    let srvs = [
        BufferBinding::structured_f32(lhs, count),
        BufferBinding::structured_f32(rhs, count),
    ];
    let uav = BufferBinding::structured_f32(output, count);
    gpu.dispatch(&pso, &constants, &srvs, &uav, linear_groups(count))
}

/// Dispatch a strided binary kernel (add_f32_strided, etc.).
/// `lhs_meta` and `rhs_meta` each contain packed dims[] then strides[] as u32 arrays.
pub fn call_binary_strided(
    gpu: &Gpu,
    pipelines: &Pipelines,
    entry_point: &str,
    count: u32,
    num_dims: u32,
    lhs: &GpuBuffer,
    lhs_count: u32,
    rhs: &GpuBuffer,
    rhs_count: u32,
    lhs_meta: &GpuBuffer,
    rhs_meta: &GpuBuffer,
    output: &GpuBuffer,
) -> Result<(), D3D12KernelError> {
    let pso = pipelines.load_pipeline(gpu, Source::Binary, entry_point)?;
    let constants = [count, num_dims];
    let srvs = [
        BufferBinding::structured_f32(lhs, lhs_count),
        BufferBinding::structured_f32(rhs, rhs_count),
        BufferBinding::raw(lhs_meta),
        BufferBinding::raw(rhs_meta),
    ];
    let uav = BufferBinding::structured_f32(output, count);
    gpu.dispatch(&pso, &constants, &srvs, &uav, linear_groups(count))
}
