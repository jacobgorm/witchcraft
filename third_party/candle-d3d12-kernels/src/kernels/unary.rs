use crate::d3d12::{BufferBinding, Gpu, GpuBuffer, Pipelines};
use crate::err::D3D12KernelError;
use crate::source::Source;
use crate::utils::linear_groups;

/// Dispatch a contiguous unary kernel.
/// The kernel reads from `input` and writes to `output`.
/// The entry point name should be like "copy_f32", "neg_f32", "exp_f32", etc.
pub fn call_unary_contiguous(
    gpu: &Gpu,
    pipelines: &Pipelines,
    entry_point: &str,
    count: u32,
    input: &GpuBuffer,
    output: &GpuBuffer,
) -> Result<(), D3D12KernelError> {
    let pso = pipelines.load_pipeline(gpu, Source::Unary, entry_point)?;
    let constants = [count];
    let srvs = [BufferBinding::structured_f32(input, count)];
    let uav = BufferBinding::structured_f32(output, count);
    gpu.dispatch(&pso, &constants, &srvs, &uav, linear_groups(count))
}

/// Dispatch a strided unary kernel (copy_f32_strided).
/// `meta_buffer` contains packed dims[] followed by strides[] as u32 arrays.
pub fn call_unary_strided(
    gpu: &Gpu,
    pipelines: &Pipelines,
    entry_point: &str,
    count: u32,
    num_dims: u32,
    input: &GpuBuffer,
    input_count: u32,
    meta_buffer: &GpuBuffer,
    output: &GpuBuffer,
) -> Result<(), D3D12KernelError> {
    let pso = pipelines.load_pipeline(gpu, Source::Unary, entry_point)?;
    let constants = [count, num_dims];
    let srvs = [
        BufferBinding::structured_f32(input, input_count),
        BufferBinding::raw(meta_buffer),
    ];
    let uav = BufferBinding::structured_f32(output, count);
    gpu.dispatch(&pso, &constants, &srvs, &uav, linear_groups(count))
}
