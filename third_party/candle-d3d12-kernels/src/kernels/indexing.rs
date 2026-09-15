use crate::d3d12::{BufferBinding, Gpu, GpuBuffer, Pipelines};
use crate::err::D3D12KernelError;
use crate::source::Source;
use crate::utils::{copy2d_groups, linear_groups};

/// Index select: output[i] = input[indices[...] * right_size + ...]
#[allow(clippy::too_many_arguments)]
pub fn call_index_select(
    gpu: &Gpu,
    pipelines: &Pipelines,
    count: u32,
    left_size: u32,
    dim_size: u32,
    right_size: u32,
    input: &GpuBuffer,
    input_count: u32,
    indices: &GpuBuffer,
    indices_count: u32,
    output: &GpuBuffer,
) -> Result<(), D3D12KernelError> {
    let pso = pipelines.load_pipeline(gpu, Source::Indexing, "index_select_f32")?;
    let constants = [count, left_size, dim_size, right_size];
    let srvs = [
        BufferBinding::structured_f32(input, input_count),
        BufferBinding::structured_u32(indices, indices_count),
    ];
    let uav = BufferBinding::structured_f32(output, count);
    gpu.dispatch(&pso, &constants, &srvs, &uav, linear_groups(count))
}

/// Gather: output[i] = input[gather_index(i)]
#[allow(clippy::too_many_arguments)]
pub fn call_gather(
    gpu: &Gpu,
    pipelines: &Pipelines,
    count: u32,
    left_size: u32,
    dim_size: u32,
    right_size: u32,
    idx_dim_size: u32,
    input: &GpuBuffer,
    input_count: u32,
    indices: &GpuBuffer,
    indices_count: u32,
    output: &GpuBuffer,
) -> Result<(), D3D12KernelError> {
    let pso = pipelines.load_pipeline(gpu, Source::Indexing, "gather_f32")?;
    let constants = [count, left_size, dim_size, right_size, idx_dim_size];
    let srvs = [
        BufferBinding::structured_f32(input, input_count),
        BufferBinding::structured_u32(indices, indices_count),
    ];
    let uav = BufferBinding::structured_f32(output, count);
    gpu.dispatch(&pso, &constants, &srvs, &uav, linear_groups(count))
}

/// Strided copy: output[dst_offset + tid] = input[strided_index(tid)]
#[allow(clippy::too_many_arguments)]
pub fn call_copy_strided(
    gpu: &Gpu,
    pipelines: &Pipelines,
    count: u32,
    num_dims: u32,
    dst_offset: u32,
    input: &GpuBuffer,
    input_count: u32,
    meta_buffer: &GpuBuffer,
    output: &GpuBuffer,
    output_count: u32,
) -> Result<(), D3D12KernelError> {
    let pso = pipelines.load_pipeline(gpu, Source::Indexing, "copy_strided_f32")?;
    let constants = [count, num_dims, dst_offset];
    let srvs = [
        BufferBinding::structured_f32(input, input_count),
        BufferBinding::raw(meta_buffer),
    ];
    let uav = BufferBinding::structured_f32(output, output_count);
    gpu.dispatch(&pso, &constants, &srvs, &uav, linear_groups(count))
}

/// 2D copy with strides.
#[allow(clippy::too_many_arguments)]
pub fn call_copy2d(
    gpu: &Gpu,
    pipelines: &Pipelines,
    d1: u32,
    d2: u32,
    src_stride: u32,
    dst_stride: u32,
    src_offset: u32,
    dst_offset: u32,
    input: &GpuBuffer,
    input_count: u32,
    output: &GpuBuffer,
    output_count: u32,
) -> Result<(), D3D12KernelError> {
    let pso = pipelines.load_pipeline(gpu, Source::Indexing, "copy2d_f32")?;
    let constants = [d1, d2, src_stride, dst_stride, src_offset, dst_offset];
    let srvs = [BufferBinding::structured_f32(input, input_count)];
    let uav = BufferBinding::structured_f32(output, output_count);
    gpu.dispatch(&pso, &constants, &srvs, &uav, copy2d_groups(d1, d2))
}
