use crate::d3d12::{BufferBinding, Gpu, GpuBuffer, Pipelines};
use crate::err::D3D12KernelError;
use crate::source::Source;

/// Reduction dispatch (sum_f32, mean_f32, max_f32, min_f32).
/// Dispatches `out_length` thread groups, each reducing `work_per_group` elements.
pub fn call_reduce(
    gpu: &Gpu,
    pipelines: &Pipelines,
    entry_point: &str,
    total_length: u32,
    out_length: u32,
    input: &GpuBuffer,
    output: &GpuBuffer,
) -> Result<(), D3D12KernelError> {
    let pso = pipelines.load_pipeline(gpu, Source::Reduce, entry_point)?;
    let work_per_group = total_length / out_length;
    let constants = [total_length, out_length, work_per_group, 0];
    let srvs = [BufferBinding::structured_f32(input, total_length)];
    let uav = BufferBinding::structured_f32(output, out_length);
    let groups = [out_length, 1, 1];
    gpu.dispatch(&pso, &constants, &srvs, &uav, groups)
}

/// Argmax/argmin dispatch. Output is u32 indices.
pub fn call_arg_reduce(
    gpu: &Gpu,
    pipelines: &Pipelines,
    entry_point: &str,
    total_length: u32,
    out_length: u32,
    input: &GpuBuffer,
    output: &GpuBuffer,
) -> Result<(), D3D12KernelError> {
    let pso = pipelines.load_pipeline(gpu, Source::Reduce, entry_point)?;
    let work_per_group = total_length / out_length;
    let constants = [total_length, out_length, work_per_group, 0];
    let srvs = [BufferBinding::structured_f32(input, total_length)];
    let uav = BufferBinding::structured_u32(output, out_length);
    let groups = [out_length, 1, 1];
    gpu.dispatch(&pso, &constants, &srvs, &uav, groups)
}

/// Softmax over the last dimension.
/// Each thread group handles one row of `row_size` elements.
pub fn call_softmax(
    gpu: &Gpu,
    pipelines: &Pipelines,
    row_size: u32,
    num_rows: u32,
    input: &GpuBuffer,
    output: &GpuBuffer,
) -> Result<(), D3D12KernelError> {
    let pso = pipelines.load_pipeline(gpu, Source::Reduce, "softmax_f32")?;
    let total = row_size * num_rows;
    let constants = [row_size, num_rows];
    let srvs = [BufferBinding::structured_f32(input, total)];
    let uav = BufferBinding::structured_f32(output, total);
    let groups = [num_rows, 1, 1];
    gpu.dispatch(&pso, &constants, &srvs, &uav, groups)
}
