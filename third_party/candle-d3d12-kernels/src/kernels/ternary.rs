use crate::d3d12::{BufferBinding, Gpu, GpuBuffer, Pipelines};
use crate::err::D3D12KernelError;
use crate::source::Source;
use crate::utils::linear_groups;

/// Where conditional: output[i] = cond[i] ? true_val[i] : false_val[i]
/// Condition is stored as packed u8 in u32 words.
pub fn call_where_cond(
    gpu: &Gpu,
    pipelines: &Pipelines,
    count: u32,
    cond: &GpuBuffer,
    cond_u32_count: u32,
    true_val: &GpuBuffer,
    false_val: &GpuBuffer,
    output: &GpuBuffer,
) -> Result<(), D3D12KernelError> {
    let pso = pipelines.load_pipeline(gpu, Source::Ternary, "where_f32")?;
    let constants = [count];
    let srvs = [
        BufferBinding::structured_u32(cond, cond_u32_count),
        BufferBinding::structured_f32(true_val, count),
        BufferBinding::structured_f32(false_val, count),
    ];
    let uav = BufferBinding::structured_f32(output, count);
    gpu.dispatch(&pso, &constants, &srvs, &uav, linear_groups(count))
}
