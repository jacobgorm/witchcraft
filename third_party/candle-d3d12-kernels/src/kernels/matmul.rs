use crate::d3d12::{BufferBinding, Gpu, GpuBuffer, Pipelines};
use crate::err::D3D12KernelError;
use crate::source::Source;
use crate::utils::matmul_groups;

/// Naive batched matrix multiplication: C = A * B
/// A is (batch, M, K), B is (batch, K, N), C is (batch, M, N).
/// Supports arbitrary strides for both inputs.
#[allow(clippy::too_many_arguments)]
pub fn call_matmul(
    gpu: &Gpu,
    pipelines: &Pipelines,
    batch_size: u32,
    m: u32,
    n: u32,
    k: u32,
    lhs_stride_b: u32,
    lhs_stride_m: u32,
    lhs_stride_k: u32,
    rhs_stride_b: u32,
    rhs_stride_k: u32,
    rhs_stride_n: u32,
    a: &GpuBuffer,
    a_count: u32,
    b: &GpuBuffer,
    b_count: u32,
    c: &GpuBuffer,
) -> Result<(), D3D12KernelError> {
    let pso = pipelines.load_pipeline(gpu, Source::Matmul, "matmul_f32")?;
    let constants = [
        batch_size,
        m,
        n,
        k,
        lhs_stride_b,
        lhs_stride_m,
        lhs_stride_k,
        rhs_stride_b,
        rhs_stride_k,
        rhs_stride_n,
    ];
    let srvs = [
        BufferBinding::structured_f32(a, a_count),
        BufferBinding::structured_f32(b, b_count),
    ];
    let c_count = batch_size * m * n;
    let uav = BufferBinding::structured_f32(c, c_count);
    gpu.dispatch(
        &pso,
        &constants,
        &srvs,
        &uav,
        matmul_groups(n, m, batch_size),
    )
}
