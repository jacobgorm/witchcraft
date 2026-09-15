use crate::d3d12::{BufferBinding, Gpu, GpuBuffer, Pipelines};
use crate::err::D3D12KernelError;
use crate::source::Source;
use crate::utils::linear_groups;

/// 1D convolution.
#[allow(clippy::too_many_arguments)]
pub fn call_conv1d(
    gpu: &Gpu,
    pipelines: &Pipelines,
    batch_size: u32,
    c_in: u32,
    c_out: u32,
    l_in: u32,
    l_out: u32,
    k_size: u32,
    stride: u32,
    padding: u32,
    dilation: u32,
    input: &GpuBuffer,
    input_count: u32,
    kernel_buf: &GpuBuffer,
    kernel_count: u32,
    output: &GpuBuffer,
) -> Result<(), D3D12KernelError> {
    let pso = pipelines.load_pipeline(gpu, Source::Conv, "conv1d_f32")?;
    let constants = [
        batch_size, c_in, c_out, l_in, l_out, k_size, stride, padding, dilation,
    ];
    let srvs = [
        BufferBinding::structured_f32(input, input_count),
        BufferBinding::structured_f32(kernel_buf, kernel_count),
    ];
    let total_out = batch_size * c_out * l_out;
    let uav = BufferBinding::structured_f32(output, total_out);
    gpu.dispatch(&pso, &constants, &srvs, &uav, linear_groups(total_out))
}
