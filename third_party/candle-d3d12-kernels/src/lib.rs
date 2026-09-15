pub mod err;
pub mod source;
pub mod utils;

#[cfg(target_os = "windows")]
pub mod d3d12;
#[cfg(target_os = "windows")]
pub mod kernels;

pub use err::D3D12KernelError;
pub use source::Source;

#[cfg(target_os = "windows")]
pub use d3d12::{BufferBinding, Gpu, GpuBuffer, Pipelines};
#[cfg(target_os = "windows")]
pub use windows::Win32::Graphics::Direct3D12::ID3D12PipelineState;
#[cfg(target_os = "windows")]
pub use kernels::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    U8,
    U32,
    F32,
    F64,
}

impl DType {
    pub fn size_in_bytes(&self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U32 => 4,
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}
