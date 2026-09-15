#[derive(thiserror::Error, Debug)]
pub enum D3D12KernelError {
    #[error("{0}")]
    Message(String),
    #[error("Shader compilation error: {0}")]
    ShaderCompilation(String),
    #[error("Device creation failed: {0}")]
    DeviceCreation(String),
    #[error("Buffer operation failed: {0}")]
    BufferError(String),
    #[error("Pipeline creation failed: {0}")]
    PipelineCreation(String),
    #[error("Dispatch failed: {0}")]
    Dispatch(String),
    #[error("Data transfer failed: {0}")]
    DataTransfer(String),
    #[error("Could not lock resource: {0}")]
    LockError(String),
    #[error("Unsupported dtype {0} for operation {1}")]
    UnsupportedDType(&'static str, &'static str),
    #[error("{inner}\n{backtrace}")]
    WithBacktrace {
        inner: Box<Self>,
        backtrace: Box<std::backtrace::Backtrace>,
    },
}

impl D3D12KernelError {
    pub fn bt(self) -> Self {
        let backtrace = std::backtrace::Backtrace::capture();
        match backtrace.status() {
            std::backtrace::BacktraceStatus::Disabled
            | std::backtrace::BacktraceStatus::Unsupported => self,
            _ => Self::WithBacktrace {
                inner: Box::new(self),
                backtrace: Box::new(backtrace),
            },
        }
    }
}

impl<T> From<std::sync::PoisonError<T>> for D3D12KernelError {
    fn from(e: std::sync::PoisonError<T>) -> Self {
        Self::LockError(e.to_string())
    }
}
