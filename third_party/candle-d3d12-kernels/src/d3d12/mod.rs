use crate::err::D3D12KernelError;
use crate::source::Source;
use std::collections::HashMap;
use std::sync::RwLock;

use windows::core::Interface;
use windows::Win32::Foundation::*;
use windows::Win32::Graphics::Direct3D::Fxc::*;
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::Threading::*;

const MAX_ROOT_CONSTANTS: u32 = 32;
const MAX_SRVS: u32 = 4;
const MAX_UAVS: u32 = 8;
const DESCRIPTOR_COUNT: u32 = MAX_SRVS + MAX_UAVS;

/// A GPU buffer in default (device-local) heap.
pub struct GpuBuffer {
    pub(crate) resource: ID3D12Resource,
    pub(crate) size_bytes: u64,
}

impl GpuBuffer {
    pub fn size(&self) -> u64 {
        self.size_bytes
    }

    pub fn gpu_virtual_address(&self) -> u64 {
        unsafe { self.resource.GetGPUVirtualAddress() }
    }
}

impl std::fmt::Debug for GpuBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBuffer")
            .field("size_bytes", &self.size_bytes)
            .finish()
    }
}

// ID3D12Resource COM objects are free-threaded and thread-safe.
unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}

/// Describes how to bind a buffer as an SRV or UAV.
pub struct BufferBinding<'a> {
    pub buffer: &'a GpuBuffer,
    /// Offset in elements from the start of the buffer.
    pub first_element: u32,
    pub num_elements: u32,
    /// Byte stride per element. Use 0 for raw (ByteAddressBuffer).
    /// Use 4 for StructuredBuffer<float> or StructuredBuffer<uint>.
    pub stride: u32,
}

impl<'a> BufferBinding<'a> {
    pub fn structured_f32(buffer: &'a GpuBuffer, count: u32) -> Self {
        Self {
            buffer,
            first_element: 0,
            num_elements: count,
            stride: 4,
        }
    }

    pub fn structured_f32_offset(buffer: &'a GpuBuffer, offset: u32, count: u32) -> Self {
        Self {
            buffer,
            first_element: offset,
            num_elements: count,
            stride: 4,
        }
    }

    pub fn structured_u32(buffer: &'a GpuBuffer, count: u32) -> Self {
        Self {
            buffer,
            first_element: 0,
            num_elements: count,
            stride: 4,
        }
    }

    /// Bind as StructuredBuffer<half> (2 bytes per element).
    pub fn structured_f16(buffer: &'a GpuBuffer, count: u32) -> Self {
        Self {
            buffer,
            first_element: 0,
            num_elements: count,
            stride: 2,
        }
    }

    /// Bind with a custom stride (for arbitrary struct sizes).
    pub fn structured(buffer: &'a GpuBuffer, count: u32, stride: u32) -> Self {
        Self {
            buffer,
            first_element: 0,
            num_elements: count,
            stride,
        }
    }

    pub fn raw(buffer: &'a GpuBuffer) -> Self {
        Self {
            buffer,
            first_element: 0,
            num_elements: (buffer.size_bytes / 4) as u32,
            stride: 0,
        }
    }
}

/// Maximum number of dispatches in a single batch.
const MAX_BATCH_DISPATCHES: u32 = 512;

/// Core D3D12 GPU context for compute operations.
pub struct Gpu {
    pub device: ID3D12Device,
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    fence_value: std::cell::Cell<u64>,
    fence_event: HANDLE,
    // Shader-visible descriptor heap for SRV/UAV bindings
    srv_uav_heap: ID3D12DescriptorHeap,
    srv_uav_increment: u32,
    // The shared root signature used by all compute pipelines
    root_signature: ID3D12RootSignature,
    // Batch dispatch state: descriptor offset within the heap
    batch_offset: std::cell::Cell<u32>,
    batch_active: std::cell::Cell<bool>,
}

// The Gpu struct contains raw HANDLE which isn't Send/Sync by default.
// D3D12 COM objects are thread-safe (free-threaded apartment).
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

impl std::fmt::Debug for Gpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gpu").finish_non_exhaustive()
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        if !self.fence_event.is_invalid() {
            unsafe {
                let _ = CloseHandle(self.fence_event);
            }
        }
    }
}

impl Gpu {
    /// Enable the D3D12 debug layer. Must be called before `new()`.
    pub fn enable_debug_layer() {
        unsafe {
            let mut debug: Option<ID3D12Debug> = None;
            let _ = D3D12GetDebugInterface(&mut debug);
            if let Some(debug) = debug {
                debug.EnableDebugLayer();
                tracing::info!("D3D12 debug layer enabled");
            }
        }
    }

    /// Create a new GPU context, selecting the adapter at the given ordinal.
    pub fn new(ordinal: usize) -> Result<Self, D3D12KernelError> {
        unsafe {
            // Create DXGI factory
            let factory: IDXGIFactory1 = CreateDXGIFactory1()
                .map_err(|e| D3D12KernelError::DeviceCreation(e.to_string()))?;

            // Enumerate adapters
            let adapter: IDXGIAdapter1 = factory
                .EnumAdapters1(ordinal as u32)
                .map_err(|e| {
                    D3D12KernelError::DeviceCreation(format!(
                        "No adapter at ordinal {ordinal}: {e}"
                    ))
                })?;

            let desc = adapter
                .GetDesc1()
                .map_err(|e| D3D12KernelError::DeviceCreation(e.to_string()))?;

            let name = String::from_utf16_lossy(
                &desc.Description[..desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len())],
            );
            tracing::info!("D3D12: Using adapter {ordinal}: {name}");

            // Create device
            let mut device: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut device)
                .map_err(|e| D3D12KernelError::DeviceCreation(e.to_string()))?;
            let device = device.ok_or_else(|| {
                D3D12KernelError::DeviceCreation("D3D12CreateDevice returned None".into())
            })?;

            // Create compute command queue
            let queue_desc = D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_COMPUTE,
                ..Default::default()
            };
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&queue_desc)
                .map_err(|e| D3D12KernelError::DeviceCreation(e.to_string()))?;

            // Create command allocator
            let allocator: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE)
                .map_err(|e| D3D12KernelError::DeviceCreation(e.to_string()))?;

            // Create command list (closed initially)
            let list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &allocator, None)
                .map_err(|e| D3D12KernelError::DeviceCreation(e.to_string()))?;
            list.Close()
                .map_err(|e| D3D12KernelError::DeviceCreation(e.to_string()))?;

            // Create fence
            let fence: ID3D12Fence = device
                .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                .map_err(|e| D3D12KernelError::DeviceCreation(e.to_string()))?;

            let fence_event = CreateEventW(None, FALSE, FALSE, None)
                .map_err(|e| D3D12KernelError::DeviceCreation(e.to_string()))?;

            // Create shader-visible descriptor heap (large enough for batched dispatches)
            let heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: DESCRIPTOR_COUNT * MAX_BATCH_DISPATCHES,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            };
            let srv_uav_heap: ID3D12DescriptorHeap = device
                .CreateDescriptorHeap(&heap_desc)
                .map_err(|e| D3D12KernelError::DeviceCreation(e.to_string()))?;

            let srv_uav_increment =
                device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

            // Create universal root signature
            let root_signature = create_root_signature(&device)?;

            // Pre-fill all SRV descriptor slots with null descriptors.
            // This avoids re-creating them per dispatch (~4 API calls × 500 dispatches/token).
            {
                let cpu_start = srv_uav_heap.GetCPUDescriptorHandleForHeapStart();
                let null_srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_FLOAT,
                    ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_SRV {
                            FirstElement: 0,
                            NumElements: 0,
                            StructureByteStride: 0,
                            Flags: D3D12_BUFFER_SRV_FLAG_NONE,
                        },
                    },
                };
                let null_uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_FLOAT,
                    ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                    Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_UAV {
                            FirstElement: 0,
                            NumElements: 0,
                            StructureByteStride: 0,
                            CounterOffsetInBytes: 0,
                            Flags: D3D12_BUFFER_UAV_FLAG_NONE,
                        },
                    },
                };
                for dispatch_idx in 0..MAX_BATCH_DISPATCHES {
                    let desc_base = dispatch_idx * DESCRIPTOR_COUNT;
                    for srv_idx in 0..MAX_SRVS {
                        let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                            ptr: cpu_start.ptr
                                + ((desc_base + srv_idx) as usize * srv_uav_increment as usize),
                        };
                        device.CreateShaderResourceView(None, Some(&null_srv_desc), handle);
                    }
                    for uav_idx in 0..MAX_UAVS {
                        let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                            ptr: cpu_start.ptr
                                + ((desc_base + MAX_SRVS + uav_idx) as usize * srv_uav_increment as usize),
                        };
                        device.CreateUnorderedAccessView(None, None, Some(&null_uav_desc), handle);
                    }
                }
            }

            Ok(Self {
                device,
                queue,
                allocator,
                list,
                fence,
                fence_value: std::cell::Cell::new(0),
                fence_event,
                srv_uav_heap,
                srv_uav_increment,
                root_signature,
                batch_offset: std::cell::Cell::new(0),
                batch_active: std::cell::Cell::new(false),
            })
        }
    }

    /// Create a GPU buffer in the default (device-local) heap.
    pub fn create_buffer(&self, size_bytes: u64) -> Result<GpuBuffer, D3D12KernelError> {
        let size_bytes = std::cmp::max(size_bytes, 256); // D3D12 minimum buffer size
        let heap_props = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size_bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let resource: ID3D12Resource = unsafe {
            let mut resource: Option<ID3D12Resource> = None;
            self.device.CreateCommittedResource(
                &heap_props,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut resource,
            )
            .map_err(|e| D3D12KernelError::BufferError(e.to_string()))?;
            resource.ok_or_else(|| {
                D3D12KernelError::BufferError("CreateCommittedResource returned None".into())
            })?
        };

        Ok(GpuBuffer {
            resource,
            size_bytes,
        })
    }

    /// Create a readback (CPU-visible) buffer for GPU→CPU transfers.
    /// Pre-allocate once and reuse to avoid per-frame allocation overhead.
    pub fn create_readback_buffer(&self, size_bytes: u64) -> Result<GpuBuffer, D3D12KernelError> {
        let size_bytes = std::cmp::max(size_bytes, 256);
        let heap_props = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size_bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let resource: ID3D12Resource = unsafe {
            let mut resource: Option<ID3D12Resource> = None;
            self.device.CreateCommittedResource(
                &heap_props,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut resource,
            )
            .map_err(|e| D3D12KernelError::BufferError(e.to_string()))?;
            resource.ok_or_else(|| {
                D3D12KernelError::BufferError("CreateCommittedResource returned None".into())
            })?
        };
        Ok(GpuBuffer {
            resource,
            size_bytes,
        })
    }

    /// Record a buffer copy into the active batch command list.
    /// Must be called between begin_batch and end_batch.
    pub fn record_copy(&self, src: &GpuBuffer, dst: &GpuBuffer, size_bytes: u64) {
        if !self.batch_active.get() { return; }
        let size = std::cmp::min(size_bytes, std::cmp::min(src.size_bytes, dst.size_bytes));
        unsafe {
            self.list.CopyBufferRegion(&dst.resource, 0, &src.resource, 0, size);
        }
    }

    /// Map a readback buffer, copy data to a Vec, and unmap.
    /// Call after end_batch to read back data that was copied during the batch.
    pub fn map_readback_buffer(&self, buf: &GpuBuffer, size_bytes: u64) -> Result<Vec<u8>, D3D12KernelError> {
        let size = std::cmp::min(size_bytes, buf.size_bytes) as usize;
        let mut result = vec![0u8; size];
        unsafe {
            let mut mapped = std::ptr::null_mut();
            buf.resource
                .Map(0, None, Some(&mut mapped))
                .map_err(|e| D3D12KernelError::DataTransfer(e.to_string()))?;
            std::ptr::copy_nonoverlapping(mapped as *const u8, result.as_mut_ptr(), size);
            buf.resource.Unmap(0, None);
        }
        Ok(result)
    }

    /// Create an upload (CPU-writable) staging buffer.
    /// Pre-allocate once and reuse to avoid per-frame allocation overhead.
    pub fn create_upload_buffer(&self, size_bytes: u64) -> Result<GpuBuffer, D3D12KernelError> {
        let size_bytes = std::cmp::max(size_bytes, 256);
        let heap_props = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size_bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let resource: ID3D12Resource = unsafe {
            let mut resource: Option<ID3D12Resource> = None;
            self.device.CreateCommittedResource(
                &heap_props,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                None,
                &mut resource,
            )
            .map_err(|e| D3D12KernelError::BufferError(e.to_string()))?;
            resource.ok_or_else(|| {
                D3D12KernelError::BufferError("CreateCommittedResource returned None".into())
            })?
        };
        Ok(GpuBuffer {
            resource,
            size_bytes,
        })
    }

    /// Write CPU data into a pre-allocated upload buffer (Map, copy, Unmap).
    pub fn write_upload_buffer(&self, buf: &GpuBuffer, data: &[u8]) -> Result<(), D3D12KernelError> {
        let size = data.len();
        assert!((size as u64) <= buf.size_bytes);
        unsafe {
            let mut mapped = std::ptr::null_mut();
            buf.resource
                .Map(0, None, Some(&mut mapped))
                .map_err(|e| D3D12KernelError::DataTransfer(e.to_string()))?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), mapped as *mut u8, size);
            buf.resource.Unmap(0, None);
        }
        Ok(())
    }

    /// Upload CPU data to a GPU buffer. Creates a temporary upload buffer internally.
    pub fn upload_to_buffer(
        &self,
        data: &[u8],
        dst: &GpuBuffer,
    ) -> Result<(), D3D12KernelError> {
        let size = data.len() as u64;
        assert!(size <= dst.size_bytes);

        // Create upload buffer
        let upload_heap_props = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            ..Default::default()
        };
        let upload_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: std::cmp::max(size, 256),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let upload_buf: ID3D12Resource = unsafe {
            let mut resource: Option<ID3D12Resource> = None;
            self.device.CreateCommittedResource(
                &upload_heap_props,
                D3D12_HEAP_FLAG_NONE,
                &upload_desc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                None,
                &mut resource,
            )
            .map_err(|e| D3D12KernelError::DataTransfer(e.to_string()))?;
            resource.ok_or_else(|| {
                D3D12KernelError::DataTransfer("CreateCommittedResource returned None".into())
            })?
        };

        // Map, copy, unmap
        unsafe {
            let mut mapped = std::ptr::null_mut();
            upload_buf
                .Map(0, None, Some(&mut mapped))
                .map_err(|e| D3D12KernelError::DataTransfer(e.to_string()))?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), mapped as *mut u8, data.len());
            upload_buf.Unmap(0, None);
        }

        // Record copy command
        self.begin_command_list()?;
        unsafe {
            self.list
                .CopyBufferRegion(&dst.resource, 0, &upload_buf, 0, size);
        }
        self.execute_and_wait()?;

        Ok(())
    }

    /// Download GPU buffer contents to CPU memory.
    pub fn download_buffer(
        &self,
        src: &GpuBuffer,
        size_bytes: u64,
    ) -> Result<Vec<u8>, D3D12KernelError> {
        let size = std::cmp::min(size_bytes, src.size_bytes);

        // Create readback buffer
        let readback_heap_props = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let readback_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: std::cmp::max(size, 256),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let readback_buf: ID3D12Resource = unsafe {
            let mut resource: Option<ID3D12Resource> = None;
            self.device.CreateCommittedResource(
                &readback_heap_props,
                D3D12_HEAP_FLAG_NONE,
                &readback_desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut resource,
            )
            .map_err(|e| D3D12KernelError::DataTransfer(e.to_string()))?;
            resource.ok_or_else(|| {
                D3D12KernelError::DataTransfer("CreateCommittedResource returned None".into())
            })?
        };

        // Record copy command
        self.begin_command_list()?;
        unsafe {
            self.list
                .CopyBufferRegion(&readback_buf, 0, &src.resource, 0, size);
        }
        self.execute_and_wait()?;

        // Map, copy, unmap
        let mut result = vec![0u8; size as usize];
        unsafe {
            let mut mapped = std::ptr::null_mut();
            readback_buf
                .Map(0, None, Some(&mut mapped))
                .map_err(|e| D3D12KernelError::DataTransfer(e.to_string()))?;
            std::ptr::copy_nonoverlapping(mapped as *const u8, result.as_mut_ptr(), size as usize);
            readback_buf.Unmap(0, None);
        }

        Ok(result)
    }

    /// Compile an HLSL compute shader to DXBC bytecode.
    pub fn compile_shader(
        &self,
        hlsl_source: &str,
        entry_point: &str,
    ) -> Result<Vec<u8>, D3D12KernelError> {
        unsafe {
            let source_bytes = hlsl_source.as_bytes();
            let entry = std::ffi::CString::new(entry_point)
                .map_err(|e| D3D12KernelError::ShaderCompilation(e.to_string()))?;
            let target = std::ffi::CString::new("cs_5_1")
                .map_err(|e| D3D12KernelError::ShaderCompilation(e.to_string()))?;

            let mut shader_blob: Option<ID3DBlob> = None;
            let mut error_blob: Option<ID3DBlob> = None;

            let hr = D3DCompile(
                source_bytes.as_ptr() as *const _,
                source_bytes.len(),
                None, // source name
                None, // defines
                None, // include handler
                windows::core::PCSTR(entry.as_ptr() as *const u8),
                windows::core::PCSTR(target.as_ptr() as *const u8),
                0,    // flags1
                0,    // flags2
                &mut shader_blob,
                Some(&mut error_blob),
            );

            if hr.is_err() {
                let msg = if let Some(err_blob) = &error_blob {
                    let ptr = err_blob.GetBufferPointer() as *const u8;
                    let len = err_blob.GetBufferSize();
                    let slice = std::slice::from_raw_parts(ptr, len);
                    String::from_utf8_lossy(slice).to_string()
                } else {
                    format!("D3DCompile failed: {hr:?}")
                };
                return Err(D3D12KernelError::ShaderCompilation(msg));
            }

            let blob = shader_blob.ok_or_else(|| {
                D3D12KernelError::ShaderCompilation("D3DCompile returned no blob".into())
            })?;
            let ptr = blob.GetBufferPointer() as *const u8;
            let len = blob.GetBufferSize();
            Ok(std::slice::from_raw_parts(ptr, len).to_vec())
        }
    }

    /// Create a compute pipeline state object from compiled shader bytecode.
    pub fn create_compute_pso(
        &self,
        bytecode: &[u8],
    ) -> Result<ID3D12PipelineState, D3D12KernelError> {
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: unsafe {
                std::mem::transmute_copy(&self.root_signature)
            },
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytecode.as_ptr() as *const _,
                BytecodeLength: bytecode.len(),
            },
            ..Default::default()
        };
        let pso: ID3D12PipelineState = unsafe {
            self.device.CreateComputePipelineState(&desc)
        }
        .map_err(|e| D3D12KernelError::PipelineCreation(e.to_string()))?;
        Ok(pso)
    }

    /// Run a compute dispatch synchronously.
    ///
    /// - `pso`: The pipeline state object for the kernel.
    /// - `root_constants`: Up to 16 u32 values for cbuffer at register(b0).
    /// - `srvs`: Input buffer bindings for t0, t1, ... (up to 4).
    /// - `uav`: Output buffer binding for u0.
    /// - `groups`: Thread group counts [x, y, z].
    pub fn dispatch(
        &self,
        pso: &ID3D12PipelineState,
        root_constants: &[u32],
        srvs: &[BufferBinding],
        uav: &BufferBinding,
        groups: [u32; 3],
    ) -> Result<(), D3D12KernelError> {
        assert!(root_constants.len() <= MAX_ROOT_CONSTANTS as usize);
        assert!(srvs.len() <= MAX_SRVS as usize);

        unsafe {
            self.begin_command_list()?;

            // Set pipeline state and root signature
            self.list.SetPipelineState(pso);
            self.list.SetComputeRootSignature(&self.root_signature);

            // Set descriptor heap
            self.list
                .SetDescriptorHeaps(&[Some(self.srv_uav_heap.clone())]);

            // Create SRV descriptors in the heap
            let cpu_start = self
                .srv_uav_heap
                .GetCPUDescriptorHandleForHeapStart();
            let gpu_start = self
                .srv_uav_heap
                .GetGPUDescriptorHandleForHeapStart();

            // Fill SRV slots (0..MAX_SRVS)
            for i in 0..MAX_SRVS {
                let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: cpu_start.ptr + (i * self.srv_uav_increment) as usize,
                };
                if (i as usize) < srvs.len() {
                    let binding = &srvs[i as usize];
                    self.create_srv(&binding.buffer.resource, binding, handle);
                } else {
                    self.create_null_srv(handle);
                }
            }

            // Fill UAV slot (MAX_SRVS)
            {
                let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: cpu_start.ptr + (MAX_SRVS * self.srv_uav_increment) as usize,
                };
                self.create_uav(&uav.buffer.resource, uav, handle);
            }

            // Set root constants (parameter 0)
            for (i, &val) in root_constants.iter().enumerate() {
                self.list
                    .SetComputeRoot32BitConstant(0, val, i as u32);
            }

            // Set SRV descriptor table (parameter 1)
            self.list.SetComputeRootDescriptorTable(1, gpu_start);

            // Set UAV descriptor table (parameter 2)
            let uav_gpu_handle = D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: gpu_start.ptr + (MAX_SRVS as u64) * (self.srv_uav_increment as u64),
            };
            self.list
                .SetComputeRootDescriptorTable(2, uav_gpu_handle);

            // Dispatch
            self.list.Dispatch(groups[0], groups[1], groups[2]);
        }

        self.execute_and_wait()?;
        Ok(())
    }

    /// Dispatch a compute shader using only UAVs (no SRVs).
    ///
    /// Used for Triton-generated HLSL where all buffer args are RWStructuredBuffer.
    /// UAVs are bound to u0, u1, u2, ... in the order provided.
    pub fn dispatch_uav_only(
        &self,
        pso: &ID3D12PipelineState,
        root_constants: &[u32],
        uavs: &[BufferBinding],
        groups: [u32; 3],
    ) -> Result<(), D3D12KernelError> {
        assert!(root_constants.len() <= MAX_ROOT_CONSTANTS as usize);
        assert!(uavs.len() <= MAX_UAVS as usize);

        unsafe {
            self.begin_command_list()?;

            self.list.SetPipelineState(pso);
            self.list.SetComputeRootSignature(&self.root_signature);

            self.list
                .SetDescriptorHeaps(&[Some(self.srv_uav_heap.clone())]);

            let cpu_start = self
                .srv_uav_heap
                .GetCPUDescriptorHandleForHeapStart();
            let gpu_start = self
                .srv_uav_heap
                .GetGPUDescriptorHandleForHeapStart();

            // Fill SRV slots with null (not used)
            for i in 0..MAX_SRVS {
                let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: cpu_start.ptr + (i * self.srv_uav_increment) as usize,
                };
                self.create_null_srv(handle);
            }

            // Fill UAV slots (MAX_SRVS + 0, MAX_SRVS + 1, ...)
            for (i, uav) in uavs.iter().enumerate() {
                let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: cpu_start.ptr
                        + ((MAX_SRVS + i as u32) * self.srv_uav_increment) as usize,
                };
                self.create_uav(&uav.buffer.resource, uav, handle);
            }
            // Fill remaining UAV slots with null
            for i in uavs.len()..MAX_UAVS as usize {
                let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: cpu_start.ptr
                        + ((MAX_SRVS + i as u32) * self.srv_uav_increment) as usize,
                };
                self.create_null_uav(handle);
            }

            // Root constants (parameter 0)
            for (i, &val) in root_constants.iter().enumerate() {
                self.list
                    .SetComputeRoot32BitConstant(0, val, i as u32);
            }

            // SRV table (parameter 1)
            self.list.SetComputeRootDescriptorTable(1, gpu_start);

            // UAV table (parameter 2)
            let uav_gpu_handle = D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: gpu_start.ptr + (MAX_SRVS as u64) * (self.srv_uav_increment as u64),
            };
            self.list
                .SetComputeRootDescriptorTable(2, uav_gpu_handle);

            self.list.Dispatch(groups[0], groups[1], groups[2]);
        }

        self.execute_and_wait()?;
        Ok(())
    }

    /// Begin a batch of dispatches. All subsequent `record_dispatch` calls
    /// will be recorded into a single command list without GPU synchronization.
    /// Call `end_batch()` to execute all recorded dispatches at once.
    pub fn begin_batch(&self) -> Result<(), D3D12KernelError> {
        self.batch_offset.set(0);
        self.batch_active.set(true);
        self.begin_command_list()?;
        unsafe {
            self.list
                .SetDescriptorHeaps(&[Some(self.srv_uav_heap.clone())]);
            self.list.SetComputeRootSignature(&self.root_signature);
        }
        Ok(())
    }

    /// Record a UAV barrier (flush all UAV caches).
    /// Only valid during an active batch. Use between dispatches that have
    /// read-after-write dependencies on the same buffers.
    pub fn record_uav_barrier(&self) {
        if !self.batch_active.get() { return; }
        unsafe {
            let barrier = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                        pResource: std::mem::ManuallyDrop::new(None),
                    }),
                },
            };
            self.list.ResourceBarrier(&[barrier]);
        }
    }

    /// Record a single dispatch. If a batch is active (`begin_batch` was called),
    /// the dispatch is recorded without GPU sync. Otherwise, falls back to
    /// single-dispatch mode (equivalent to `dispatch_uav_only`).
    pub fn record_dispatch(
        &self,
        pso: &ID3D12PipelineState,
        root_constants: &[u32],
        uavs: &[BufferBinding],
        groups: [u32; 3],
    ) -> Result<(), D3D12KernelError> {
        // Fall back to single-dispatch mode if no batch is active
        if !self.batch_active.get() {
            return self.dispatch_uav_only(pso, root_constants, uavs, groups);
        }

        assert!(root_constants.len() <= MAX_ROOT_CONSTANTS as usize);
        assert!(uavs.len() <= MAX_UAVS as usize);

        let offset = self.batch_offset.get();
        assert!(
            offset < MAX_BATCH_DISPATCHES,
            "batch dispatch limit exceeded"
        );

        unsafe {
            self.list.SetPipelineState(pso);
            // Root signature set once in begin_batch — skip here.

            let desc_base = offset * DESCRIPTOR_COUNT;

            let cpu_start = self
                .srv_uav_heap
                .GetCPUDescriptorHandleForHeapStart();
            let gpu_start = self
                .srv_uav_heap
                .GetGPUDescriptorHandleForHeapStart();

            let cpu_base = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: cpu_start.ptr + (desc_base * self.srv_uav_increment) as usize,
            };
            let gpu_base = D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: gpu_start.ptr + (desc_base as u64) * (self.srv_uav_increment as u64),
            };

            // SRV slots pre-filled with null at init time — skip here.

            // Fill UAV slots
            for (i, uav) in uavs.iter().enumerate() {
                let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: cpu_base.ptr
                        + ((MAX_SRVS + i as u32) * self.srv_uav_increment) as usize,
                };
                self.create_uav(&uav.buffer.resource, uav, handle);
            }
            // Unused UAV slots pre-filled with null at init time — skip here.

            // Root constants
            for (i, &val) in root_constants.iter().enumerate() {
                self.list
                    .SetComputeRoot32BitConstant(0, val, i as u32);
            }

            // SRV table
            self.list.SetComputeRootDescriptorTable(1, gpu_base);

            // UAV table
            let uav_gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: gpu_base.ptr + (MAX_SRVS as u64) * (self.srv_uav_increment as u64),
            };
            self.list.SetComputeRootDescriptorTable(2, uav_gpu);

            self.list.Dispatch(groups[0], groups[1], groups[2]);
        }

        self.batch_offset.set(offset + 1);
        Ok(())
    }

    /// Execute all recorded dispatches and wait for completion.
    pub fn end_batch(&self) -> Result<(), D3D12KernelError> {
        self.batch_active.set(false);
        self.execute_and_wait()?;
        self.batch_offset.set(0);
        Ok(())
    }

    /// Compile an HLSL compute shader targeting SM 6.2+ with DXC.
    ///
    /// Supports half precision (float16_t / half) via `-enable-16bit-types`.
    /// Shells out to `dxc.exe` (from Windows SDK or standalone DXC).
    /// Falls back to FXC (SM 5.1) if DXC is not available.
    pub fn compile_shader_sm6(
        &self,
        hlsl_source: &str,
        entry_point: &str,
    ) -> Result<Vec<u8>, D3D12KernelError> {
        match Self::compile_shader_dxc(hlsl_source, entry_point) {
            Ok(bytecode) => Ok(bytecode),
            Err(e) => {
                eprintln!("warning: DXC compilation failed ({e}), falling back to FXC SM 5.1");
                self.compile_shader(hlsl_source, entry_point)
            }
        }
    }

    /// Compile HLSL to DXIL bytecode using DXC command-line compiler.
    ///
    /// Searches for `dxc.exe` on PATH and in Windows SDK directories.
    fn compile_shader_dxc(
        hlsl_source: &str,
        entry_point: &str,
    ) -> Result<Vec<u8>, D3D12KernelError> {
        use std::process::Command;

        let dxc_path = Self::find_dxc()
            .ok_or_else(|| D3D12KernelError::ShaderCompilation(
                "dxc.exe not found on PATH or in Windows SDK".into()
            ))?;

        let temp_dir = std::env::temp_dir();
        let input_path = temp_dir.join("triton_kernel.hlsl");
        let output_path = temp_dir.join("triton_kernel.cso");

        std::fs::write(&input_path, hlsl_source)
            .map_err(|e| D3D12KernelError::ShaderCompilation(format!("write temp HLSL: {e}")))?;

        let output = Command::new(&dxc_path)
            .args([
                "-T", "cs_6_2",
                "-enable-16bit-types",
                "-E", entry_point,
                "-Fo",
            ])
            .arg(&output_path)
            .arg(&input_path)
            .output()
            .map_err(|e| D3D12KernelError::ShaderCompilation(format!("dxc exec failed: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            return Err(D3D12KernelError::ShaderCompilation(
                format!("dxc failed:\n{stderr}\n{stdout}")
            ));
        }

        std::fs::read(&output_path)
            .map_err(|e| D3D12KernelError::ShaderCompilation(format!("read compiled CSO: {e}")))
    }

    /// Find dxc.exe: check Windows SDK directories, then PATH.
    fn find_dxc() -> Option<std::path::PathBuf> {
        // Search Windows SDK directories (newest version first)
        for base in [
            r"C:\Program Files (x86)\Windows Kits\10\bin",
            r"C:\Program Files\Windows Kits\10\bin",
        ] {
            let sdk_base = std::path::Path::new(base);
            if let Ok(entries) = std::fs::read_dir(sdk_base) {
                let mut versions: Vec<_> = entries
                    .filter_map(|e| e.ok())
                    .filter(|e| e.file_name().to_string_lossy().starts_with("10."))
                    .collect();
                versions.sort_by(|a, b| b.file_name().cmp(&a.file_name()));

                for entry in versions {
                    let dxc = entry.path().join("x64").join("dxc.exe");
                    if dxc.exists() {
                        return Some(dxc);
                    }
                }
            }
        }

        // Try PATH as fallback
        if let Ok(output) = std::process::Command::new("dxc").arg("-help").output() {
            if output.status.success() {
                return Some("dxc".into());
            }
        }

        None
    }

    /// GPU-to-GPU buffer copy with byte offsets.
    pub fn copy_buffer_region(
        &self,
        src: &GpuBuffer,
        src_offset_bytes: u64,
        dst: &GpuBuffer,
        dst_offset_bytes: u64,
        size_bytes: u64,
    ) -> Result<(), D3D12KernelError> {
        self.begin_command_list()?;
        unsafe {
            self.list.CopyBufferRegion(
                &dst.resource,
                dst_offset_bytes,
                &src.resource,
                src_offset_bytes,
                size_bytes,
            );
        }
        self.execute_and_wait()?;
        Ok(())
    }

    fn begin_command_list(&self) -> Result<(), D3D12KernelError> {
        unsafe {
            self.allocator
                .Reset()
                .map_err(|e| D3D12KernelError::Dispatch(e.to_string()))?;
            self.list
                .Reset(&self.allocator, None)
                .map_err(|e| D3D12KernelError::Dispatch(e.to_string()))?;
        }
        Ok(())
    }

    fn execute_and_wait(&self) -> Result<(), D3D12KernelError> {
        unsafe {
            self.list
                .Close()
                .map_err(|e| D3D12KernelError::Dispatch(e.to_string()))?;

            let lists = [Some(self.list.cast::<ID3D12CommandList>().map_err(|e| {
                D3D12KernelError::Dispatch(e.to_string())
            })?)];
            self.queue.ExecuteCommandLists(&lists);

            let val = self.fence_value.get() + 1;
            self.fence_value.set(val);

            self.queue
                .Signal(&self.fence, val)
                .map_err(|e| D3D12KernelError::Dispatch(e.to_string()))?;

            // Spin-wait: avoids kernel-mode WaitForSingleObject overhead (~3-5ms).
            while self.fence.GetCompletedValue() < val {
                std::hint::spin_loop();
            }
        }
        Ok(())
    }

    unsafe fn create_srv(
        &self,
        resource: &ID3D12Resource,
        binding: &BufferBinding,
        handle: D3D12_CPU_DESCRIPTOR_HANDLE,
    ) {
        if binding.stride == 0 {
            // Raw buffer (ByteAddressBuffer)
            let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        FirstElement: binding.first_element as u64,
                        NumElements: binding.num_elements,
                        StructureByteStride: 0,
                        Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                    },
                },
            };
            self.device
                .CreateShaderResourceView(Some(resource), Some(&desc), handle);
        } else {
            // Structured buffer
            let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_UNKNOWN,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        FirstElement: binding.first_element as u64,
                        NumElements: binding.num_elements,
                        StructureByteStride: binding.stride,
                        Flags: D3D12_BUFFER_SRV_FLAG_NONE,
                    },
                },
            };
            self.device
                .CreateShaderResourceView(Some(resource), Some(&desc), handle);
        }
    }

    unsafe fn create_uav(
        &self,
        resource: &ID3D12Resource,
        binding: &BufferBinding,
        handle: D3D12_CPU_DESCRIPTOR_HANDLE,
    ) {
        if binding.stride == 0 {
            // Raw buffer (RWByteAddressBuffer)
            let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV {
                        FirstElement: binding.first_element as u64,
                        NumElements: binding.num_elements,
                        StructureByteStride: 0,
                        Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                        CounterOffsetInBytes: 0,
                    },
                },
            };
            self.device
                .CreateUnorderedAccessView(Some(resource), None, Some(&desc), handle);
        } else {
            // Structured buffer
            let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_UNKNOWN,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV {
                        FirstElement: binding.first_element as u64,
                        NumElements: binding.num_elements,
                        StructureByteStride: binding.stride,
                        Flags: D3D12_BUFFER_UAV_FLAG_NONE,
                        CounterOffsetInBytes: 0,
                    },
                },
            };
            self.device
                .CreateUnorderedAccessView(Some(resource), None, Some(&desc), handle);
        }
    }

    unsafe fn create_null_srv(&self, handle: D3D12_CPU_DESCRIPTOR_HANDLE) {
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_FLOAT,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: 0,
                    NumElements: 0,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_NONE,
                },
            },
        };
        self.device
            .CreateShaderResourceView(None, Some(&desc), handle);
    }

    unsafe fn create_null_uav(&self, handle: D3D12_CPU_DESCRIPTOR_HANDLE) {
        let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_FLOAT,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    FirstElement: 0,
                    NumElements: 0,
                    StructureByteStride: 0,
                    CounterOffsetInBytes: 0,
                    Flags: D3D12_BUFFER_UAV_FLAG_NONE,
                },
            },
        };
        self.device
            .CreateUnorderedAccessView(None, None, Some(&desc), handle);
    }
}

/// Pipeline cache: maps (Source, entry_point) -> compiled PSO.
pub struct Pipelines {
    cache: RwLock<HashMap<(Source, String), ID3D12PipelineState>>,
}

impl std::fmt::Debug for Pipelines {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipelines").finish_non_exhaustive()
    }
}

impl Pipelines {
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
        }
    }

    pub fn load_pipeline(
        &self,
        gpu: &Gpu,
        source: Source,
        entry_point: &str,
    ) -> Result<ID3D12PipelineState, D3D12KernelError> {
        let key = (source, entry_point.to_string());

        // Check cache
        {
            let cache = self.cache.read()?;
            if let Some(pso) = cache.get(&key) {
                return Ok(pso.clone());
            }
        }

        // Compile and create
        let hlsl = source.hlsl_source();
        let bytecode = gpu.compile_shader(hlsl, entry_point)?;
        let pso = gpu.create_compute_pso(&bytecode)?;

        // Insert into cache
        {
            let mut cache = self.cache.write()?;
            cache.insert(key, pso.clone());
        }

        Ok(pso)
    }

    /// Load a pipeline from raw HLSL source code (for Triton-generated kernels).
    ///
    /// Caches by (entry_point, source hash) to avoid recompilation.
    pub fn load_pipeline_from_hlsl(
        &self,
        gpu: &Gpu,
        hlsl_source: &str,
        entry_point: &str,
    ) -> Result<ID3D12PipelineState, D3D12KernelError> {
        // Use a sentinel Source for custom HLSL (reuse Matmul as key slot)
        // and include a hash of the source to differentiate
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        hlsl_source.hash(&mut hasher);
        let hash = hasher.finish();
        let cache_key = format!("triton_{entry_point}_{hash:x}");
        let key = (Source::Matmul, cache_key.clone());

        {
            let cache = self.cache.read()?;
            if let Some(pso) = cache.get(&key) {
                return Ok(pso.clone());
            }
        }

        let bytecode = gpu.compile_shader(hlsl_source, entry_point)?;
        let pso = gpu.create_compute_pso(&bytecode)?;

        {
            let mut cache = self.cache.write()?;
            cache.insert(key, pso.clone());
        }

        Ok(pso)
    }
}

/// Create the shared root signature used by all compute pipelines.
///
/// Layout:
/// - Parameter 0: Root constants (32 DWORDs at register b0)
/// - Parameter 1: Descriptor table with SRV range (t0-t3, 4 descriptors)
/// - Parameter 2: Descriptor table with UAV range (u0-u7, 8 descriptors)
fn create_root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature, D3D12KernelError> {
    unsafe {
        let srv_range = D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: MAX_SRVS,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
        };

        let uav_range = D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
            NumDescriptors: MAX_UAVS,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
        };

        // We need to keep these alive until D3D12SerializeRootSignature is called.
        let srv_range_ref = &srv_range as *const D3D12_DESCRIPTOR_RANGE;
        let uav_range_ref = &uav_range as *const D3D12_DESCRIPTOR_RANGE;

        let parameters = [
            // Parameter 0: Root constants
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Num32BitValues: MAX_ROOT_CONSTANTS,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            // Parameter 1: SRV descriptor table
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                        NumDescriptorRanges: 1,
                        pDescriptorRanges: srv_range_ref,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            // Parameter 2: UAV descriptor table
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                        NumDescriptorRanges: 1,
                        pDescriptorRanges: uav_range_ref,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
        ];

        let root_sig_desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: parameters.len() as u32,
            pParameters: parameters.as_ptr(),
            NumStaticSamplers: 0,
            pStaticSamplers: std::ptr::null(),
            Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
        };

        let mut signature_blob: Option<ID3DBlob> = None;
        let mut error_blob: Option<ID3DBlob> = None;

        D3D12SerializeRootSignature(
            &root_sig_desc,
            D3D_ROOT_SIGNATURE_VERSION_1,
            &mut signature_blob,
            Some(&mut error_blob),
        )
        .map_err(|e| {
            let msg = if let Some(err) = &error_blob {
                let ptr = err.GetBufferPointer() as *const u8;
                let len = err.GetBufferSize();
                let s = std::slice::from_raw_parts(ptr, len);
                String::from_utf8_lossy(s).to_string()
            } else {
                e.to_string()
            };
            D3D12KernelError::PipelineCreation(format!("Root signature serialization: {msg}"))
        })?;

        let blob = signature_blob.ok_or_else(|| {
            D3D12KernelError::PipelineCreation("Root signature blob is None".into())
        })?;

        let root_sig: ID3D12RootSignature = device
            .CreateRootSignature(
                0,
                std::slice::from_raw_parts(
                    blob.GetBufferPointer() as *const u8,
                    blob.GetBufferSize(),
                ),
            )
            .map_err(|e| {
                D3D12KernelError::PipelineCreation(format!("CreateRootSignature: {e}"))
            })?;

        Ok(root_sig)
    }
}
