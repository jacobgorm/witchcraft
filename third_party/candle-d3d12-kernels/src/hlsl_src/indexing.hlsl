// Indexing operations: index_select, gather, scatter.

// --- Index Select ---
// output[i] = input[indices[i / right_size] * right_size + i % right_size]
// where the selection is along dimension `dim`.

cbuffer IndexSelectParams : register(b0) {
    uint is_count;          // Total output elements
    uint is_left_size;      // Product of dims before the selected dim
    uint is_dim_size;       // Size of the source dim
    uint is_right_size;     // Product of dims after the selected dim
};

StructuredBuffer<float> is_input : register(t0);
StructuredBuffer<uint> is_indices : register(t1);
RWStructuredBuffer<float> is_output : register(u0);

[numthreads(256, 1, 1)]
void index_select_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= is_count) return;
    uint right_i = tid.x % is_right_size;
    uint idx_i = (tid.x / is_right_size) % is_left_size;
    uint left_i = tid.x / (is_right_size * is_left_size);
    uint src_idx = is_indices[idx_i];
    uint src_offset = left_i * is_dim_size * is_right_size + src_idx * is_right_size + right_i;
    is_output[tid.x] = is_input[src_offset];
}

// --- Gather ---
cbuffer GatherParams : register(b0) {
    uint g_count;        // Total output elements
    uint g_left_size;    // Product of dims before gathered dim in output
    uint g_dim_size;     // Size of source dim
    uint g_right_size;   // Product of dims after gathered dim in output
    uint g_idx_dim_size; // Size of index dim
};

StructuredBuffer<float> g_input : register(t0);
StructuredBuffer<uint> g_indices : register(t1);
RWStructuredBuffer<float> g_output : register(u0);

[numthreads(256, 1, 1)]
void gather_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= g_count) return;
    uint right_i = tid.x % g_right_size;
    uint idx_i = (tid.x / g_right_size) % g_idx_dim_size;
    uint left_i = tid.x / (g_right_size * g_idx_dim_size);
    uint src_idx = g_indices[tid.x]; // index from the indices tensor
    uint src_offset = left_i * g_dim_size * g_right_size + src_idx * g_right_size + right_i;
    g_output[tid.x] = g_input[src_offset];
}

// --- Copy strided (used for copy_strided_src) ---
cbuffer CopyStridedParams : register(b0) {
    uint cs_count;
    uint cs_num_dims;
    uint cs_dst_offset;
};

StructuredBuffer<float> cs_input : register(t0);
ByteAddressBuffer cs_meta : register(t1);  // dims[] then strides[]
RWStructuredBuffer<float> cs_output : register(u0);

[numthreads(256, 1, 1)]
void copy_strided_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= cs_count) return;
    uint dims_offset = 0;
    uint strides_offset = cs_num_dims * 4;

    uint strided_i = 0;
    uint idx = tid.x;
    for (uint d = 0; d < cs_num_dims; d++) {
        uint dim_idx = cs_num_dims - 1 - d;
        uint dim_size = cs_meta.Load(dims_offset + dim_idx * 4);
        uint stride = cs_meta.Load(strides_offset + dim_idx * 4);
        strided_i += (idx % dim_size) * stride;
        idx /= dim_size;
    }
    cs_output[cs_dst_offset + tid.x] = cs_input[strided_i];
}

// --- Copy 2D ---
cbuffer Copy2DParams : register(b0) {
    uint c2_d1;
    uint c2_d2;
    uint c2_src_stride;
    uint c2_dst_stride;
    uint c2_src_offset;
    uint c2_dst_offset;
};

[numthreads(16, 16, 1)]
void copy2d_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= c2_d1 || tid.y >= c2_d2) return;
    uint src_idx = c2_src_offset + tid.x * c2_src_stride + tid.y;
    uint dst_idx = c2_dst_offset + tid.x * c2_dst_stride + tid.y;
    cs_output[dst_idx] = cs_input[src_idx];
}
