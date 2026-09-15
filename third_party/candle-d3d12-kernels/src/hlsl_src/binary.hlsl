// Binary operations on f32 tensors.

cbuffer Params : register(b0) {
    uint count;
};

StructuredBuffer<float> lhs : register(t0);
StructuredBuffer<float> rhs : register(t1);
RWStructuredBuffer<float> output : register(u0);

[numthreads(256, 1, 1)]
void add_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    output[tid.x] = lhs[tid.x] + rhs[tid.x];
}

[numthreads(256, 1, 1)]
void sub_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    output[tid.x] = lhs[tid.x] - rhs[tid.x];
}

[numthreads(256, 1, 1)]
void mul_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    output[tid.x] = lhs[tid.x] * rhs[tid.x];
}

[numthreads(256, 1, 1)]
void div_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    output[tid.x] = lhs[tid.x] / rhs[tid.x];
}

[numthreads(256, 1, 1)]
void min_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    output[tid.x] = min(lhs[tid.x], rhs[tid.x]);
}

[numthreads(256, 1, 1)]
void max_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    output[tid.x] = max(lhs[tid.x], rhs[tid.x]);
}

// --- Strided versions ---

cbuffer StridedParams : register(b0) {
    uint s_count;
    uint s_num_dims;
};

ByteAddressBuffer lhs_meta : register(t2);   // lhs dims[] then strides[]
ByteAddressBuffer rhs_meta : register(t3);   // rhs dims[] then strides[]

uint strided_index(uint idx, uint num_dims, ByteAddressBuffer m, uint dims_off, uint strides_off) {
    uint si = 0;
    for (uint d = 0; d < num_dims; d++) {
        uint di = num_dims - 1 - d;
        uint ds = m.Load(dims_off + di * 4);
        uint st = m.Load(strides_off + di * 4);
        si += (idx % ds) * st;
        idx /= ds;
    }
    return si;
}

[numthreads(256, 1, 1)]
void add_f32_strided(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= s_count) return;
    uint li = strided_index(tid.x, s_num_dims, lhs_meta, 0, s_num_dims * 4);
    uint ri = strided_index(tid.x, s_num_dims, rhs_meta, 0, s_num_dims * 4);
    output[tid.x] = lhs[li] + rhs[ri];
}

[numthreads(256, 1, 1)]
void sub_f32_strided(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= s_count) return;
    uint li = strided_index(tid.x, s_num_dims, lhs_meta, 0, s_num_dims * 4);
    uint ri = strided_index(tid.x, s_num_dims, rhs_meta, 0, s_num_dims * 4);
    output[tid.x] = lhs[li] - rhs[ri];
}

[numthreads(256, 1, 1)]
void mul_f32_strided(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= s_count) return;
    uint li = strided_index(tid.x, s_num_dims, lhs_meta, 0, s_num_dims * 4);
    uint ri = strided_index(tid.x, s_num_dims, rhs_meta, 0, s_num_dims * 4);
    output[tid.x] = lhs[li] * rhs[ri];
}

[numthreads(256, 1, 1)]
void div_f32_strided(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= s_count) return;
    uint li = strided_index(tid.x, s_num_dims, lhs_meta, 0, s_num_dims * 4);
    uint ri = strided_index(tid.x, s_num_dims, rhs_meta, 0, s_num_dims * 4);
    output[tid.x] = lhs[li] / rhs[ri];
}
