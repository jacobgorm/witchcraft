// Affine transform: output[i] = input[i] * mul + add

cbuffer Params : register(b0) {
    uint count;
    float mul_val;
    float add_val;
};

StructuredBuffer<float> input : register(t0);
RWStructuredBuffer<float> output : register(u0);

[numthreads(256, 1, 1)]
void affine_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    output[tid.x] = input[tid.x] * mul_val + add_val;
}

// Strided version
cbuffer StridedParams : register(b0) {
    uint s_count;
    float s_mul_val;
    float s_add_val;
    uint s_num_dims;
};

ByteAddressBuffer meta : register(t1);  // dims[] then strides[]

[numthreads(256, 1, 1)]
void affine_f32_strided(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= s_count) return;
    uint dims_offset = 0;
    uint strides_offset = s_num_dims * 4;

    uint strided_i = 0;
    uint idx = tid.x;
    for (uint d = 0; d < s_num_dims; d++) {
        uint dim_idx = s_num_dims - 1 - d;
        uint dim_size = meta.Load(dims_offset + dim_idx * 4);
        uint stride = meta.Load(strides_offset + dim_idx * 4);
        strided_i += (idx % dim_size) * stride;
        idx /= dim_size;
    }

    output[tid.x] = input[strided_i] * s_mul_val + s_add_val;
}
