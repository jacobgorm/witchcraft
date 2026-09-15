// Ternary operations: where_cond

cbuffer Params : register(b0) {
    uint count;
};

StructuredBuffer<uint> cond : register(t0);
StructuredBuffer<float> true_val : register(t1);
StructuredBuffer<float> false_val : register(t2);
RWStructuredBuffer<float> output : register(u0);

[numthreads(256, 1, 1)]
void where_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    // Condition is stored as u8 packed in u32s
    uint byte_idx = tid.x;
    uint dword_idx = byte_idx / 4;
    uint byte_in_dword = byte_idx % 4;
    uint shift = byte_in_dword * 8;
    uint c = (cond[dword_idx] >> shift) & 0xFF;
    output[tid.x] = (c != 0) ? true_val[tid.x] : false_val[tid.x];
}
