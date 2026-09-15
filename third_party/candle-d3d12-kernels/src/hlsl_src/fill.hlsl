// Fill a buffer with a constant f32 value.

cbuffer Params : register(b0) {
    uint count;
    float value;
};

RWStructuredBuffer<float> output : register(u0);

[numthreads(256, 1, 1)]
void fill_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    output[tid.x] = value;
}

RWStructuredBuffer<uint> output_u32 : register(u0);

[numthreads(256, 1, 1)]
void fill_u32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    output_u32[tid.x] = asuint(value);
}

RWByteAddressBuffer output_u8 : register(u0);

[numthreads(256, 1, 1)]
void fill_u8(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    // Store byte by loading the dword, masking, and storing back
    uint byte_offset = tid.x;
    uint dword_offset = (byte_offset / 4) * 4;
    uint byte_in_dword = byte_offset % 4;
    uint shift = byte_in_dword * 8;
    uint mask = ~(0xFF << shift);
    uint val = (uint(value) & 0xFF) << shift;
    uint orig;
    output_u8.InterlockedAnd(dword_offset, mask, orig);
    output_u8.InterlockedOr(dword_offset, val, orig);
}
