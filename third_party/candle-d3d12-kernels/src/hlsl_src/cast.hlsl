// Type casting operations.
// Since HLSL structured buffers are typed, we use ByteAddressBuffer
// for generic access and interpret as needed.

cbuffer Params : register(b0) {
    uint count;
};

// f32 -> u32
StructuredBuffer<float> cast_in_f32 : register(t0);
RWStructuredBuffer<uint> cast_out_u32 : register(u0);

[numthreads(256, 1, 1)]
void cast_f32_to_u32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    cast_out_u32[tid.x] = (uint)cast_in_f32[tid.x];
}

// u32 -> f32
StructuredBuffer<uint> cast_in_u32 : register(t0);
RWStructuredBuffer<float> cast_out_f32 : register(u0);

[numthreads(256, 1, 1)]
void cast_u32_to_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    cast_out_f32[tid.x] = (float)cast_in_u32[tid.x];
}

// f64 -> f32 (using ByteAddressBuffer for f64)
ByteAddressBuffer cast_in_raw : register(t0);

[numthreads(256, 1, 1)]
void cast_f64_to_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    // Load f64 as two uint32s
    uint lo = cast_in_raw.Load(tid.x * 8);
    uint hi = cast_in_raw.Load(tid.x * 8 + 4);
    // Extract sign, exponent, mantissa from f64
    uint sign = (hi >> 31) & 1;
    int exp64 = (int)((hi >> 20) & 0x7FF) - 1023;
    uint mant_hi = hi & 0xFFFFF;

    // Convert to f32
    float result;
    if (exp64 == 1024) {
        // Inf or NaN
        result = (mant_hi == 0 && lo == 0) ? asfloat(sign << 31 | 0x7F800000)
                                            : asfloat(sign << 31 | 0x7FC00000);
    } else if (exp64 < -126) {
        result = 0.0f;
    } else if (exp64 > 127) {
        result = asfloat(sign << 31 | 0x7F800000); // Inf
    } else {
        uint f32_exp = (uint)(exp64 + 127) & 0xFF;
        uint f32_mant = mant_hi >> (20 - 23 + 20); // top 23 bits of mantissa
        f32_mant = (mant_hi << 3) | (lo >> 29);
        result = asfloat((sign << 31) | (f32_exp << 23) | (f32_mant & 0x7FFFFF));
    }
    cast_out_f32[tid.x] = result;
}
