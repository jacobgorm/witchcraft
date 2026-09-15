// Unary operations on f32 tensors.

#define PI 3.14159265358979323846f
#define GELU_COEF_A 0.044715f
#define BETA 0.7978845608028654f  // sqrt(2/pi)

cbuffer Params : register(b0) {
    uint count;
};

StructuredBuffer<float> input : register(t0);
RWStructuredBuffer<float> output : register(u0);

// --- Contiguous kernels ---

[numthreads(256, 1, 1)]
void copy_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    output[tid.x] = input[tid.x];
}

[numthreads(256, 1, 1)]
void neg_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    output[tid.x] = -input[tid.x];
}

[numthreads(256, 1, 1)]
void abs_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    output[tid.x] = abs(input[tid.x]);
}

[numthreads(256, 1, 1)]
void exp_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    output[tid.x] = exp(input[tid.x]);
}

[numthreads(256, 1, 1)]
void log_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    output[tid.x] = log(input[tid.x]);
}

[numthreads(256, 1, 1)]
void sin_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    output[tid.x] = sin(input[tid.x]);
}

[numthreads(256, 1, 1)]
void cos_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    output[tid.x] = cos(input[tid.x]);
}

[numthreads(256, 1, 1)]
void sqrt_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    output[tid.x] = sqrt(input[tid.x]);
}

[numthreads(256, 1, 1)]
void sqr_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    float v = input[tid.x];
    output[tid.x] = v * v;
}

[numthreads(256, 1, 1)]
void recip_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    output[tid.x] = 1.0f / input[tid.x];
}

[numthreads(256, 1, 1)]
void relu_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    output[tid.x] = max(input[tid.x], 0.0f);
}

[numthreads(256, 1, 1)]
void sigmoid_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    output[tid.x] = 1.0f / (1.0f + exp(-input[tid.x]));
}

[numthreads(256, 1, 1)]
void tanh_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    output[tid.x] = tanh(input[tid.x]);
}

[numthreads(256, 1, 1)]
void gelu_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    float x = input[tid.x];
    // GELU using tanh approximation
    output[tid.x] = 0.5f * x * (1.0f + tanh(BETA * (x + GELU_COEF_A * x * x * x)));
}

[numthreads(256, 1, 1)]
void gelu_erf_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    float x = input[tid.x];
    // GELU using erf: 0.5 * x * (1 + erf(x / sqrt(2)))
    // HLSL doesn't have erf(), approximate with tanh-based formula
    // erf(x) ≈ tanh(sqrt(2/pi) * (x + 0.044715 * x^3)) for the GELU context
    // But the actual erf-based GELU is: 0.5 * x * (1 + erf(x * 0.7071067811865476))
    // We'll use the rational approximation of erf
    float a = x * 0.7071067811865476f; // x / sqrt(2)
    float t = 1.0f / (1.0f + 0.3275911f * abs(a));
    float t2 = t * t;
    float t3 = t2 * t;
    float t4 = t3 * t;
    float t5 = t4 * t;
    float erf_approx = 1.0f - (0.254829592f * t - 0.284496736f * t2 + 1.421413741f * t3
                                - 1.453152027f * t4 + 1.061405429f * t5) * exp(-a * a);
    if (a < 0.0f) erf_approx = -erf_approx;
    output[tid.x] = 0.5f * x * (1.0f + erf_approx);
}

[numthreads(256, 1, 1)]
void silu_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    float x = input[tid.x];
    output[tid.x] = x / (1.0f + exp(-x));
}

[numthreads(256, 1, 1)]
void ceil_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    output[tid.x] = ceil(input[tid.x]);
}

[numthreads(256, 1, 1)]
void floor_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    output[tid.x] = floor(input[tid.x]);
}

[numthreads(256, 1, 1)]
void round_f32(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) return;
    output[tid.x] = round(input[tid.x]);
}

// --- Strided kernel (generic, dispatched per-op via defines) ---

cbuffer StridedParams : register(b0) {
    uint s_count;
    uint s_num_dims;
};

ByteAddressBuffer meta : register(t1);

// Strided copy (used as the generic strided path)
[numthreads(256, 1, 1)]
void copy_f32_strided(uint3 tid : SV_DispatchThreadID) {
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
    output[tid.x] = input[strided_i];
}
