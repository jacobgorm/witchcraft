// Naive matrix multiplication: C = A * B
// A is (B, M, K), B is (B, K, N), C is (B, M, N)
// One thread per output element.

cbuffer Params : register(b0) {
    uint batch_size;
    uint M;
    uint N;
    uint K;
    uint lhs_stride_b;   // Batch stride for A
    uint lhs_stride_m;   // Row stride for A
    uint lhs_stride_k;   // K-dimension stride for A
    uint rhs_stride_b;   // Batch stride for B
    uint rhs_stride_k;   // Row stride for B (K-dimension)
    uint rhs_stride_n;   // Column stride for B (N-dimension)
};

StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> B : register(t1);
RWStructuredBuffer<float> C : register(u0);

[numthreads(8, 8, 1)]
void matmul_f32(uint3 tid : SV_DispatchThreadID) {
    uint col = tid.x;  // n
    uint row = tid.y;  // m
    uint batch = tid.z; // b

    if (col >= N || row >= M || batch >= batch_size) return;

    uint a_base = batch * lhs_stride_b + row * lhs_stride_m;
    uint b_base = batch * rhs_stride_b + col * rhs_stride_n;

    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += A[a_base + k * lhs_stride_k] * B[b_base + k * rhs_stride_k];
    }

    C[batch * M * N + row * N + col] = sum;
}
