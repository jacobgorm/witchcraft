// 1D convolution: standard and causal

cbuffer Params : register(b0) {
    uint batch_size;
    uint c_in;       // Input channels
    uint c_out;      // Output channels
    uint l_in;       // Input length
    uint l_out;      // Output length
    uint k_size;     // Kernel size
    uint stride;
    uint padding;
    uint dilation;
};

StructuredBuffer<float> input : register(t0);   // (batch, c_in, l_in)
StructuredBuffer<float> kernel : register(t1);  // (c_out, c_in, k_size)
RWStructuredBuffer<float> output : register(u0); // (batch, c_out, l_out)

[numthreads(256, 1, 1)]
void conv1d_f32(uint3 tid : SV_DispatchThreadID) {
    uint total = batch_size * c_out * l_out;
    if (tid.x >= total) return;

    uint l = tid.x % l_out;
    uint co = (tid.x / l_out) % c_out;
    uint b = tid.x / (l_out * c_out);

    float sum = 0.0f;
    for (uint ci = 0; ci < c_in; ci++) {
        for (uint ki = 0; ki < k_size; ki++) {
            int in_pos = (int)(l * stride) + (int)(ki * dilation) - (int)padding;
            if (in_pos >= 0 && (uint)in_pos < l_in) {
                float inp = input[b * c_in * l_in + ci * l_in + (uint)in_pos];
                float w = kernel[co * c_in * k_size + ci * k_size + ki];
                sum += inp * w;
            }
        }
    }
    output[b * c_out * l_out + co * l_out + l] = sum;
}
