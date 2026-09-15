// Reduction operations: sum, mean, max, argmax, softmax.

// --- Sum reduction ---
// Each threadgroup reduces one output element.
// work_per_group elements are reduced per threadgroup.

cbuffer Params : register(b0) {
    uint total_length;     // Total input elements
    uint out_length;       // Number of output elements
    uint work_per_group;   // Elements per output = total_length / out_length
    uint stride;           // Stride between elements to reduce (for non-contiguous)
};

StructuredBuffer<float> input : register(t0);
RWStructuredBuffer<float> output : register(u0);

groupshared float shared_data[256];

[numthreads(256, 1, 1)]
void sum_f32(uint3 tid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint out_idx = gid.x;
    uint local_id = gtid.x;
    uint base = out_idx * work_per_group;

    // Each thread accumulates its portion
    float acc = 0.0f;
    for (uint i = local_id; i < work_per_group; i += 256) {
        if (base + i < total_length) {
            acc += input[base + i];
        }
    }
    shared_data[local_id] = acc;

    GroupMemoryBarrierWithGroupSync();

    // Tree reduction in shared memory
    for (uint s = 128; s > 0; s >>= 1) {
        if (local_id < s) {
            shared_data[local_id] += shared_data[local_id + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (local_id == 0) {
        output[out_idx] = shared_data[0];
    }
}

[numthreads(256, 1, 1)]
void mean_f32(uint3 tid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint out_idx = gid.x;
    uint local_id = gtid.x;
    uint base = out_idx * work_per_group;

    float acc = 0.0f;
    for (uint i = local_id; i < work_per_group; i += 256) {
        if (base + i < total_length) {
            acc += input[base + i];
        }
    }
    shared_data[local_id] = acc;

    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128; s > 0; s >>= 1) {
        if (local_id < s) {
            shared_data[local_id] += shared_data[local_id + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (local_id == 0) {
        output[out_idx] = shared_data[0] / float(work_per_group);
    }
}

[numthreads(256, 1, 1)]
void max_f32(uint3 tid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint out_idx = gid.x;
    uint local_id = gtid.x;
    uint base = out_idx * work_per_group;

    float acc = -3.402823e+38f; // -FLT_MAX
    for (uint i = local_id; i < work_per_group; i += 256) {
        if (base + i < total_length) {
            acc = max(acc, input[base + i]);
        }
    }
    shared_data[local_id] = acc;

    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128; s > 0; s >>= 1) {
        if (local_id < s) {
            shared_data[local_id] = max(shared_data[local_id], shared_data[local_id + s]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (local_id == 0) {
        output[out_idx] = shared_data[0];
    }
}

[numthreads(256, 1, 1)]
void min_f32(uint3 tid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint out_idx = gid.x;
    uint local_id = gtid.x;
    uint base = out_idx * work_per_group;

    float acc = 3.402823e+38f; // FLT_MAX
    for (uint i = local_id; i < work_per_group; i += 256) {
        if (base + i < total_length) {
            acc = min(acc, input[base + i]);
        }
    }
    shared_data[local_id] = acc;

    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128; s > 0; s >>= 1) {
        if (local_id < s) {
            shared_data[local_id] = min(shared_data[local_id], shared_data[local_id + s]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (local_id == 0) {
        output[out_idx] = shared_data[0];
    }
}

// --- Argmax/Argmin ---

RWStructuredBuffer<uint> output_idx : register(u0);

groupshared float shared_vals[256];
groupshared uint shared_idxs[256];

[numthreads(256, 1, 1)]
void argmax_f32(uint3 tid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint out_idx = gid.x;
    uint local_id = gtid.x;
    uint base = out_idx * work_per_group;

    float best_val = -3.402823e+38f;
    uint best_idx = 0;
    for (uint i = local_id; i < work_per_group; i += 256) {
        if (base + i < total_length) {
            float v = input[base + i];
            if (v > best_val) {
                best_val = v;
                best_idx = i;
            }
        }
    }
    shared_vals[local_id] = best_val;
    shared_idxs[local_id] = best_idx;

    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128; s > 0; s >>= 1) {
        if (local_id < s) {
            if (shared_vals[local_id + s] > shared_vals[local_id]) {
                shared_vals[local_id] = shared_vals[local_id + s];
                shared_idxs[local_id] = shared_idxs[local_id + s];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (local_id == 0) {
        output_idx[out_idx] = shared_idxs[0];
    }
}

[numthreads(256, 1, 1)]
void argmin_f32(uint3 tid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint out_idx = gid.x;
    uint local_id = gtid.x;
    uint base = out_idx * work_per_group;

    float best_val = 3.402823e+38f;
    uint best_idx = 0;
    for (uint i = local_id; i < work_per_group; i += 256) {
        if (base + i < total_length) {
            float v = input[base + i];
            if (v < best_val) {
                best_val = v;
                best_idx = i;
            }
        }
    }
    shared_vals[local_id] = best_val;
    shared_idxs[local_id] = best_idx;

    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128; s > 0; s >>= 1) {
        if (local_id < s) {
            if (shared_vals[local_id + s] < shared_vals[local_id]) {
                shared_vals[local_id] = shared_vals[local_id + s];
                shared_idxs[local_id] = shared_idxs[local_id + s];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (local_id == 0) {
        output_idx[out_idx] = shared_idxs[0];
    }
}

// --- Softmax (last dimension) ---
// Each threadgroup handles one row (one output softmax)

cbuffer SoftmaxParams : register(b0) {
    uint sm_row_size;    // Size of the last dimension
    uint sm_num_rows;    // Number of rows
};

[numthreads(256, 1, 1)]
void softmax_f32(uint3 tid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint row = gid.x;
    uint local_id = gtid.x;
    uint base = row * sm_row_size;

    // Step 1: Find max
    float local_max = -3.402823e+38f;
    for (uint i = local_id; i < sm_row_size; i += 256) {
        local_max = max(local_max, input[base + i]);
    }
    shared_data[local_id] = local_max;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128; s > 0; s >>= 1) {
        if (local_id < s)
            shared_data[local_id] = max(shared_data[local_id], shared_data[local_id + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    float row_max = shared_data[0];
    GroupMemoryBarrierWithGroupSync();

    // Step 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (uint j = local_id; j < sm_row_size; j += 256) {
        float e = exp(input[base + j] - row_max);
        output[base + j] = e;
        local_sum += e;
    }
    shared_data[local_id] = local_sum;
    GroupMemoryBarrierWithGroupSync();

    for (uint s2 = 128; s2 > 0; s2 >>= 1) {
        if (local_id < s2)
            shared_data[local_id] += shared_data[local_id + s2];
        GroupMemoryBarrierWithGroupSync();
    }
    float row_sum = shared_data[0];
    GroupMemoryBarrierWithGroupSync();

    // Step 3: Normalize
    float inv_sum = 1.0f / row_sum;
    for (uint k = local_id; k < sm_row_size; k += 256) {
        output[base + k] *= inv_sum;
    }
}
