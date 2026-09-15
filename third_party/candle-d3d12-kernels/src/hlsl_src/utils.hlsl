// Shared utilities for candle D3D12 compute shaders.

uint get_strided_index(uint idx, uint num_dims, ByteAddressBuffer meta, uint dims_offset, uint strides_offset) {
    uint strided_i = 0;
    for (uint d = 0; d < num_dims; d++) {
        uint dim_idx = num_dims - 1 - d;
        uint dim_size = meta.Load(dims_offset + dim_idx * 4);
        uint stride = meta.Load(strides_offset + dim_idx * 4);
        strided_i += (idx % dim_size) * stride;
        idx /= dim_size;
    }
    return strided_i;
}
