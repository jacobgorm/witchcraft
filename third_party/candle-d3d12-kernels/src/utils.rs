/// Calculate dispatch group count for a 1D kernel with the given thread group size.
pub fn div_ceil(n: u32, d: u32) -> u32 {
    (n + d - 1) / d
}

/// Standard 1D dispatch: 256 threads per group.
pub fn linear_groups(count: u32) -> [u32; 3] {
    [div_ceil(count, 256), 1, 1]
}

/// 2D dispatch for matmul (8x8 threads per group).
pub fn matmul_groups(n: u32, m: u32, batch: u32) -> [u32; 3] {
    [div_ceil(n, 8), div_ceil(m, 8), batch]
}

/// 2D dispatch for copy2d (16x16 threads per group).
pub fn copy2d_groups(d1: u32, d2: u32) -> [u32; 3] {
    [div_ceil(d1, 16), div_ceil(d2, 16), 1]
}
