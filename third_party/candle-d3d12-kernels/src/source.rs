pub const AFFINE: &str = include_str!("hlsl_src/affine.hlsl");
pub const BINARY: &str = include_str!("hlsl_src/binary.hlsl");
pub const CAST: &str = include_str!("hlsl_src/cast.hlsl");
pub const CONV: &str = include_str!("hlsl_src/conv.hlsl");
pub const FILL: &str = include_str!("hlsl_src/fill.hlsl");
pub const INDEXING: &str = include_str!("hlsl_src/indexing.hlsl");
pub const MATMUL: &str = include_str!("hlsl_src/matmul.hlsl");
pub const REDUCE: &str = include_str!("hlsl_src/reduce.hlsl");
pub const TERNARY: &str = include_str!("hlsl_src/ternary.hlsl");
pub const UNARY: &str = include_str!("hlsl_src/unary.hlsl");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    Affine,
    Binary,
    Cast,
    Conv,
    Fill,
    Indexing,
    Matmul,
    Reduce,
    Ternary,
    Unary,
}

impl Source {
    pub fn hlsl_source(&self) -> &'static str {
        match self {
            Self::Affine => AFFINE,
            Self::Binary => BINARY,
            Self::Cast => CAST,
            Self::Conv => CONV,
            Self::Fill => FILL,
            Self::Indexing => INDEXING,
            Self::Matmul => MATMUL,
            Self::Reduce => REDUCE,
            Self::Ternary => TERNARY,
            Self::Unary => UNARY,
        }
    }
}
