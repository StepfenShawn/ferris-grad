pub mod scalar;
pub mod tensor;
pub use crate::tensor::Tensor;

pub mod nn;
pub use crate::nn::{Linear, MLP};
