use crate::scalar::Scalar;
use anyhow::{Ok, Result};
use ndarray::{ArrayD, Ix2, IxDyn, Zip};

#[derive(Debug, Clone)]
pub struct Tensor {
    data: ArrayD<Scalar>,
}

impl Tensor {
    pub fn new(data: ArrayD<Scalar>) -> Self {
        Tensor { data }
    }

    pub fn from_vec(data: Vec<Scalar>, shape: Vec<usize>) -> Result<Self> {
        let arr = ArrayD::from_shape_vec(IxDyn(&shape), data)?;
        Ok(Self::new(arr))
    }

    pub fn zeros(shape: Vec<usize>) -> Result<Self> {
        let lens = shape.iter().fold(1, |acc, x| acc * x);
        Ok(Self::from_vec(vec![Scalar::from_f64(0.); lens], shape)?)
    }

    pub fn ones(shape: Vec<usize>) -> Result<Self> {
        let lens = shape.iter().fold(1, |acc, x| acc * x);
        Ok(Self::from_vec(vec![Scalar::from_f64(1.); lens], shape)?)
    }

    pub fn from_fn<F>(shape: Vec<usize>, f: F) -> Result<Self>
    where
        F: FnMut(IxDyn) -> Scalar,
    {
        let arr = ArrayD::from_shape_fn(shape, f);
        Ok(Self::new(arr))
    }

    pub fn shape(&self) -> Vec<usize> {
        self.data.shape().to_vec()
    }
}
