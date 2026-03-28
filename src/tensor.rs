use std::ops::{Add, Mul, Sub};

use crate::scalar::Scalar;
use anyhow::{Ok, Result, anyhow};
use ndarray::{ArrayD, Dimension, IntoDimension, Ix2, IxDyn};

#[derive(Clone)]
pub struct Tensor {
    data: ArrayD<Scalar>,
}

impl std::fmt::Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.data)
    }
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

    pub fn rand(shape: Vec<usize>) -> Result<Self> {
        Ok(Self::from_fn(shape, |_| {
            Scalar::from_f64(rand::random::<f64>())
        })?)
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

    pub fn get<I>(&self, index: I) -> &Scalar
    where
        I: IntoDimension,
    {
        let idx = index.into_dimension().into_dyn();
        self.data.get(idx).expect("failed to get scalar")
    }

    pub fn sum(&self) -> Scalar {
        self.data.iter().map(|x| x.clone()).sum()
    }

    pub fn dot(&self, other: &Tensor) -> Result<Tensor> {
        let lhs = self.data.view().into_dimensionality::<Ix2>()?;
        let rhs = other.data.view().into_dimensionality::<Ix2>()?;

        let (m, k) = lhs.dim();
        let (k2, n) = rhs.dim();
        if k != k2 {
            return Err(anyhow!(
                "dot shape mismatch: left is ({m}, {k}), right is ({k2}, {n})"
            ));
        }

        let values = (0..m)
            .map(|row| {
                (0..n)
                    .map(|col| {
                        (0..k)
                            .map(|t| lhs[(row, t)].clone() * rhs[(t, col)].clone())
                            .sum()
                    })
                    .collect::<Vec<Scalar>>()
            })
            .flatten()
            .collect();

        let result = ArrayD::from_shape_vec(IxDyn(&[m, n]), values)?;
        Ok(Self::new(result))
    }

    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        let result = &self.data + &other.data;
        Ok(Self::new(result))
    }

    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        let result = &self.data - &other.data;
        Ok(Self::new(result))
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        let result = &self.data * &other.data;
        Ok(Self::new(result))
    }

    pub fn for_each<F>(&self, f: F)
    where
        F: Fn(&Scalar),
    {
        for s in self.data.iter() {
            f(s)
        }
    }
}

impl Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, other: Tensor) -> Self::Output {
        Tensor::add(&self, &other).expect("failed to add tensors")
    }
}

impl<'a, 'b> Add<&'b Tensor> for &'a Tensor {
    type Output = Tensor;
    fn add(self, other: &'b Tensor) -> Self::Output {
        Tensor::add(self, other).expect("failed to add tensors")
    }
}

impl Sub<Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, other: Tensor) -> Self::Output {
        Tensor::sub(&self, &other).expect("failed to sub tensors")
    }
}

impl<'a, 'b> Sub<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    fn sub(self, other: &'b Tensor) -> Self::Output {
        Tensor::sub(&self, &other).expect("failed to sub tensors")
    }
}

impl Mul<Tensor> for Tensor {
    type Output = Tensor;
    fn mul(self, other: Tensor) -> Self::Output {
        Tensor::mul(&self, &other).expect("failed to mul tensors")
    }
}

impl<'a, 'b> Mul<&'b Tensor> for &'a Tensor {
    type Output = Tensor;
    fn mul(self, other: &'b Tensor) -> Self::Output {
        Tensor::mul(self, other).expect("failed to mul tensors")
    }
}
