use std::ops::{Add, Index, Mul, Sub};

use crate::scalar::Scalar;
use anyhow::{Ok, Result, anyhow};
use ndarray::{ArrayD, ArrayView, Axis, Dimension, IntoDimension, Ix2, IxDyn};

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
    /// Wraps a dynamic-rank ndarray of scalars as a `Tensor`.
    pub fn new(data: ArrayD<Scalar>) -> Self {
        Tensor { data }
    }

    /// Reshapes a flat row-major vector into a tensor with the given dimensions.
    pub fn from_vec(data: Vec<Scalar>, shape: Vec<usize>) -> Result<Self> {
        let arr = ArrayD::from_shape_vec(IxDyn(&shape), data)?;
        Ok(Self::new(arr))
    }

    /// Returns a tensor filled with the scalar value `0`.
    pub fn zeros(shape: Vec<usize>) -> Result<Self> {
        let lens = shape.iter().fold(1, |acc, x| acc * x);
        Ok(Self::from_vec(vec![Scalar::from_f64(0.); lens], shape)?)
    }

    /// Returns a tensor filled with the scalar value `0`, with the same size as `other`.
    pub fn zeros_like(other: &Tensor) -> Result<Self> {
        Ok(Self::zeros(other.shape())?)
    }

    /// Returns a tensor filled with the scalar value `1`.
    pub fn ones(shape: Vec<usize>) -> Result<Self> {
        let lens = shape.iter().fold(1, |acc, x| acc * x);
        Ok(Self::from_vec(vec![Scalar::from_f64(1.); lens], shape)?)
    }

    /// Returns a tensor filled with the scalar value `1`, with the same size as `other`.
    pub fn ones_like(other: &Tensor) -> Result<Self> {
        Ok(Self::ones(other.shape())?)
    }

    /// Returns a tensor filled with random scalar value.
    pub fn rand(shape: Vec<usize>) -> Result<Self> {
        Ok(Self::from_fn(shape, |_| {
            Scalar::from_f64(rand::random::<f64>())
        })?)
    }

    /// Returns a tensor filled with random scalar value, with the same size as `other`.
    pub fn rand_like(other: &Tensor) -> Result<Self> {
        Ok(Self::rand(other.shape()))?
    }

    /// Builds a tensor by evaluating `f` at each multi-index in `shape`.
    pub fn from_fn<F>(shape: Vec<usize>, f: F) -> Result<Self>
    where
        F: FnMut(IxDyn) -> Scalar,
    {
        let arr = ArrayD::from_shape_fn(shape, f);
        Ok(Self::new(arr))
    }

    /// Returns the tensor shape as a vector of axis lengths.
    pub fn shape(&self) -> Vec<usize> {
        self.data.shape().to_vec()
    }

    /// Stacks a list of tensors along a given axis.
    pub fn stack(tensors: Vec<Tensor>, axis: usize) -> Result<Tensor> {
        let arr = ndarray::stack(
            Axis(axis),
            &tensors
                .iter()
                .map(|t| t.data.view())
                .collect::<Vec<ArrayView<Scalar, IxDyn>>>(),
        )?;
        Ok(Self::new(arr))
    }

    /// Concatenates a list of tensors along a given axis.
    pub fn concat(tensors: Vec<Tensor>, axis: usize) -> Result<Tensor> {
        let arr = ndarray::concatenate(
            Axis(axis),
            &tensors
                .iter()
                .map(|t| t.data.view())
                .collect::<Vec<ArrayView<Scalar, IxDyn>>>(),
        )?;
        Ok(Self::new(arr))
    }

    /// Borrows the scalar at `index` (panics if out of bounds).
    pub fn get<I>(&self, index: I) -> &Scalar
    where
        I: IntoDimension,
    {
        let idx = index.into_dimension().into_dyn();
        self.data.get(idx).expect("failed to get scalar")
    }

    /// Reduces all elements to a single scalar sum.
    pub fn sum(&self) -> Scalar {
        self.data.iter().map(|x| x.clone()).sum()
    }

    /// Returns the mean of all elements as a scalar.
    pub fn mean(&self) -> Scalar {
        let n = self.shape().iter().fold(1., |acc, s| acc * (*s as f64));
        self.sum() * Scalar::from_f64(1. / n)
    }

    /// Matrix multiply: `(m, k) @ (k, n)`; both operands must be 2-D.
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

    /// Element-wise addition; shapes must broadcast per ndarray rules.
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        let result = &self.data + &other.data;
        Ok(Self::new(result))
    }

    /// Element-wise subtraction; shapes must broadcast per ndarray rules.
    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        let result = &self.data - &other.data;
        Ok(Self::new(result))
    }

    /// Element-wise multiplication; shapes must broadcast per ndarray rules.
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        let result = &self.data * &other.data;
        Ok(Self::new(result))
    }

    /// Element-wise unary map; shape is preserved.
    pub fn map_scalars<F>(&self, f: F) -> Tensor
    where
        F: FnMut(Scalar) -> Scalar,
    {
        let shape = self.shape();
        let v: Vec<Scalar> = self.data.iter().cloned().map(f).collect();
        Self::from_vec(v, shape).expect("shape unchanged")
    }

    /// Applies ReLU to every element (shape unchanged).
    pub fn relu(&self) -> Tensor {
        self.map_scalars(|s| s.relu())
    }

    /// Applies `tanh` to every element (shape unchanged).
    pub fn tanh(&self) -> Tensor {
        self.map_scalars(|s| s.tanh())
    }

    /// Applies `sigmoid` to every element (shape unchanged).
    pub fn sigmoid(&self) -> Tensor {
        self.map_scalars(|s| s.sigmoid())
    }

    /// Visits each scalar reference in storage order.
    pub fn for_each<F: FnMut(&Scalar)>(&self, f: F) {
        self.data.iter().for_each(f);
    }

    /// Transpose (swaps the last two axes, ndarray semantics).
    pub fn t(&self) -> Tensor {
        Self::new(self.data.t().to_owned())
    }
}

impl Add<Tensor> for Tensor {
    type Output = Tensor;
    /// Element-wise add with another owned tensor.
    fn add(self, other: Tensor) -> Self::Output {
        Tensor::add(&self, &other).expect("failed to add tensors")
    }
}

impl<'a, 'b> Add<&'b Tensor> for &'a Tensor {
    type Output = Tensor;
    /// Element-wise add with a referenced tensor on the right.
    fn add(self, other: &'b Tensor) -> Self::Output {
        Tensor::add(self, other).expect("failed to add tensors")
    }
}

impl Sub<Tensor> for Tensor {
    type Output = Tensor;

    /// Element-wise subtract another owned tensor.
    fn sub(self, other: Tensor) -> Self::Output {
        Tensor::sub(&self, &other).expect("failed to sub tensors")
    }
}

impl<'a, 'b> Sub<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    /// Element-wise subtract a referenced tensor on the right.
    fn sub(self, other: &'b Tensor) -> Self::Output {
        Tensor::sub(&self, &other).expect("failed to sub tensors")
    }
}

impl Mul<Tensor> for Tensor {
    type Output = Tensor;
    /// Element-wise multiply with another owned tensor.
    fn mul(self, other: Tensor) -> Self::Output {
        Tensor::mul(&self, &other).expect("failed to mul tensors")
    }
}

impl<'a, 'b> Mul<&'b Tensor> for &'a Tensor {
    type Output = Tensor;
    /// Element-wise multiply with a referenced tensor on the right.
    fn mul(self, other: &'b Tensor) -> Self::Output {
        Tensor::mul(self, other).expect("failed to mul tensors")
    }
}

macro_rules! impl_index_trait {
    ($t: tt) => {
        impl Index<$t> for Tensor {
            type Output = Scalar;

            /// Indexing sugar for `get`; panics if out of bounds.
            fn index(&self, index: $t) -> &Self::Output {
                self.get(index)
            }
        }
    };
}

impl_index_trait!(usize);
impl_index_trait!((usize, usize));
impl_index_trait!((usize, usize, usize));
impl_index_trait!((usize, usize, usize, usize));
impl_index_trait!((usize, usize, usize, usize, usize));
impl_index_trait!((usize, usize, usize, usize, usize, usize));
