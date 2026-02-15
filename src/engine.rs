use anyhow::Result;
use ndarray::{ArrayD, IxDyn};
use std::cell::RefCell;
use std::rc::Rc;

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum Operation {
    None,
    Add,
    Mul,
}

type PropagateFn = fn(&_Tensor) -> Result<_Tensor>;

#[derive(Debug, Clone)]
struct _Tensor {
    data: ArrayD<f32>,
    grad: Option<ArrayD<f32>>,
    op: Operation,
    requires_grad: bool,
    parents: Vec<Tensor>,
    propagate_fn: Option<PropagateFn>,
}

impl _Tensor {
    pub fn new(data: ArrayD<f32>, requires_grad: bool) -> Self {
        Self {
            data,
            grad: None,
            op: Operation::None,
            requires_grad,
            parents: Vec::new(),
            propagate_fn: None,
        }
    }

    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Result<Self> {
        let shape = IxDyn(&shape);
        let arr = ArrayD::from_shape_vec(shape, data)?;
        Ok(Self::new(arr, requires_grad))
    }

    pub fn shape(&self) -> Vec<usize> {
        self.data.shape().to_vec()
    }

    pub fn ndim(&self) -> usize {
        self.data.shape().len()
    }

    pub fn size(&self) -> usize {
        self.data.shape().iter().product()
    }

    pub fn zeros(shape: Vec<usize>, requires_grad: bool) -> Result<Self> {
        let shape = IxDyn(&shape);
        let arr = ArrayD::zeros(shape);
        Ok(Self::new(arr, requires_grad))
    }

    pub fn ones(shape: Vec<usize>, requires_grad: bool) -> Result<Self> {
        let shape = IxDyn(&shape);
        let arr = ArrayD::ones(shape);
        Ok(Self::new(arr, requires_grad))
    }
}

#[derive(Debug, Clone)]
pub struct Tensor(Rc<RefCell<_Tensor>>);

impl Tensor {
    pub fn new(data: ArrayD<f32>, requires_grad: bool) -> Self {
        Self(Rc::new(RefCell::new(_Tensor::new(data, requires_grad))))
    }

    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Result<Self> {
        Ok(Self(Rc::new(RefCell::new(_Tensor::from_vec(
            data,
            shape,
            requires_grad,
        )?))))
    }

    pub fn zeros(shape: Vec<usize>, requires_grad: bool) -> Result<Self> {
        Ok(Self(Rc::new(RefCell::new(_Tensor::zeros(
            shape,
            requires_grad,
        )?))))
    }
    pub fn ones(shape: Vec<usize>, requires_grad: bool) -> Result<Self> {
        Ok(Self(Rc::new(RefCell::new(_Tensor::ones(
            shape,
            requires_grad,
        )?))))
    }

    pub fn shape(&self) -> Vec<usize> {
        self.0.borrow().shape()
    }

    pub fn ndim(&self) -> usize {
        self.0.borrow().ndim()
    }

    pub fn size(&self) -> usize {
        self.0.borrow().size()
    }
}
