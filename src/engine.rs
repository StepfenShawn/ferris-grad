use anyhow::Result;
use ndarray::{ArrayD, IxDyn};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub enum Operation {
    None,
    Add,
    Mul,
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: ArrayD<f32>,
    pub grad: Option<ArrayD<f32>>,
    pub op: Operation,
    pub requires_grad: bool,
}

impl Tensor {
    pub fn new(data: ArrayD<f32>, requires_grad: bool) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            data,
            grad: None,
            op: Operation::None,
            requires_grad,
        }))
    }

    pub fn from_vec(
        data: Vec<f32>,
        shape: Vec<usize>,
        requires_grad: bool,
    ) -> Result<Rc<RefCell<Self>>> {
        let shape = IxDyn(&shape);
        let arr = ArrayD::from_shape_vec(shape, data)?;
        Ok(Self::new(arr, requires_grad))
    }
}
