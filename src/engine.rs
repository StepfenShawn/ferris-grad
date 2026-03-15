use anyhow::Result;
use ndarray::{ArrayD, IxDyn, Zip, array};
use std::cell::RefCell;
use std::collections::{HashSet, VecDeque};
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::{Add, Deref, Mul};
use std::rc::Rc;

#[derive(Debug, Clone)]
pub enum Operation {
    None,
    Add,
    Mul,
    Pow,
    Tanh,
}

type PropagateFn = fn(&_Tensor);

#[derive(Clone)]
pub struct _Tensor {
    data: ArrayD<f32>,
    grad: ArrayD<f32>,
    op: Operation,
    prev: Vec<Tensor>,
    propagate_fn: Option<PropagateFn>,
}

impl _Tensor {
    pub fn new(
        data: ArrayD<f32>,
        op: Operation,
        prev: Vec<Tensor>,
        propagate_fn: Option<PropagateFn>,
    ) -> Self {
        let shape = data.shape();
        let grad = ArrayD::zeros(IxDyn(&shape));
        Self {
            data: data,
            grad,
            op,
            prev,
            propagate_fn,
        }
    }

    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        let arr = ArrayD::from_shape_vec(IxDyn(&shape), data)?;
        Ok(Self::new(arr, Operation::None, Vec::new(), None))
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

    pub fn zeros(shape: Vec<usize>) -> Result<Self> {
        let arr = ArrayD::zeros(IxDyn(&shape));
        Ok(Self::new(arr, Operation::None, Vec::new(), None))
    }

    pub fn ones(shape: Vec<usize>) -> Result<Self> {
        let arr = ArrayD::ones(IxDyn(&shape));
        Ok(Self::new(arr, Operation::None, Vec::new(), None))
    }
}

impl Debug for _Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("_Tensor")
            .field("data", &self.data)
            .field("grad", &self.grad)
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct Tensor(Rc<RefCell<_Tensor>>);

impl Hash for Tensor {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state);
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Tensor {}

impl Deref for Tensor {
    type Target = Rc<RefCell<_Tensor>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Tensor {
    pub fn new(t: _Tensor) -> Self {
        Self(Rc::new(RefCell::new(t)))
    }

    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        Ok(Self::new(_Tensor::from_vec(data, shape)?))
    }

    pub fn zeros(shape: Vec<usize>) -> Result<Self> {
        Ok(Self(Rc::new(RefCell::new(_Tensor::zeros(shape)?))))
    }
    pub fn ones(shape: Vec<usize>) -> Result<Self> {
        Ok(Self(Rc::new(RefCell::new(_Tensor::ones(shape)?))))
    }

    pub fn shape(&self) -> Vec<usize> {
        self.borrow().shape()
    }

    pub fn ndim(&self) -> usize {
        self.borrow().ndim()
    }

    pub fn size(&self) -> usize {
        self.borrow().size()
    }

    pub fn grad(&self) -> ArrayD<f32> {
        self.borrow().grad.clone()
    }

    pub fn backward(&self) -> Result<()> {
        let mut visited = HashSet::new();
        let mut queue: VecDeque<Tensor> = VecDeque::new();
        queue.push_back(self.clone());
        self.borrow_mut().grad = ArrayD::ones(IxDyn(&self.shape()));
        while let Some(tensor) = queue.pop_front() {
            if visited.contains(&tensor) {
                continue;
            }
            visited.insert(tensor.clone());
            let borrowed_tensor = tensor.borrow();
            if let Some(propagate_fn) = borrowed_tensor.propagate_fn {
                let _ = propagate_fn(&borrowed_tensor);
            }
            for child in &borrowed_tensor.prev {
                if !visited.contains(child) {
                    queue.push_back(child.clone());
                }
            }
        }
        Ok(())
    }

    pub fn powf(&self, other: Tensor) -> Tensor {
        let base_data = self.borrow().data.clone();
        let power_data = other.borrow().data.clone();

        let result = Zip::from(&base_data)
            .and(&power_data)
            .map_collect(|x, y| x.powf(*y));

        let propagate_fn: PropagateFn = |t: &_Tensor| {
            let mut base = t.prev[0].borrow_mut();
            let mut power = t.prev[1].borrow_mut();

            let x = base.data.clone();
            let y = power.data.clone();
            let out = t.data.clone();

            // d(x^y)/dx = y * x^(y-1)
            let x_pow_y_minus_1 = Zip::from(&x)
                .and(&(y.clone() - 1.0))
                .map_collect(|a, b| a.powf(*b));
            let base_local_grad = (&y * &x_pow_y_minus_1) * &t.grad;
            base.grad += &base_local_grad;

            // d(x^y)/dy = x^y * ln(x)
            let ln_x = x.mapv(|v| v.ln());
            let power_local_grad = (&out * &ln_x) * &t.grad;
            power.grad += &power_local_grad;
        };

        Tensor::new(_Tensor::new(
            result,
            Operation::Pow,
            vec![self.clone(), other],
            Some(propagate_fn),
        ))
    }

    pub fn tanh(&self) -> Tensor {
        let result = self.borrow().data.tanh();
        let propagate_fn: PropagateFn = |t: &_Tensor| {
            let mut _prev = t.prev[0].borrow_mut();
            // d/dx tanh(x) = 1 - tanh(x)^2
            let local_grad = &t.grad * &(1.0 - t.data.mapv(|v| v.powi(2)));
            _prev.grad += &local_grad;
        };
        Tensor::new(_Tensor::new(
            result,
            Operation::Tanh,
            vec![self.clone()],
            Some(propagate_fn),
        ))
    }
}

fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let propagate_fn = |t: &_Tensor| {
        let mut first_tensor = t.prev[0].borrow_mut();
        let mut second_tensor = t.prev[1].borrow_mut();

        first_tensor.grad += &t.grad;
        second_tensor.grad += &t.grad;
    };

    let result = a.borrow().data.clone() + b.borrow().data.clone();

    Tensor::new(_Tensor::new(
        result,
        Operation::Add,
        vec![a.clone(), b.clone()],
        Some(propagate_fn),
    ))
}

fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    let propagate_fn = |t: &_Tensor| {
        let mut first_tensor = t.prev[0].borrow_mut();
        let mut second_tensor = t.prev[1].borrow_mut();

        first_tensor.grad += &(second_tensor.data.clone() * &t.grad);
        second_tensor.grad += &(first_tensor.data.clone() * &t.grad);
    };

    let result = a.borrow().data.clone() * b.borrow().data.clone();

    Tensor::new(_Tensor::new(
        result,
        Operation::Mul,
        vec![a.clone(), b.clone()],
        Some(propagate_fn),
    ))
}

impl Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, other: Tensor) -> Self::Output {
        add(&self, &other)
    }
}

impl Mul<Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, other: Tensor) -> Self::Output {
        mul(&self, &other)
    }
}
