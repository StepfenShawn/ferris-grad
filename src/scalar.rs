use std::{
    cell::RefCell,
    collections::{HashSet, VecDeque},
    hash::Hash,
    ops::{Add, Deref, Mul},
    rc::Rc,
};

#[derive(Debug, Clone)]
pub enum Operation {
    None,
    Add,
    Mul,
    Pow,
    Tanh,
}

type PropagateFn = fn(&_Scalar);

#[derive(Debug, Clone)]
pub struct _Scalar {
    data: f64,
    grad: f64,
    op: Operation,
    prev: Vec<Scalar>,
    propagate_fn: Option<PropagateFn>,
}

#[derive(Debug, Clone)]
pub struct Scalar(Rc<RefCell<_Scalar>>);

impl Hash for Scalar {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state);
    }
}

impl PartialEq for Scalar {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Scalar {}

impl Deref for Scalar {
    type Target = Rc<RefCell<_Scalar>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Scalar {
    pub fn new(s: _Scalar) -> Self {
        Self(Rc::new(RefCell::new(s)))
    }

    pub fn from_f64(f: f64) -> Self {
        Self::new(_Scalar {
            data: f,
            grad: 0.,
            op: Operation::None,
            prev: vec![],
            propagate_fn: None,
        })
    }

    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    pub fn backward(&self) {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(self.clone());
        self.borrow_mut().grad = 0.;
        while let Some(ref scalar) = queue.pop_front() {
            if visited.contains(scalar) {
                continue;
            }
            visited.insert(scalar.clone());
            let borrowed_scalar = scalar.borrow();
            if let Some(f) = borrowed_scalar.propagate_fn {
                f(&borrowed_scalar);
            }
            for child in &borrowed_scalar.prev {
                if !visited.contains(child) {
                    queue.push_back(child.clone());
                }
            }
        }
    }
}

fn add(a: &Scalar, b: &Scalar) -> Scalar {
    let propagate_fn = |v: &_Scalar| {
        let mut first_scalar = v.prev[0].borrow_mut();
        let mut second_scalar = v.prev[1].borrow_mut();

        first_scalar.grad += v.grad;
        second_scalar.grad += v.grad;
    };

    let result = a.borrow().data + b.borrow().data;

    Scalar::new(_Scalar {
        data: result,
        grad: 0.,
        op: Operation::Add,
        prev: vec![a.clone(), b.clone()],
        propagate_fn: Some(propagate_fn),
    })
}

fn mul(a: &Scalar, b: &Scalar) -> Scalar {
    let propagate_fn = |v: &_Scalar| {
        let mut first_scalar = v.prev[0].borrow_mut();
        let mut second_scalar = v.prev[1].borrow_mut();

        first_scalar.grad += second_scalar.data * v.grad;
        second_scalar.grad += first_scalar.data * v.grad;
    };

    let result = a.borrow().data * b.borrow().data;

    Scalar::new(_Scalar {
        data: result,
        grad: 0.,
        op: Operation::Mul,
        prev: vec![a.clone(), b.clone()],
        propagate_fn: Some(propagate_fn),
    })
}

impl Add<Scalar> for Scalar {
    type Output = Scalar;
    fn add(self, other: Scalar) -> Self::Output {
        add(&self, &other)
    }
}

impl Mul<Scalar> for Scalar {
    type Output = Scalar;

    fn mul(self, other: Scalar) -> Self::Output {
        mul(&self, &other)
    }
}
