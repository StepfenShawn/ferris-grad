use std::{
    cell::RefCell,
    collections::{HashSet, VecDeque},
    hash::Hash,
    iter::Sum,
    ops::{Add, Deref, Mul, Neg, Sub},
    rc::Rc,
};

#[derive(Debug, Clone)]
pub enum Operation {
    None,
    Add,
    Mul,
    Pow,
    Log,
    Exp,
    Relu,
    Tanh,
}

type PropagateFn = fn(&_Scalar);

#[derive(Clone)]
pub struct _Scalar {
    data: f64,
    grad: f64,
    op: Operation,
    prev: Vec<Scalar>,
    propagate_fn: Option<PropagateFn>,
}

impl std::fmt::Display for _Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.data)
    }
}

#[derive(Clone)]
pub struct Scalar(Rc<RefCell<_Scalar>>);

impl std::fmt::Display for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.data())
    }
}

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

    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn zero_grad(&self) {
        self.0.borrow_mut().grad = 0.;
    }

    pub fn apply<F: Fn(f64) -> f64>(&self, f: F) {
        self.0.borrow_mut().data = f(self.data());
    }

    pub fn adjust(&self, factor: f64) {
        let mut value = self.0.borrow_mut();
        value.data += factor * value.grad;
    }

    pub fn backward(&self) {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(self.clone());
        self.borrow_mut().grad = 1.;
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

impl Scalar {
    pub fn pow(&self, other: &Scalar) -> Scalar {
        let result = self.data().powf(other.data());
        let propagate_fn = |v: &_Scalar| {
            let mut base_scalar = v.prev[0].borrow_mut();
            let power_scalar = v.prev[1].borrow_mut();
            base_scalar.grad +=
                power_scalar.data * (base_scalar.data.powf(power_scalar.data - 1.)) * v.grad;
        };

        Scalar::new(_Scalar {
            data: result,
            grad: 0.,
            op: Operation::Pow,
            prev: vec![self.clone(), other.clone()],
            propagate_fn: Some(propagate_fn),
        })
    }

    pub fn log(&self) -> Scalar {
        let result = self.data().ln();
        let propagate_fn = |v: &_Scalar| {
            let mut prev = v.prev[0].borrow_mut();
            prev.grad += (1. / prev.data) * v.grad;
        };

        Scalar::new(_Scalar {
            data: result,
            grad: 0.,
            op: Operation::Log,
            prev: vec![self.clone()],
            propagate_fn: Some(propagate_fn),
        })
    }

    pub fn exp(&self) -> Scalar {
        let result = self.data().exp();
        let propagate_fn = |v: &_Scalar| {
            let mut prev = v.prev[0].borrow_mut();
            prev.grad += prev.data.exp() * v.grad;
        };

        Scalar::new(_Scalar {
            data: result,
            grad: 0.,
            op: Operation::Exp,
            prev: vec![self.clone()],
            propagate_fn: Some(propagate_fn),
        })
    }

    pub fn relu(&self) -> Scalar {
        let result = self.data().max(0.);
        let propagate_fn = |v: &_Scalar| {
            let mut prev = v.prev[0].borrow_mut();
            prev.grad += (if prev.data > 0. { 1. } else { 0. }) * v.grad;
        };

        Scalar::new(_Scalar {
            data: result,
            grad: 0.,
            op: Operation::Relu,
            prev: vec![self.clone()],
            propagate_fn: Some(propagate_fn),
        })
    }

    pub fn tanh(&self) -> Scalar {
        let result = self.data().tanh();
        let propagate_fn = |v: &_Scalar| {
            let mut prev = v.prev[0].borrow_mut();
            prev.grad += (1.0 - v.data.powf(2.0)) * v.grad;
        };

        Scalar::new(_Scalar {
            data: result,
            grad: 0.,
            op: Operation::Tanh,
            prev: vec![self.clone()],
            propagate_fn: Some(propagate_fn),
        })
    }
}
fn add(a: &Scalar, b: &Scalar) -> Scalar {
    let propagate_fn = |v: &_Scalar| {
        if v.prev[0] == v.prev[1] {
            let mut s = v.prev[0].borrow_mut();
            s.grad += 2.0 * v.grad;
        } else {
            let mut first_scalar = v.prev[0].borrow_mut();
            let mut second_scalar = v.prev[1].borrow_mut();
            first_scalar.grad += v.grad;
            second_scalar.grad += v.grad;
        }
    };

    let result = a.data() + b.data();

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
        if v.prev[0] == v.prev[1] {
            let mut s = v.prev[0].borrow_mut();
            s.grad += 2.0 * s.data * v.grad;
        } else {
            let mut first_scalar = v.prev[0].borrow_mut();
            let mut second_scalar = v.prev[1].borrow_mut();
            first_scalar.grad += second_scalar.data * v.grad;
            second_scalar.grad += first_scalar.data * v.grad;
        }
    };

    let result = a.data() * b.data();

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

impl<'a, 'b> Add<&'b Scalar> for &'a Scalar {
    type Output = Scalar;
    fn add(self, other: &'b Scalar) -> Self::Output {
        add(self, other)
    }
}

impl Mul<Scalar> for Scalar {
    type Output = Scalar;

    fn mul(self, other: Scalar) -> Self::Output {
        mul(&self, &other)
    }
}

impl<'a, 'b> Mul<&'b Scalar> for &'a Scalar {
    type Output = Scalar;
    fn mul(self, other: &'b Scalar) -> Self::Output {
        mul(self, other)
    }
}

impl<T: Into<f64>> From<T> for Scalar {
    fn from(value: T) -> Self {
        let value = value.into();
        Self::from_f64(value)
    }
}

impl Neg for Scalar {
    type Output = Scalar;
    fn neg(self) -> Self::Output {
        mul(&self, &Scalar::from_f64(-1.0))
    }
}

impl<'a> Neg for &'a Scalar {
    type Output = Scalar;
    fn neg(self) -> Self::Output {
        mul(self, &Scalar::from_f64(-1.0))
    }
}

impl Sub<Scalar> for Scalar {
    type Output = Scalar;
    fn sub(self, other: Scalar) -> Self::Output {
        add(&self, &(-other))
    }
}

impl<'a, 'b> Sub<&'b Scalar> for &'a Scalar {
    type Output = Scalar;
    fn sub(self, other: &'b Scalar) -> Self::Output {
        add(self, &(-other))
    }
}

impl Sum for Scalar {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Scalar::from_f64(0.0), |acc, x| acc + x)
    }
}
