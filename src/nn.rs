use crate::{scalar::Scalar, tensor::Tensor};

#[derive(Clone)]
pub struct Linear {
    w: Tensor,
    b: Tensor,
}

impl Linear {
    pub fn new(nin: usize, nout: usize) -> Linear {
        let rand_value = rand::random::<f64>();
        Linear {
            w: Tensor::rand(vec![nin, nout]).expect("failed to create w"),
            b: Tensor::from_fn(vec![nout], |_| Scalar::from_f64(rand_value))
                .expect("failed to crate b"),
        }
    }
}

/// One step in a [`Sequential`] stack: affine layer or element-wise activation.
#[derive(Clone)]
pub enum Block {
    Linear(Linear),
    Relu,
    Tanh,
    Sigmoid
}

/// Ordered list of blocks (e.g. `Linear` → `Relu` → `Linear` → …).
#[derive(Clone)]
pub struct Sequential {
    blocks: Vec<Block>,
}

impl Sequential {
    pub fn new(blocks: Vec<Block>) -> Self {
        Sequential { blocks }
    }
}

impl std::fmt::Display for Sequential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Sequential:")?;
        for (i, b) in self.blocks.iter().enumerate() {
            match b {
                Block::Linear(l) => writeln!(f, "  [{i}] Linear w: {} b: {}", l.w, l.b)?,
                Block::Relu => writeln!(f, "  [{i}] Relu")?,
                Block::Tanh => writeln!(f, "  [{i}] Tanh")?,
                Block::Sigmoid => writeln!(f, " [{i}] Sigmoid")?
            }
        }
        Ok(())
    }
}

pub trait Module {
    type Input<'a>
    where
        Self: 'a;
    type Output;

    fn parameters(&mut self) -> Vec<&mut Tensor>;
    fn forward<'a>(&self, inputs: Self::Input<'a>) -> Self::Output
    where
        Self: 'a;
}

impl Module for Linear {
    type Input<'a> = &'a Tensor;
    type Output = Tensor;

    fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.w, &mut self.b]
    }

    fn forward<'a>(&self, inputs: Self::Input<'a>) -> Self::Output
    where
        Self: 'a,
    {
        &inputs.dot(&self.w).unwrap() + &self.b
    }
}

impl Module for Block {
    type Input<'a> = &'a Tensor;
    type Output = Tensor;

    fn parameters(&mut self) -> Vec<&mut Tensor> {
        match self {
            Block::Linear(l) => l.parameters(),
            Block::Relu | Block::Tanh | Block::Sigmoid => vec![],
        }
    }

    fn forward<'a>(&self, inputs: Self::Input<'a>) -> Self::Output
    where
        Self: 'a,
    {
        match self {
            Block::Linear(l) => l.forward(inputs),
            Block::Relu => inputs.relu(),
            Block::Tanh => inputs.tanh(),
            Block::Sigmoid => inputs.sigmoid()
        }
    }
}

impl Module for Sequential {
    type Input<'a> = &'a Tensor;
    type Output = Tensor;

    fn parameters(&mut self) -> Vec<&mut Tensor> {
        self.blocks
            .iter_mut()
            .flat_map(|b| b.parameters())
            .collect()
    }

    fn forward<'a>(&self, inputs: Self::Input<'a>) -> Self::Output
    where
        Self: 'a,
    {
        self.blocks
            .iter()
            .fold(inputs.clone(), |acc, b| b.forward(&acc))
    }
}
