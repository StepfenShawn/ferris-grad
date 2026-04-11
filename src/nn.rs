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
            }
        }
        Ok(())
    }
}

pub trait Module {
    // Retrieve all trainable parameters as a list of tensor.
    fn parameters(&mut self) -> Vec<&mut Tensor>;

    // Perform a forward pass through the module.
    fn forward(&self, inputs: &Tensor) -> Tensor;
}

impl Module for Linear {
    fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.w, &mut self.b]
    }

    fn forward(&self, inputs: &Tensor) -> Tensor {
        &inputs.dot(&self.w).unwrap() + &self.b
    }
}

impl Module for Block {
    fn parameters(&mut self) -> Vec<&mut Tensor> {
        match self {
            Block::Linear(l) => l.parameters(),
            Block::Relu | Block::Tanh => vec![],
        }
    }

    fn forward(&self, inputs: &Tensor) -> Tensor {
        match self {
            Block::Linear(l) => l.forward(inputs),
            Block::Relu => inputs.relu(),
            Block::Tanh => inputs.tanh(),
        }
    }
}

impl Module for Sequential {
    fn parameters(&mut self) -> Vec<&mut Tensor> {
        self.blocks
            .iter_mut()
            .flat_map(|b| b.parameters())
            .collect()
    }

    fn forward(&self, inputs: &Tensor) -> Tensor {
        self.blocks
            .iter()
            .fold(inputs.clone(), |acc, b| b.forward(&acc))
    }
}
