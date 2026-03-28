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

#[derive(Clone)]
pub struct MLP {
    layers: Vec<Linear>,
}

impl std::fmt::Display for MLP {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let info = self
            .layers
            .iter()
            .map(|l| format!("w: {} b: {}\n", l.w, l.b))
            .collect::<Vec<String>>()
            .concat();
        write!(f, "MLP:\n {}", info)
    }
}

impl MLP {
    pub fn new(layers: Vec<Linear>) -> MLP {
        MLP { layers }
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

impl Module for MLP {
    fn parameters(&mut self) -> Vec<&mut Tensor> {
        self.layers
            .iter_mut()
            .map(|layer| layer.parameters())
            .flatten()
            .collect()
    }

    fn forward(&self, inputs: &Tensor) -> Tensor {
        self.layers
            .iter()
            .fold(inputs.clone(), |acc, layer| layer.forward(&acc))
    }
}
