use crate::{scalar::Scalar, tensor::Tensor};

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
pub struct MLP {
    layers: Vec<Linear>,
}

impl MLP {
    pub fn new(layers: Vec<Linear>) -> MLP {
        MLP { layers }
    }
}

pub trait Module {
    // Retrieve all trainable parameters as a list of tensor.
    fn parameters(&self) -> Vec<Tensor>;

    // Perform a forward pass through the module.
    fn forward(&self, inputs: &Tensor) -> Tensor;
}

impl Module for Linear {
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.w.clone(), self.b.clone()]
    }

    fn forward(&self, inputs: &Tensor) -> Tensor {
        inputs.dot(&self.w).unwrap() + self.b.clone()
    }
}

impl Module for MLP {
    fn parameters(&self) -> Vec<Tensor> {
        self.layers.get(0).expect("MLP has no layers").parameters()
    }

    fn forward(&self, inputs: &Tensor) -> Tensor {
        self.layers
            .iter()
            .fold(inputs.clone(), |acc, layer| layer.forward(&acc))
    }
}
