use anyhow::{Ok, Result};
use ferris_grad::{Tensor, nn::Module};

struct Layer {
    w_ih: Tensor,
    b_ih: Tensor,

    w_hh: Tensor,
    b_hh: Tensor,

    w_hy: Tensor,
    b_hy: Tensor,
}

impl Layer {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Result<Self> {
        Ok(Layer {
            w_ih: Tensor::rand([input_size, hidden_size].into())?,
            b_ih: Tensor::zeros([hidden_size].into())?,
            w_hh: Tensor::rand([hidden_size, hidden_size].into())?,
            b_hh: Tensor::zeros([hidden_size].into())?,
            w_hy: Tensor::rand([hidden_size, output_size].into())?,
            b_hy: Tensor::zeros([output_size].into())?,
        })
    }
}

impl Module for Layer {
    fn parameters(&mut self) -> Vec<&mut Tensor> {
        todo!()
    }

    fn forward(&self, inputs: &Tensor) -> Tensor {
        todo!()
    }
}

struct RNN {
    layers: Vec<Layer>,
}

fn main() -> Result<()> {
    // let rnn = RNN::new();
    Ok(())
}
