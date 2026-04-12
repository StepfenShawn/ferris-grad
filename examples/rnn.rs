use anyhow::{Ok, Result};
use ferris_grad::{Tensor, nn::Module};

struct Layer {
    w_ih: Tensor,
    b_ih: Tensor,

    w_hh: Tensor,
    b_hh: Tensor,

    w_hy: Tensor,
    b_hy: Tensor,

    h: Tensor, // hidden state
}

impl Layer {
    pub fn new(
        batch_size: usize,
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
    ) -> Result<Self> {
        Ok(Layer {
            w_ih: Tensor::rand([input_size, hidden_size].into())?,
            b_ih: Tensor::zeros([hidden_size].into())?,
            w_hh: Tensor::rand([hidden_size, hidden_size].into())?,
            b_hh: Tensor::zeros([hidden_size].into())?,
            w_hy: Tensor::rand([hidden_size, output_size].into())?,
            b_hy: Tensor::zeros([output_size].into())?,
            h: Tensor::zeros([batch_size, hidden_size].into())?,
        })
    }
}

impl Module for Layer {
    type Input<'a> = (&'a Tensor, &'a Tensor);
    type Output = Result<(Tensor, Tensor)>;

    fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![
            &mut self.w_ih,
            &mut self.b_ih,
            &mut self.w_hh,
            &mut self.b_hh,
            &mut self.w_hy,
            &mut self.b_hy,
        ]
    }

    fn forward<'a>(&self, inputs: Self::Input<'a>) -> Self::Output
    where
        Self: 'a,
    {
        let (x, h_prev) = inputs;
        let h_next =
            x.dot(&self.w_ih)? + self.b_ih.clone() + h_prev.dot(&self.w_hh)? + self.b_hh.clone();
        let out = h_next.dot(&self.w_hy)? + self.b_hy.clone();
        Ok((h_next, out))
    }
}

struct RNN {
    layers: Vec<Layer>,
}

fn main() -> Result<()> {
    // let rnn = RNN::new();
    Ok(())
}
