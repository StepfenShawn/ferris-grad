use anyhow::{Ok, Result};
use ferris_grad::{Tensor, nn::Module};

struct DecoderLayer {
    attn_wq: Tensor,
    attn_wk: Tensor,
    attn_wv: Tensor,
    attn_wo: Tensor,
    mlp_fc1: Tensor,
    mlp_fc2: Tensor,
}

impl DecoderLayer {
    fn new(n_embd: usize) -> Self {
        let attn_wq = Tensor::rand([n_embd, n_embd].into()).expect("failed to create attn_wq");
        let attn_wk = Tensor::rand([n_embd, n_embd].into()).expect("failed to create attn_wk");
        let attn_wv = Tensor::rand([n_embd, n_embd].into()).expect("failed to create attn_wv");
        let attn_wo = Tensor::rand([n_embd, n_embd].into()).expect("failed to create attn_wo");
        let mlp_fc1 = Tensor::rand([4 * n_embd, n_embd].into()).expect("failed to create mlp_fc1");
        let mlp_fc2 = Tensor::rand([n_embd, 4 * n_embd].into()).expect("failed to create mlp_fc2");

        Self {
            attn_wq,
            attn_wk,
            attn_wv,
            attn_wo,
            mlp_fc1,
            mlp_fc2,
        }
    }
}

impl Module for DecoderLayer {
    fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![
            &mut self.attn_wq,
            &mut self.attn_wk,
            &mut self.attn_wv,
            &mut self.attn_wo,
            &mut self.mlp_fc1,
            &mut self.mlp_fc2,
        ]
    }

    fn forward(&self, inputs: &Tensor) -> Tensor {
        todo!()
    }
}

fn main() -> Result<()> {
    todo!("transformer in ferris_grad!");
    Ok(())
}
