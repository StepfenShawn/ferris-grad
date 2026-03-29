use anyhow::{Ok, Result};
use ferris_grad::{Tensor, scalar::Scalar};
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use std::fs;

const N_LAYER: usize = 1;
const N_EMBD: usize = 16;
const BLOCK_SIZE: usize = 32;
const N_HEAD: usize = 8;
const HEAD_DIM: usize = N_EMBD / N_HEAD;

const LEARNING_RATE: f64 = 0.0001;
const BETA1: f64 = 0.85;
const BETA2: f64 = 0.99;
const EPS_ADAM: f64 = 1e-8;
const NUM_STEPS: usize = 500;
const TEMPERATURE: f64 = 0.5;

fn gauss_08() -> Scalar {
    let u1: f64 = rand::random::<f64>().max(1e-10);
    let u2: f64 = rand::random::<f64>();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    Scalar::from_f64(z * 0.08)
}

fn linear_1d(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let sh = w.shape();
    let nout = sh[0];
    let nin = sh[1];
    debug_assert_eq!(x.shape(), vec![nin]);
    let mut out = Vec::with_capacity(nout);
    for i in 0..nout {
        let mut acc = Scalar::from_f64(0.0);
        for j in 0..nin {
            acc = &acc + &(&w[(i, j)] * &x[j]);
        }
        out.push(acc);
    }
    Tensor::from_vec(out, vec![nout])
}

fn rmsnorm(x: &Tensor) -> Result<Tensor> {
    let n = x.shape()[0];
    let mut sum_sq = Scalar::from_f64(0.0);
    for i in 0..n {
        sum_sq = &sum_sq + &(&x[i] * &x[i]);
    }
    let inv_n = Scalar::from_f64(1.0 / n as f64);
    let ms = &sum_sq * &inv_n;
    let eps = Scalar::from_f64(1e-5);
    let scale = (&ms + &eps).pow(&Scalar::from_f64(-0.5));
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(&x[i] * &scale);
    }
    Tensor::from_vec(out, vec![n])
}

fn softmax(logits: &[Scalar]) -> Vec<Scalar> {
    let max_val = logits
        .iter()
        .map(|x| x.data())
        .fold(f64::NEG_INFINITY, f64::max);
    let max_s = Scalar::from_f64(max_val);
    let exps: Vec<Scalar> = logits.iter().map(|v| (v - &max_s).exp()).collect();
    let total: Scalar = exps.iter().cloned().sum();
    let inv_total = total.pow(&Scalar::from_f64(-1.0));
    exps.iter().map(|e| e * &inv_total).collect()
}

struct LayerParams {
    attn_wq: Tensor,
    attn_wk: Tensor,
    attn_wv: Tensor,
    attn_wo: Tensor,
    mlp_fc1: Tensor,
    mlp_fc2: Tensor,
}

struct GptParams {
    wte: Tensor,
    wpe: Tensor,
    lm_head: Tensor,
    layers: Vec<LayerParams>,
}

impl GptParams {
    fn new(vocab_size: usize) -> Result<Self> {
        let mut layers = Vec::with_capacity(N_LAYER);
        for _ in 0..N_LAYER {
            layers.push(LayerParams {
                attn_wq: Tensor::from_fn([N_EMBD, N_EMBD].into(), |_| gauss_08())?,
                attn_wk: Tensor::from_fn([N_EMBD, N_EMBD].into(), |_| gauss_08())?,
                attn_wv: Tensor::from_fn([N_EMBD, N_EMBD].into(), |_| gauss_08())?,
                attn_wo: Tensor::from_fn([N_EMBD, N_EMBD].into(), |_| gauss_08())?,
                mlp_fc1: Tensor::from_fn([4 * N_EMBD, N_EMBD].into(), |_| gauss_08())?,
                mlp_fc2: Tensor::from_fn([N_EMBD, 4 * N_EMBD].into(), |_| gauss_08())?,
            });
        }
        Ok(Self {
            wte: Tensor::from_fn([vocab_size, N_EMBD].into(), |_| gauss_08())?,
            wpe: Tensor::from_fn([BLOCK_SIZE, N_EMBD].into(), |_| gauss_08())?,
            lm_head: Tensor::from_fn([vocab_size, N_EMBD].into(), |_| gauss_08())?,
            layers,
        })
    }

    fn count_scalars(&self) -> usize {
        let count = |t: &Tensor| t.shape().iter().fold(1, |acc, x| acc * x);
        [&self.wte, &self.wpe, &self.lm_head]
            .into_iter()
            .chain(self.layers.iter().flat_map(|layer| {
                [
                    &layer.attn_wq,
                    &layer.attn_wk,
                    &layer.attn_wv,
                    &layer.attn_wo,
                    &layer.mlp_fc1,
                    &layer.mlp_fc2,
                ]
                .into_iter()
            }))
            .map(|t| count(t))
            .sum()
    }
}

fn gpt(
    token_id: usize,
    pos_id: usize,
    keys: &mut [Vec<Tensor>],
    values: &mut [Vec<Tensor>],
    p: &GptParams,
) -> Result<Tensor> {
    let mut x_vec = Vec::with_capacity(N_EMBD);
    for j in 0..N_EMBD {
        x_vec.push(&p.wte[(token_id, j)] + &p.wpe[(pos_id, j)]);
    }
    let mut x = Tensor::from_vec(x_vec, vec![N_EMBD])?;
    x = rmsnorm(&x)?;

    for li in 0..N_LAYER {
        let layer = &p.layers[li];
        let x_residual = x.clone();
        x = rmsnorm(&x)?;
        let q = linear_1d(&x, &layer.attn_wq)?;
        let k = linear_1d(&x, &layer.attn_wk)?;
        let v = linear_1d(&x, &layer.attn_wv)?;
        keys[li].push(k.clone());
        values[li].push(v.clone());

        let mut x_attn = Vec::with_capacity(N_EMBD);
        for h in 0..N_HEAD {
            let hs = h * HEAD_DIM;
            let k_len = keys[li].len();
            let mut attn_logits = Vec::with_capacity(k_len);
            for t in 0..k_len {
                let k_row = &keys[li][t];
                let mut dot = Scalar::from_f64(0.0);
                for j in 0..HEAD_DIM {
                    dot = &dot + &(&q[(hs + j)] * &k_row[(hs + j)]);
                }
                let inv_scale = Scalar::from_f64(1.0 / (HEAD_DIM as f64).sqrt());
                attn_logits.push(&dot * &inv_scale);
            }
            let attn_weights = softmax(&attn_logits);
            for j in 0..HEAD_DIM {
                let mut head_out = Scalar::from_f64(0.0);
                for t in 0..k_len {
                    let v_row = &values[li][t];
                    head_out = &head_out + &(&attn_weights[t] * &v_row[(hs + j)]);
                }
                x_attn.push(head_out);
            }
        }
        x = linear_1d(&Tensor::from_vec(x_attn, vec![N_EMBD])?, &layer.attn_wo)?;
        x = Tensor::add(&x, &x_residual)?;

        let x_residual = x.clone();
        x = rmsnorm(&x)?;
        x = linear_1d(&x, &layer.mlp_fc1)?;
        let mut relu_vec = Vec::with_capacity(4 * N_EMBD);
        for i in 0..(4 * N_EMBD) {
            relu_vec.push(x[i].relu());
        }
        x = Tensor::from_vec(relu_vec, vec![4 * N_EMBD])?;
        x = linear_1d(&x, &layer.mlp_fc2)?;
        x = Tensor::add(&x, &x_residual)?;
    }

    linear_1d(&x, &p.lm_head)
}

fn logits_to_softmax_probs(logits: &Tensor, vocab_size: usize) -> Vec<Scalar> {
    let mut v = Vec::with_capacity(vocab_size);
    for i in 0..vocab_size {
        v.push(logits[i].clone());
    }
    softmax(&v)
}

fn adam_update(p: &GptParams, m: &mut [f64], v: &mut [f64], step: usize, lr_t: f64) {
    let mut i = 0usize;
    let mut visit = |s: &Scalar| {
        let g = s.grad();
        m[i] = BETA1 * m[i] + (1.0 - BETA1) * g;
        v[i] = BETA2 * v[i] + (1.0 - BETA2) * g * g;
        let t = (step + 1) as f64;
        let m_hat = m[i] / (1.0 - BETA1.powf(t));
        let v_hat = v[i] / (1.0 - BETA2.powf(t));
        let delta = -lr_t * m_hat / (v_hat.sqrt() + EPS_ADAM);
        s.apply(|v| v + delta);
        s.zero_grad();
        i += 1;
    };
    p.wte.for_each(&mut visit);
    p.wpe.for_each(&mut visit);
    p.lm_head.for_each(&mut visit);
    for layer in &p.layers {
        layer.attn_wq.for_each(&mut visit);
        layer.attn_wk.for_each(&mut visit);
        layer.attn_wv.for_each(&mut visit);
        layer.attn_wo.for_each(&mut visit);
        layer.mlp_fc1.for_each(&mut visit);
        layer.mlp_fc2.for_each(&mut visit);
    }
}

fn main() -> Result<()> {
    let raw = fs::read_to_string("inputs.txt")?;
    let mut docs: Vec<String> = raw
        .lines()
        .filter_map(|l| {
            let t = l.trim();
            if t.is_empty() {
                None
            } else {
                Some(t.to_string())
            }
        })
        .collect();

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    docs.shuffle(&mut rng);
    println!("num docs: {}", docs.len());

    let uchars: Vec<char> = {
        let mut set: Vec<char> = docs.iter().flat_map(|s| s.chars()).collect();
        set.sort_unstable();
        set.dedup();
        set
    };
    let bos = uchars.len();
    let vocab_size = uchars.len() + 1;
    println!("vocab size: {vocab_size}");

    let params = GptParams::new(vocab_size)?;
    let n_params = params.count_scalars();
    println!("num params: {n_params}");

    let mut m = vec![0.0f64; n_params];
    let mut v = vec![0.0f64; n_params];

    for step in 0..NUM_STEPS {
        let doc = &docs[step % docs.len()];
        let mut tokens = vec![bos];
        for ch in doc.chars() {
            let idx = uchars
                .binary_search(&ch)
                .expect("char in doc must be in vocab");
            tokens.push(idx);
        }
        tokens.push(bos);

        let n = (tokens.len() - 1).min(BLOCK_SIZE);
        let mut keys: Vec<Vec<Tensor>> = (0..N_LAYER).map(|_| Vec::new()).collect();
        let mut values: Vec<Vec<Tensor>> = (0..N_LAYER).map(|_| Vec::new()).collect();

        let mut losses: Vec<Scalar> = Vec::with_capacity(n);
        for pos_id in 0..n {
            let token_id = tokens[pos_id];
            let target_id = tokens[pos_id + 1];
            let logits = gpt(token_id, pos_id, &mut keys, &mut values, &params)?;
            let probs = logits_to_softmax_probs(&logits, vocab_size);
            let loss_t = -&probs[target_id].log();
            losses.push(loss_t);
        }

        let mut sum_loss = Scalar::from_f64(0.0);
        for lt in &losses {
            sum_loss = &sum_loss + lt;
        }
        let inv_n = Scalar::from_f64(1.0 / n as f64);
        let loss = &sum_loss * &inv_n;

        loss.backward();
        let lr_t = LEARNING_RATE * (1.0 - step as f64 / NUM_STEPS as f64);
        adam_update(&params, &mut m, &mut v, step, lr_t);

        print!(
            "step {:4} / {:4} | loss {:.4}\r",
            step + 1,
            NUM_STEPS,
            loss.data()
        );
    }

    println!("\n--- inference (new, hallucinated names) ---");
    let mut rng_inf = rand::rngs::StdRng::seed_from_u64(43);
    for sample_idx in 0..20 {
        let mut keys: Vec<Vec<Tensor>> = (0..N_LAYER).map(|_| Vec::new()).collect();
        let mut values: Vec<Vec<Tensor>> = (0..N_LAYER).map(|_| Vec::new()).collect();
        let mut token_id = bos;
        let mut sample = String::new();
        for pos_id in 0..BLOCK_SIZE {
            let logits = gpt(token_id, pos_id, &mut keys, &mut values, &params)?;
            let inv_t = Scalar::from_f64(1.0 / TEMPERATURE);
            let mut scaled = Vec::with_capacity(vocab_size);
            for i in 0..vocab_size {
                scaled.push(&logits[i] * &inv_t);
            }
            let probs = softmax(&scaled);
            let r: f64 = rng_inf.random();
            let mut cum = 0.0;
            let mut next = vocab_size - 1;
            for i in 0..vocab_size {
                cum += probs[i].data();
                if r < cum {
                    next = i;
                    break;
                }
            }
            token_id = next;
            if token_id == bos {
                break;
            }
            sample.push(uchars[token_id]);
        }
        println!("sample {:2}: {}", sample_idx + 1, sample);
    }

    Ok(())
}
