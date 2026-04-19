# ferris_grad
A PyTorch-like autograd engine in under 1000 lines of Rust code.🦀    

# How ferris_grad works?
ferris_grad is just build in 3 files:
| file | description |
| --- | --- |
| `scalar.rs` | implement the scalar data structure, compute graph, backward rule, and basic operations. |
| `tensor.rs` | implement the tensor data structure and basic operations. |
| `nn.rs` | implement the neural network layers and modules. |

# How to use ferris_grad?
Add this to your `Cargo.toml`:
```toml
[dependencies]
ferris_grad = "*"
```
and then you can use `ferris_grad` in your project:
```rust
use anyhow::Result;
use ferris_grad::{Tensor, nn::Module};

fn main() -> Result<()> {
    let a = Tensor::from_vec(vec![1.0.into(), 2.0.into(), 3.0.into()], [3, 1].into())?;
    let b = Tensor::rand([3, 1].into())?;
    let c = &a * &b;
    println!("{}", c);
    Ok(())
}
```

# Examples
All the examples are in the `examples` folder.
| file | description |
| --- | --- |
| [`examples/tensor_example.rs`](examples/tensor_example.rs) | basic usage of tensor api. |
| [`examples/backward_scalar.rs`](examples/backward_scalar.rs) | basic usage of backward api. |
| [`examples/sgd.rs`](examples/sgd.rs) | training a MLP example using sgd. |
| [`examples/sgd_with_plotter.rs`](examples/sgd_with_plotter.rs) | training a MLP example using sgd with plotter. |
| [`examples/rnn.rs`](examples/rnn.rs) | training a recurrent neural network in ferris_grad. |
| [`examples/gpt.rs`](examples/gpt.rs) | training a mini GPT in ferris_grad! |

# License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.