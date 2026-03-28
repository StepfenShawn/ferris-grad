# ferris_grad
Pytorch-like autograd engine in Rust.    

# Examples
tensor api:  
```rust
use anyhow::Result;
use ferris_grad::Tensor;

fn main() -> Result<()> {
    let a = Tensor::from_vec(vec![1.0.into(), 2.0.into(), 3.0.into()], [3, 1].into())?;
    let b = Tensor::rand([3, 1].into())?;
    let c = &a * &b;
    println!("{:?}", c);
    c.get((1, 0)).backward();
    println!("{:?}", a.get((1, 0)).grad());
    Ok(())
}
```
sgd implement in ferrs_grad:    
```rust
use anyhow::{Ok, Result};
use ferris_grad::{Linear, MLP, Tensor, nn::Module, scalar::Scalar};

fn main() -> Result<()> {
    let training_inputs = Tensor::from_vec(
        vec![
            vec![2.0, 3.0, 11.0],
            vec![30.0, 1.0, 0.5],
            vec![5.5, 1.0, 6.0],
            vec![11.0, 1.0, 1.0],
        ]
        .iter()
        .flatten()
        .map(|x| (*x).into())
        .collect(),
        [4, 3].into(),
    )?;

    let target_outputs = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 2.0]
            .iter()
            .map(|x| (*x).into())
            .collect(),
        [4, 1].into(),
    )?;

    let mut network = MLP::new(vec![
        Linear::new(3, 4),
        Linear::new(4, 4),
        Linear::new(4, 1),
    ]);
    let lr = 0.00005;
    for i in 0..500 {
        let diff = &network.forward(&training_inputs) - &target_outputs;
        let loss = diff.mul(&diff)?.sum();
        println!("Step {}: Loss: {:?}", i + 1, loss.data());

        network.parameters().iter().for_each(|p| {
            p.for_each(|s| s.zero_grad());
        });

        loss.backward();
        network.parameters().iter().for_each(|p| {
            p.for_each(|s| s.adjust(-lr));
        });
    }

    let input = Tensor::from_vec(
        vec![1.0, 1.0, 1.0].iter().map(|x| (*x).into()).collect(),
        [1, 3].into(),
    )?;
    println!("{}", network.forward(&input));
    Ok(())
}

```