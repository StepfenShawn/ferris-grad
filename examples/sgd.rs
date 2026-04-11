use anyhow::{Ok, Result};
use ferris_grad::{Block, Linear, Sequential, Tensor, nn::Module};

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

    let mut network = Sequential::new(vec![
        Block::Linear(Linear::new(3, 4)),
        Block::Relu,
        Block::Linear(Linear::new(4, 4)),
        Block::Relu,
        Block::Linear(Linear::new(4, 1)),
    ]);
    let lr = 0.00005;
    for i in 0..500 {
        let diff = &network.forward(&training_inputs) - &target_outputs;
        let loss = diff.mul(&diff)?.sum();
        println!("step {} | loss: {:?}", i + 1, loss.data());
        loss.backward();
        network.parameters().iter().for_each(|p| {
            p.for_each(|s| {
                s.adjust(-lr);
                s.zero_grad();
            });
        });
    }

    let input = Tensor::from_vec(
        vec![1.0, 1.0, 1.0].iter().map(|x| (*x).into()).collect(),
        [1, 3].into(),
    )?;
    println!("{}", network.forward(&input));
    Ok(())
}
