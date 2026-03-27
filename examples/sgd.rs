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
        .collect::<Vec<Scalar>>(),
        [4, 3].into(),
    )?;

    let target_outputs = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 2.0]
            .iter()
            .map(|x| (*x).into())
            .collect::<Vec<Scalar>>(),
        [1, 4].into(),
    )?;

    let network = MLP::new(vec![Linear::new(3, 4)]);
    println!("{:?}", training_inputs);
    println!("{:?}", network);

    for i in 0..100 {
        let loss = (&network.forward(&training_inputs) - &target_outputs).sum();
        loss.backward();
        println!("Loss: {:?}", loss.data());
        // for param in network.parameters() {
        //     let grad = param.grad() * 0.01;
        //     param.data = param.data - grad;
        // }
    }
    Ok(())
}
