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
        .into_iter()
        .flatten()
        .map(|x| Scalar::from_f64(*x))
        .collect::<Vec<Scalar>>(),
        [4, 3].into(),
    )?;

    let target_outputs = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 2.0]
            .iter()
            .map(|x| Scalar::from_f64(*x))
            .collect::<Vec<Scalar>>(),
        [1, 4].into(),
    )?;

    let network = MLP::new(vec![Linear::new(3, 4)]);
    println!("{:?}", training_inputs);
    println!("{:?}", network);
    for _ in 0..100 {
        println!("{:?}", network.forward(&training_inputs));
    }
    Ok(())
}
