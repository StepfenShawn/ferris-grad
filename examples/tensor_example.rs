use anyhow::Result;
use ferris_grad::Tensor;

fn main() -> Result<()> {
    let a = Tensor::from_vec(vec![1.0.into(), 2.0.into(), 3.0.into()], [3, 1].into())?;
    let b = Tensor::rand([3, 1].into())?;
    let c = &a * &b;
    println!("{}", c);
    c[(1, 0)].backward();
    println!("{}", a[(1, 0)].grad());

    let d = Tensor::stack(vec![a.clone(), a.clone()], 0)?;
    println!("{}", d);
    Ok(())
}
