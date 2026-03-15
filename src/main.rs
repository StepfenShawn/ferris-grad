mod engine;
use anyhow::Result;
use engine::Tensor;

fn main() -> Result<()> {
    let a = Tensor::from_vec([1.0, 2.0, 3.0].into(), [3, 1].into())?;
    let b = Tensor::zeros([3, 1].into())?;
    let c = a.clone() + b.clone();
    let _ = c.backward();
    println!("{:?} ---------\n {:?}", a.grad(), b.grad());
    Ok(())
}
