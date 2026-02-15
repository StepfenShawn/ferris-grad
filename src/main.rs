mod engine;
use anyhow::Result;
use engine::Tensor;

fn main() -> Result<()> {
    let a = Tensor::from_vec([1.0, 2.0, 3.0].into(), [3].into(), true)?;
    println!("{:?}", a);
    Ok(())
}
