mod engine;
use anyhow::Result;
use engine::Tensor;

fn main() -> Result<()> {
    let a = Tensor::from_vec([1.0, 2.0, 3.0].into(), [3].into(), true)?;
    println!("{:?}", a);
    println!("shape: {:?}", a.shape());
    println!("ndim: {:?}", a.ndim());
    println!("size: {:?}", a.size());
    let b = Tensor::zeros([3, 3].into(), true)?;
    println!("{:?}", b);
    let c = Tensor::ones([3, 3].into(), true)?;
    println!("{:?}", c);
    Ok(())
}
