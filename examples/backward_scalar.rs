use anyhow::{Ok, Result};
use ferris_grad::scalar::Scalar;

fn main() -> Result<()> {
    let a = Scalar::from_f64(2.);
    let b = Scalar::from_f64(3.);
    let c = &a * &b;
    let l = &c + &a;
    l.backward();
    println!("{}", a.grad());
    println!("{}", b.grad());
    Ok(())
}