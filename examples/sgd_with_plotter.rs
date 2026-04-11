use anyhow::{Context, Ok, Result};
use ferris_grad::{Block, Linear, Sequential, Tensor, nn::Module};
use plotters::prelude::*;
use std::time::{SystemTime, UNIX_EPOCH};

fn plot_loss_curve(path: &str, losses: &[f64]) -> Result<()> {
    if losses.is_empty() {
        return Ok(());
    }
    let root = BitMapBackend::new(path, (900, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let n = losses.len();
    let x_max = if n <= 1 { 1.0 } else { (n - 1) as f32 };
    let mut min_y = losses.iter().copied().fold(f64::INFINITY, f64::min);
    let mut max_y = losses.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if (max_y - min_y).abs() < 1e-12 {
        max_y = min_y + 1.0;
        min_y -= 1.0;
    } else {
        let pad = (max_y - min_y) * 0.05;
        min_y -= pad;
        max_y += pad;
    }

    let mut chart = ChartBuilder::on(&root)
        .caption("sgd loss:", ("sans-serif", 30).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0f32..x_max, min_y..max_y)?;

    chart
        .configure_mesh()
        .x_desc("step")
        .y_desc("loss")
        .draw()?;

    chart.draw_series(LineSeries::new(
        losses.iter().enumerate().map(|(i, &v)| (i as f32, v)),
        &BLUE,
    ))?;

    root.present()?;
    Ok(())
}

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
    let mut losses = Vec::with_capacity(500);
    for i in 0..100 {
        let diff = &network.forward(&training_inputs) - &target_outputs;
        let loss = diff.mul(&diff)?.mean();
        losses.push(loss.data());
        println!("step {} | loss: {:?}", i + 1, loss.data());
        loss.backward();
        network.parameters().iter().for_each(|p| {
            p.for_each(|s| {
                s.adjust(-lr);
                s.zero_grad();
            });
        });
    }

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp_png = std::env::temp_dir().join(format!("ferris_sgd_loss_{nanos}.png"));
    let tmp_str = tmp_png.to_str().context(
        "The temporary directory path must be UTF-8 (required for plotters to write PNG files).",
    )?;

    plot_loss_curve(tmp_str, &losses)?;
    open::that(&tmp_png).context("Cannot open the image with the system default program")?;

    let input = Tensor::from_vec(
        vec![1.0, 1.0, 1.0].iter().map(|x| (*x).into()).collect(),
        [1, 3].into(),
    )?;
    println!("{}", network.forward(&input));
    Ok(())
}
