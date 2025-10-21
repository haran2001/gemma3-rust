mod nn;

use nn::{FeedForward};
use candle_core::{Device};
use candle_core::DType;
use candle_core::Tensor;


use candle_nn::{VarBuilder, VarMap};


fn main() -> Result<(), Box<dyn std::error::Error>> {
    
    // Create VarBuilder
    let device = Device::Cpu;
    let varmap = VarMap::new();

    let emb_dim = 1;
    let hidden_dim = 1;
    let batch_size = 1;
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model = FeedForward::new(emb_dim, hidden_dim, vb)?;

    // let input = Tensor::new();
    let input = Tensor::randn(0f32, 1f32, (batch_size, emb_dim), &device)?;
    let output = model.forward(&input)?;

    println!("{:?}", output);
    Ok(())
}
