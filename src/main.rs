mod nn;
mod rms;
mod compute_rope_params;
mod apply_rope;

use nn::{FeedForward};
use rms::RMSNorm;
use compute_rope_params::compute_rope_params;
use apply_rope::apply_rope;

use candle_core::{Device};
use candle_core::DType;
use candle_core::Tensor;


use candle_nn::{VarBuilder, VarMap};


fn main() -> Result<(), Box<dyn std::error::Error>> {
    
    // Create VarBuilder
    let device = Device::Cpu;
    let varmap = VarMap::new();

    //Model layer
    let emb_dim = 1;
    let hidden_dim = 1;
    let batch_size = 1;

    //RMS norm layer
    let eps = 1e-5;
    let bias = true;

    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model = FeedForward::new(emb_dim, hidden_dim, vb.pp("feedforward"))?;
    let rmsnorm = RMSNorm::new(emb_dim, eps, bias, vb.pp("rmsnorm"))?;

    // let input = Tensor::new();
    let input = Tensor::randn(0f32, 1f32, (batch_size, emb_dim), &device)?;
    let output_fwd = model.forward(&input)?;
    let output_rms = rmsnorm.forward(&output_fwd);

    println!("{:?}", input);
    println!("{:?}", output_fwd);
    println!("{:?}", output_rms);
    
    Ok(())
}
