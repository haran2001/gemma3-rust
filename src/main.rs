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
    //emb_dim = num_heads * seq_len * head_dim
    let emb_dim = 64;  // e.g., 2 * 4 * 8 = 64
    let (num_heads, seq_len) = (2, 4);
    
    let hidden_dim = 1;
    let batch_size = 1;

    //RMS norm layer
    let eps = 1e-5;
    let bias = true;


    // RoPE params
    let (head_dim, theta_base, context_length) = (8, 10000.0, 32);

    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model = FeedForward::new(emb_dim, hidden_dim, vb.pp("feedforward"))?;
    let rmsnorm = RMSNorm::new(emb_dim, eps, bias, vb.pp("rmsnorm"))?;
    
    // Feedforward through NN and rmsnorm
    let input = Tensor::randn(0f32, 1f32, (batch_size, emb_dim), &device)?;
    let output_fwd = model.forward(&input)?;
    let output_rms = rmsnorm.forward(&output_fwd)?;
    let output_4d = output_rms.reshape((batch_size, num_heads, seq_len, head_dim))?;

    //compute and apply RoPE
    let (cos, sin) = compute_rope_params(head_dim, theta_base, context_length, DType::F32, &device)?;
    let x_rope = apply_rope(&output_4d, &sin, &cos)?;
    

    println!("{:?}", x_rope);
    // println!("{:?}", output_fwd);
    // println!("{:?}", output_rms);
    
    Ok(())
}
