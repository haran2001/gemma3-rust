use candle_core::{Tensor, Module};
use candle_nn::{Linear, VarBuilder};

#[derive(Debug)]
pub struct FeedForward {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl FeedForward {
    pub fn new(
        emb_dim: usize,
        hidden_dim: usize,
        // dtype: DType,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let fc1 = candle_nn::linear_no_bias(emb_dim, hidden_dim, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear_no_bias(emb_dim, hidden_dim, vb.pp("fc2"))?;
        let fc3 = candle_nn::linear_no_bias(hidden_dim, emb_dim, vb.pp("fc3"))?;
        
        Ok(Self { fc1, fc2, fc3 })
    }
    
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x_fc1 = self.fc1.forward(x)?;
        let x_fc2 = self.fc2.forward(x)?;
        
        // GELU activation with tanh approximation
        let x_gelu = x_fc1.gelu_erf()?;  // Note: candle uses erf by default
        // For tanh approximation, you'd need: x_fc1.gelu()?
        
        // Element-wise multiplication (gating)
        let gated = x_gelu.mul(&x_fc2)?;
        
        // Final projection
        self.fc3.forward(&gated)
    }
}