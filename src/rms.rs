// class RMSNorm(nn.Module):
//     def __init__(self, emb_dim, eps=1e-6, bias=False):
//         super().__init__()
//         self.eps = eps
//         # Gemma3 stores zero-centered weights and uses (1 + weight) during forward
//         self.scale = nn.Parameter(torch.zeros(emb_dim))
//         self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

//     def forward(self, x):
//         # Match HF Gemma3: compute norm in float32, then scale by (1 + w)
//         input_dtype = x.dtype
//         x_f = x.float()
//         var = x_f.pow(2).mean(dim=-1, keepdim=True)
//         x_norm = x_f * torch.rsqrt(var + self.eps)
//         out = x_norm * (1.0 + self.scale.float())
         
//         if self.shift is not None:
//             out = out + self.shift.float()
         
//         return out.to(input_dtype)


use candle_core::{Device, Tensor, DType, Module};
use candle_nn::{Linear, VarBuilder};

pub struct RMSNorm {
    eps: f64,
    scale: f64,
    shift: Linear,
}

impl FeedForward {
    pub fn new(
        emb_dim: usize,
        hidden_dim: usize,
        dtype: DType,
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