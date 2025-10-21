use candle_core::{Tensor, DType};
use candle_nn::VarBuilder;

pub struct RMSNorm {
    eps: f64,
    scale: Tensor,
    shift: Option<Tensor>,
}

impl RMSNorm {
    pub fn new(
        emb_dim: usize,
        eps: f64,
        bias: bool,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        // Gemma3 stores zero-centered weights and uses (1 + weight) during forward
        let scale = vb.get(emb_dim, "scale")?;
        let shift = if bias {
            Some(vb.get(emb_dim, "shift")?)
        } else {
            None
        };

        Ok(Self { eps, scale, shift })
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // Match HF Gemma3: compute norm in float32, then scale by (1 + w)
        let input_dtype = x.dtype();
        let x_f = x.to_dtype(DType::F32)?;

        // Compute variance: mean of squared values along last dimension
        let var = x_f.sqr()?.mean_keepdim(x_f.dims().len() - 1)?;

        // Normalize: x / sqrt(var + eps)
        let x_norm = x_f.broadcast_div(&(var + self.eps)?.sqrt()?)?;

        // Scale by (1 + scale)
        let scale_f32 = self.scale.to_dtype(DType::F32)?;
        let one_plus_scale = (scale_f32 + 1.0)?;
        let mut out = x_norm.broadcast_mul(&one_plus_scale)?;

        // Add shift if bias is enabled
        if let Some(shift) = &self.shift {
            let shift_f32 = shift.to_dtype(DType::F32)?;
            out = out.broadcast_add(&shift_f32)?;
        }

        // Convert back to original dtype
        out.to_dtype(input_dtype)
    }
}