use candle_core::{Tensor, Device};

pub fn apply_rope(x: &Tensor, sin: &Tensor, cos: &Tensor) -> candle_core::Result<Tensor> {
    // println!("hello world");
    // let out = Tensor::new(1.0, &Device::Cpu)?;
    // let out = x + sin + cos;
    let dims = x.dims();
    let head_dim = dims[3];
    let seq_len = dims[2];

    assert_eq!(head_dim % 2, 0, "Head dimension must be even");

    let half_dim = head_dim / 2;
    let x1 = x.narrow(3, 0, half_dim);
    let x2 = x.narrow(3, half_dim, half_dim);

    let cos = cos.narrow(0, 0, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
    let sin = cos.narrow(0, 0, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;

    let neg_x2 = x2?.neg()?;

    let rotated = Tensor::cat(&[&neg_x2, &x1?], 3)?;
    let cos = x.broadcast_mul(&cos)?;
    let rotated_sin = rotated.broadcast_mul(&sin)?;
    let x_rotated = cos.add(&rotated_sin)?;

    x_rotated.to_dtype(x.dtype())
}