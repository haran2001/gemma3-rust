use candle_core::{Tensor, Device, DType};
// Imports necessary types from candle_core (a Rust tensor library similar to PyTorch)
// - Tensor: multi-dimensional array for computations
// - Device: specifies CPU/GPU for computation
// - DType: data type (f32, f64, etc.)

pub fn compute_rope_params(
    head_dim: usize,        // Dimension of each attention head (e.g., 64, 128)
    theta_base: f64,        // Base frequency for RoPE (typically 10000.0)
    context_length: usize,  // Maximum sequence length to precompute for
    dtype: DType,           // Data type for tensors
    device: &Device,        // Computation device (CPU/GPU)
) -> candle_core::Result<(Tensor, Tensor)> {
    // Returns a Result containing cosine and sine matrices for RoPE

    assert_eq!(head_dim % 2, 0, "Embedding dimension must be even");
    // RoPE requires even dimensions since it applies rotations to pairs of dimensions

    // Compute the inverse frequencies
    // inv_freq = 1.0 / (theta_base ** (arange(0, head_dim, 2) / head_dim))
    let half_dim = head_dim / 2;
    // We only need half the dimensions since we process pairs

    let mut inv_freq_vec = Vec::with_capacity(half_dim);
    // Pre-allocate vector for efficiency

    for i in 0..half_dim {
        let exponent = (2 * i) as f64 / head_dim as f64;
        // Creates exponents: 0/head_dim, 2/head_dim, 4/head_dim, ...
        
        inv_freq_vec.push(1.0 / theta_base.powf(exponent));
        // Computes inverse frequencies: 1/(theta^0), 1/(theta^(2/d)), 1/(theta^(4/d)), ...
        // Lower dimensions get higher frequencies, higher dimensions get lower frequencies
    }

    let inv_freq = Tensor::from_vec(inv_freq_vec, half_dim, device)?.to_dtype(dtype)?;
    // Convert the vector to a tensor with shape (head_dim/2,) on the specified device

    // Generate position indices
    let positions: Vec<f32> = (0..context_length).map(|i| i as f32).collect();
    // Creates [0.0, 1.0, 2.0, ..., context_length-1.0]

    let positions = Tensor::from_vec(positions, context_length, device)?.to_dtype(dtype)?;
    // Convert to tensor with shape (context_length,)

    // Compute the angles: positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    // Shape: (context_length, head_dim // 2)
    let positions = positions.unsqueeze(1)?;  // (context_length, 1)
    // Adds a dimension: [0, 1, 2, ...] becomes [[0], [1], [2], ...]

    let inv_freq = inv_freq.unsqueeze(0)?;    // (1, head_dim // 2)
    // Adds a dimension: [f1, f2, ...] becomes [[f1, f2, ...]]

    let angles = positions.broadcast_mul(&inv_freq)?;  // (context_length, head_dim // 2)
    // Broadcasting multiplication creates a matrix where:
    // angles[i][j] = position[i] * inv_freq[j]
    // This gives us the rotation angles for each position and frequency

    // Expand angles to match the head_dim by concatenating with itself
    // Shape: (context_length, head_dim)
    let angles = Tensor::cat(&[&angles, &angles], 1)?;
    // Duplicates the angles along dimension 1 because RoPE applies the same
    // rotation to pairs of dimensions (e.g., dims 0-1 use angle[0], dims 2-3 use angle[1])

    // Precompute sine and cosine
    let cos = angles.cos()?;
    // Computes cosine of all angles for rotation matrix

    let sin = angles.sin()?;
    // Computes sine of all angles for rotation matrix

    Ok((cos, sin))
    // Returns the precomputed cos and sin matrices that will be used to
    // apply rotary embeddings to query/key vectors during attention
}