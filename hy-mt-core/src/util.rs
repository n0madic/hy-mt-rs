//! Small leaf-level helpers shared across modules.

use candle_core::{DType, Tensor};

use crate::Result;

/// Cast a tensor to `dtype` only if its current dtype differs. The
/// `if needed { to_dtype } else { keep }` pattern is repeated in many
/// places (linear projection in/out, transformer load, safetensors loader);
/// keeping a single helper avoids drift.
#[inline]
pub fn cast_if(t: Tensor, dtype: DType) -> Result<Tensor> {
    if t.dtype() == dtype {
        Ok(t)
    } else {
        Ok(t.to_dtype(dtype)?)
    }
}
