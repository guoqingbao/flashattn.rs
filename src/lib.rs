mod ffi;
use candle::cuda_backend::WrapErr;
use candle::{CpuStorage, Layout, Result, Shape, Tensor};
use candle_core as candle;
use candle_core::backend::BackendStorage;
use half::{bf16, f16};
#[cfg(feature = "flash-decoding")]
use once_cell::sync::Lazy;
#[cfg(feature = "flash-decoding")]
use std::collections::HashMap;
#[cfg(feature = "flash-decoding")]
use std::sync::Mutex;

pub struct FlashAttn {
    pub softmax_scale: f32,
    pub window_size_left: Option<usize>,
    pub window_size_right: Option<usize>,
    pub softcap: Option<f32>,
}

impl FlashAttn {
    fn cuda_fwd_t<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
        OutT: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        k: &candle::CudaStorage,
        k_l: &Layout,
        v: &candle::CudaStorage,
        v_l: &Layout,
        is_bf16: bool,
        is_e4m3: bool,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        let dev = q.device();

        let q = q.as_cuda_slice::<T>()?;
        let k = k.as_cuda_slice::<T>()?;
        let v = v.as_cuda_slice::<T>()?;
        let q = q.slice(q_l.start_offset()..);
        let k = k.slice(k_l.start_offset()..);
        let v = v.slice(v_l.start_offset()..);

        let q_stride = q_l.stride();
        let k_stride = k_l.stride();
        let v_stride = v_l.stride();

        let q_rank = q_stride.len();
        let k_rank = k_stride.len();
        let v_rank = v_stride.len();

        if q_rank != 4 || k_rank != 4 || v_rank != 4 {
            candle::bail!(
                "flash-attn expects input tensors of rank 4 (q: {q_rank}, k: {k_rank}, v: {v_rank}"
            )
        }
        if q_stride[q_rank - 1] != 1 {
            candle::bail!("the last dim of q must be contiguous {q_stride:?}")
        }
        if k_stride[k_rank - 1] != 1 {
            candle::bail!("the last dim of k must be contiguous {k_stride:?}")
        }
        if v_stride[v_rank - 1] != 1 {
            candle::bail!("the last dim of v must be contiguous {v_stride:?}")
        }

        let (b_sz, seqlen_q, num_heads, head_size_og) = q_l.shape().dims4()?;
        let (b_k, seqlen_k, num_heads_k, head_size_k) = k_l.shape().dims4()?;
        let (b_v, seqlen_kv, num_heads_kv, head_size_v) = v_l.shape().dims4()?;
        if b_k != b_sz {
            candle::bail!(
                "batch size mismatch q {:?} and k {:?}",
                q_l.shape(),
                k_l.shape()
            )
        }
        if b_k != b_v || seqlen_k != seqlen_kv || num_heads_k != num_heads_kv {
            candle::bail!("shape mismatch k {:?} and v {:?}", k_l.shape(), v_l.shape())
        }
        if head_size_k != head_size_og {
            candle::bail!(
                "head size mismatch q {:?} and k {:?}",
                q_l.shape(),
                k_l.shape()
            )
        }
        if head_size_og > 256 {
            candle::bail!("only supports head dimension at most 256 (got {head_size_og})")
        }
        if head_size_og % 8 != 0 {
            // TODO: Handle head sizes that are not a multiple of 8 via some padding.
            candle::bail!("only supports head sizes that are a multiple of 8 (got {head_size_og})")
        }
        if num_heads % num_heads_k != 0 {
            candle::bail!("number of k/v heads {num_heads_k} must divide number of heads in query {num_heads}")
        }
        if head_size_v != head_size_og {
            let valid = (head_size_og > 128
                && head_size_og <= 192
                && head_size_v > 96
                && head_size_v <= 128)
                || (head_size_og <= 64 && head_size_v <= 512);
            if !valid {
                candle::bail!("unsupported v head dim {head_size_v} for q head dim {head_size_og}")
            }
        }
        if head_size_v != head_size_og {
            let valid = (head_size_og > 128
                && head_size_og <= 192
                && head_size_v > 96
                && head_size_v <= 128)
                || (head_size_og <= 64 && head_size_v <= 512);
            if !valid {
                candle::bail!("unsupported v head dim {head_size_v} for q head dim {head_size_og}")
            }
        }

        // if window_size_left > self.max_seqlen_k or None => -1
        let mut window_size_left = self
            .window_size_left
            .filter(|v| v <= &seqlen_k)
            .map(|v| v as i32)
            .unwrap_or(-1);

        // if window_size_right > self.max_seqlen_k or None => -1
        let mut window_size_right = self
            .window_size_right
            .filter(|v| v <= &seqlen_k)
            .map(|v| v as i32)
            .unwrap_or(-1);

        let out_shape = Shape::from((b_sz, seqlen_q, num_heads, head_size_v));
        let out_l = Layout::contiguous(&out_shape);
        let o_stride = out_l.stride();
        let o_rank = o_stride.len();

        let elem_count = out_shape.elem_count();
        let dst = unsafe { dev.alloc::<OutT>(elem_count) }.w()?;
        let softmax_lse = dev
            .alloc_zeros::<f32>(b_sz * num_heads * seqlen_q, false)
            .w()?;

        let is_bf16 = if is_bf16 { 1 } else { 0 };
        let is_e4m3 = if is_e4m3 { 1 } else { 0 };

        if window_size_left < 0 && window_size_right >= 0 {
            window_size_left = seqlen_k as i32;
        }
        if window_size_left >= 0 && window_size_right < 0 {
            window_size_right = seqlen_k as i32;
        }

        unsafe {
            let q_ptr = *q.device_ptr() as *const core::ffi::c_void;
            let k_ptr = *k.device_ptr() as *const core::ffi::c_void;
            let v_ptr = *v.device_ptr() as *const core::ffi::c_void;
            let dst_ptr = *dst.device_ptr() as *mut core::ffi::c_void;
            let softmax_lse_ptr = *softmax_lse.device_ptr() as *mut core::ffi::c_void;
            ffi::run_mha(
                q_ptr,
                k_ptr,
                v_ptr,
                std::ptr::null(),
                dst_ptr,
                softmax_lse_ptr,
                /* cu_seqlens_q_ptr */ std::ptr::null(),
                /* cu_seqlens_k_ptr */ std::ptr::null(),
                /* seqused_q_ptr */ std::ptr::null(),
                /* seqused_k_ptr */ std::ptr::null(),
                /* leftpad_k_ptr */ std::ptr::null(),
                /* kv_batch_idx_ptr */ std::ptr::null(),
                /* q_descale_ptr */ std::ptr::null(),
                /* k_descale_ptr */ std::ptr::null(),
                /* v_descale_ptr */ std::ptr::null(),
                /* q_batch_stride */ q_stride[0] as u32,
                /* k_batch_stride */ k_stride[0] as u32,
                /* v_batch_stride */ v_stride[0] as u32,
                /* o_batch_stride */ o_stride[0] as u32,
                /* q_row_stride   */ q_stride[q_rank - 3] as u32,
                /* k_row_stride   */ k_stride[k_rank - 3] as u32,
                /* v_row_stride   */ v_stride[v_rank - 3] as u32,
                /* o_row_stride   */ o_stride[o_rank - 3] as u32,
                /* q_head_stride  */ q_stride[q_rank - 2] as u32,
                /* k_head_stride  */ k_stride[k_rank - 2] as u32,
                /* v_head_stride  */ v_stride[v_rank - 2] as u32,
                /* o_head_stride  */ o_stride[o_rank - 2] as u32,
                /* v_dim_stride */ v_stride[v_rank - 1] as u32,
                /* q_descale_batch_stride */ 0,
                /* q_descale_head_stride */ 0,
                /* k_descale_batch_stride */ 0,
                /* k_descale_head_stride */ 0,
                /* v_descale_batch_stride */ 0,
                /* v_descale_head_stride */ 0,
                /* b */ b_sz as u32,
                /* b_k */ b_k as u32,
                /* h */ num_heads as u32,
                /* h_k */ num_heads_k as u32,
                /* d */ head_size_og as u32,
                /* dv */ head_size_v as u32,
                /* seqlen_q */ seqlen_q as u32,
                /* seqlen_k */ seqlen_k as u32,
                /* total_q */ (b_sz * seqlen_q) as u32,
                /* total_k */ (b_k * seqlen_k) as u32,
                /* softmax_scale*/ self.softmax_scale,
                /* is_bf16 */ is_bf16,
                /* is_e4m3 */ is_e4m3,
                /* window_size_left */ window_size_left,
                /* window_size_right */ window_size_right,
                /* attention_chunk */ 0,
                /* page_size */ 1,
                /* page_table_batch_stride */ 0u32,
                /* num_pages */ 0,
                /* num_splits */ 1,
                /* softmax_lseaccum_ptr */ std::ptr::null_mut(),
                /* oaccum_ptr */ std::ptr::null_mut(),
                /* oaccum_split_stride */ 0u32,
                /* oaccum_batch_stride */ 0u32,
                /* oaccum_row_stride */ 0u32,
                /* oaccum_head_stride */ 0u32,
                /* lseaccum_split_stride */ 0u32,
                /* lseaccum_batch_stride */ 0u32,
                /* lseaccum_head_stride */ 0u32,
                /* softcap */ self.softcap.unwrap_or(0f32),
                /* pack_gqa */ -1,
                *dev.cu_stream() as i64,
            )
        }

        let dst = candle::CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, out_shape))
    }
}

impl candle::CustomOp3 for FlashAttn {
    fn name(&self) -> &'static str {
        "flash-attn"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for flash-attn")
    }

    fn cuda_fwd(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        k: &candle::CudaStorage,
        k_l: &Layout,
        v: &candle::CudaStorage,
        v_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        match q.dtype() {
            candle::DType::F16 => self.cuda_fwd_t::<f16, f16>(q, q_l, k, k_l, v, v_l, false, false),
            candle::DType::BF16 => {
                self.cuda_fwd_t::<bf16, bf16>(q, q_l, k, k_l, v, v_l, true, false)
            }
            candle::DType::U8 => self.cuda_fwd_t::<u8, bf16>(q, q_l, k, k_l, v, v_l, true, true),
            dt => candle::bail!("flash-attn is only supported for f16/bf16/u8 ({dt:?})"),
        }
    }
}

/// Flash-attention v3 layer.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(batch, seq_len_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
///
/// The resulting tensor has dimensions `(batch, seq_len_q, num_heads_q, head_size_v)`.
pub fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let window_size_left = None;
    let window_size_right = if causal { Some(0) } else { None };

    let op = FlashAttn {
        softmax_scale,
        window_size_left,
        window_size_right,
        softcap: None,
    };
    q.apply_op3(k, v, op)
}

pub fn flash_attn_softcap(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    softcap: Option<f32>,
    causal: bool,
) -> Result<Tensor> {
    let window_size_left = None;
    let window_size_right = if causal { Some(0) } else { None };

    let op = FlashAttn {
        softmax_scale,
        window_size_left,
        window_size_right,
        softcap,
    };
    q.apply_op3(k, v, op)
}

/// Flash-attention v3 layer.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(batch, seq_len_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(batch, seq_len_kv, num_heads_kv, head_size)`.
/// * `window_size_left` - Limit left attention to value tokens.
/// * `window_size_right` - Limit right attention to value tokens.
///
/// # Causal mask
///
/// `window_size_left=None` with `window_size_right=Some(0)` applies a causal mask to the result
/// of  `Q @ K^T`
///
/// The resulting tensor has dimensions `(batch, seq_len_q, num_heads_q, head_size_v)`.
pub fn flash_attn_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor> {
    let op = FlashAttn {
        softmax_scale,
        window_size_left,
        window_size_right,
        softcap: None,
    };
    q.apply_op3(k, v, op)
}

pub fn flash_attn_windowed_softcap(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    softcap: Option<f32>,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor> {
    let op = FlashAttn {
        softmax_scale,
        window_size_left,
        window_size_right,
        softcap,
    };
    q.apply_op3(k, v, op)
}

struct FlashAttnVarLen {
    pub softmax_scale: f32,
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    pub seqlens_q: Tensor,
    pub seqlens_k: Tensor,
    pub block_table: Option<Tensor>,
    pub window_size_left: Option<usize>,
    pub window_size_right: Option<usize>,
    pub softcap: Option<f32>,
}

impl FlashAttnVarLen {
    fn cuda_fwd_t<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
        OutT: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        k: &candle::CudaStorage,
        k_l: &Layout,
        v: &candle::CudaStorage,
        v_l: &Layout,
        is_bf16: bool,
        is_e4m3: bool,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        let dev = q.device();

        let (seqlens_q, seqlens_q_layout) = self.seqlens_q.storage_and_layout();
        let seqlens_q = match &*seqlens_q {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("seqlens_q must be a cuda tensor"),
        };
        let seqlens_q = match seqlens_q_layout.contiguous_offsets() {
            Some((o1, o2)) => seqlens_q.slice(o1..o2),
            None => candle::bail!("seqlens_q has to be contiguous"),
        };

        let (seqlens_k, seqlens_k_layout) = self.seqlens_k.storage_and_layout();
        let seqlens_k = match &*seqlens_k {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("seqlens_k must be a cuda tensor"),
        };
        let seqlens_k = match seqlens_k_layout.contiguous_offsets() {
            Some((o1, o2)) => seqlens_k.slice(o1..o2),
            None => candle::bail!("seqlens_k has to be contiguous"),
        };

        let q = q.as_cuda_slice::<T>()?;
        let k = k.as_cuda_slice::<T>()?;
        let v = v.as_cuda_slice::<T>()?;
        let q = q.slice(q_l.start_offset()..);
        let k = k.slice(k_l.start_offset()..);
        let v = v.slice(v_l.start_offset()..);

        let q_stride = q_l.stride();
        let k_stride = k_l.stride();
        let v_stride = v_l.stride();

        let q_rank = q_stride.len();
        let k_rank = k_stride.len();
        let v_rank = v_stride.len();

        if q_stride[q_rank - 1] != 1 {
            candle::bail!("the last dim of q must be contiguous {q_stride:?}")
        }
        if k_stride[k_rank - 1] != 1 {
            candle::bail!("the last dim of k must be contiguous {k_stride:?}")
        }
        if v_stride[v_rank - 1] != 1 {
            candle::bail!("the last dim of v must be contiguous {v_stride:?}")
        }

        let (total_q, num_heads, head_size_og) = q_l.shape().dims3()?;

        let (
            page_block_size,
            num_heads_k,
            num_pages,
            head_size_v,
            k_row_stride,
            v_row_stride,
            k_head_stride,
            v_head_stride,
        ) = if self.block_table.is_some() {
            if q_rank != 3 || k_rank != 4 || v_rank != 4 {
                candle::bail!(
                    "flash-attn-varlen expects input tensors of rank 3,4,4 (q: {q_rank}, k: {k_rank}, v: {v_rank}"
                )
            }
            let (num_pages, page_block_size, num_heads_k, head_size_k) = k_l.shape().dims4()?;
            let (v_num_pages, v_page_block_size, v_num_heads_k, head_size_v) =
                v_l.shape().dims4()?;
            if (num_pages, page_block_size, num_heads_k)
                != (v_num_pages, v_page_block_size, v_num_heads_k)
            {
                candle::bail!("shape mismatch k {:?} and v {:?}", k_l.shape(), v_l.shape())
            }
            if head_size_k != head_size_og {
                candle::bail!(
                    "head size mismatch q {:?} and k {:?}",
                    q_l.shape(),
                    k_l.shape()
                )
            }
            // `row_stride` is the stride for the `block_size` dimension -> stride[1]
            // `head_stride` is the stride for the `num_heads_k` dimension -> stride[2]
            (
                page_block_size,
                num_heads_k,
                num_pages,
                head_size_v,
                k_stride[1],
                v_stride[1],
                k_stride[2],
                v_stride[2],
            )
        } else {
            if q_rank != 3 || k_rank != 3 || v_rank != 3 {
                candle::bail!(
                    "flash-attn-varlen expects input tensors of rank 3 (q: {q_rank}, k: {k_rank}, v: {v_rank}"
                )
            }
            let (total_k, num_heads_k, head_size_k) = k_l.shape().dims3()?;
            let (total_v, num_heads_kv, head_size_v) = v_l.shape().dims3()?;
            if (total_k, num_heads_k) != (total_v, num_heads_kv) {
                candle::bail!("shape mismatch k {:?} and v {:?}", k_l.shape(), v_l.shape())
            }
            if head_size_k != head_size_og {
                candle::bail!(
                    "head size mismatch q {:?} and k {:?}",
                    q_l.shape(),
                    k_l.shape()
                )
            }
            if num_heads % num_heads_k != 0 {
                candle::bail!("number of k/v heads {num_heads_k} must divide number of heads in query {num_heads}")
            }
            // `row_stride` is the stride for the `block_size` dimension -> stride[1]
            // `head_stride` is the stride for the `num_heads_k` dimension -> stride[2]
            (
                1,
                num_heads_k,
                0,
                head_size_v,
                k_stride[0],
                v_stride[0],
                k_stride[1],
                v_stride[1],
            )
        };

        if head_size_og > 256 {
            candle::bail!("only supports head dimension at most 256 (got {head_size_og})")
        }
        if head_size_og % 8 != 0 {
            // TODO: Handle head sizes that are not a multiple of 8 via some padding.
            candle::bail!("only supports head sizes that are a multiple of 8 (got {head_size_og})")
        }
        if head_size_v != head_size_og {
            let valid = (head_size_og > 128
                && head_size_og <= 192
                && head_size_v > 96
                && head_size_v <= 128)
                || (head_size_og <= 64 && head_size_v <= 512);
            if !valid {
                candle::bail!("unsupported v head dim {head_size_v} for q head dim {head_size_og}")
            }
        }

        let nseqlens_q = seqlens_q_layout.shape().dims1()?;
        if nseqlens_q < 2 {
            candle::bail!("seqlens_q should have a len >= 2 {nseqlens_q}")
        }
        let nseqlens_k = seqlens_k_layout.shape().dims1()?;
        if nseqlens_k != nseqlens_q {
            candle::bail!("seqlens_q and seqlens_k should have the same number of elements {nseqlens_q} <> {nseqlens_k}")
        }

        let batch_size = nseqlens_q - 1;

        let (
            block_table_ptr,
            block_table_batch_stride,
            block_table_batch_size,
            _max_num_blocks_per_seq,
        ) = if let Some(block_table) = &self.block_table {
            let max_num_blocks_per_seq = block_table.dim(1)?;
            let block_table_batch_size = block_table.dim(0)?;
            let (block_table, block_table_layout) = block_table.storage_and_layout();
            let block_table_batch_stride = block_table_layout.stride()[0];
            let block_table = match &*block_table {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
                _ => candle::bail!("block_table must be a cuda tensor"),
            };

            let block_table = block_table.slice(block_table_layout.start_offset()..);

            (
                *block_table.device_ptr() as *const core::ffi::c_void,
                block_table_batch_stride,
                block_table_batch_size,
                max_num_blocks_per_seq,
            )
        } else {
            (std::ptr::null(), 0usize, 0usize, 0usize)
        };

        // if window_size_left > self.max_seqlen_k or None => -1
        let mut window_size_left = self
            .window_size_left
            .filter(|v| v <= &self.max_seqlen_k)
            .map(|v| v as i32)
            .unwrap_or(-1);

        // if window_size_right > self.max_seqlen_k or None => -1
        let mut window_size_right = self
            .window_size_right
            .filter(|v| v <= &self.max_seqlen_k)
            .map(|v| v as i32)
            .unwrap_or(-1);

        let out_shape = Shape::from((total_q, num_heads, head_size_v));
        let out_l = Layout::contiguous(&out_shape);
        let o_stride = out_l.stride();
        let o_rank = o_stride.len();

        let elem_count = out_shape.elem_count();
        let dst = unsafe { dev.alloc::<OutT>(elem_count) }.w()?;
        let softmax_lse = dev.alloc_zeros::<f32>(num_heads * total_q, false).w()?;

        let is_bf16 = if is_bf16 { 1 } else { 0 };
        let is_e4m3 = if is_e4m3 { 1 } else { 0 };

        if window_size_left >= self.max_seqlen_k as i32 {
            window_size_left = -1;
        }
        if window_size_right >= self.max_seqlen_k as i32 {
            window_size_right = -1;
        }
        if window_size_left < 0 && window_size_right >= 0 {
            window_size_left = self.max_seqlen_k as i32;
        }
        if window_size_left >= 0 && window_size_right < 0 {
            window_size_right = self.max_seqlen_k as i32;
        }

        let (k_batch_stride, v_batch_stride, total_k) = if self.block_table.is_some() {
            (
                k_stride[0] as u32,
                v_stride[0] as u32,
                (num_pages * page_block_size) as usize,
            )
        } else {
            (0, 0, k_l.shape().dims3()?.0)
        };

        unsafe {
            let q_ptr = *q.device_ptr() as *const core::ffi::c_void;
            let k_ptr = *k.device_ptr() as *const core::ffi::c_void;
            let v_ptr = *v.device_ptr() as *const core::ffi::c_void;
            let dst_ptr = *dst.device_ptr() as *mut core::ffi::c_void;
            let softmax_lse_ptr = *softmax_lse.device_ptr() as *mut core::ffi::c_void;
            let seqlens_q_ptr = *seqlens_q.device_ptr() as *const core::ffi::c_int;
            let seqlens_k_ptr = *seqlens_k.device_ptr() as *const core::ffi::c_int;
            ffi::run_mha(
                q_ptr,
                k_ptr,
                v_ptr,
                block_table_ptr,
                dst_ptr,
                softmax_lse_ptr,
                /* cu_seqlens_q_ptr */ seqlens_q_ptr,
                /* cu_seqlens_k_ptr */ seqlens_k_ptr,
                /* seqused_q_ptr */ std::ptr::null(),
                /* seqused_k_ptr */ std::ptr::null(),
                /* leftpad_k_ptr */ std::ptr::null(),
                /* kv_batch_idx_ptr */ std::ptr::null(),
                /* q_descale_ptr */ std::ptr::null(),
                /* k_descale_ptr */ std::ptr::null(),
                /* v_descale_ptr */ std::ptr::null(),
                /* q_batch_stride */ q_stride[0] as u32,
                /* k_batch_stride */ k_batch_stride,
                /* v_batch_stride */ v_batch_stride,
                /* o_batch_stride */ o_stride[0] as u32,
                /* q_row_stride   */ q_stride[0] as u32,
                /* k_row_stride   */ k_row_stride as u32,
                /* v_row_stride   */ v_row_stride as u32,
                /* o_row_stride   */ o_stride[0] as u32,
                /* q_head_stride  */ q_stride[1] as u32,
                /* k_head_stride  */ k_head_stride as u32,
                /* v_head_stride  */ v_head_stride as u32,
                /* o_head_stride  */ o_stride[o_rank - 2] as u32,
                /* v_dim_stride */ v_stride[v_rank - 1] as u32,
                /* q_descale_batch_stride */ 0,
                /* q_descale_head_stride */ 0,
                /* k_descale_batch_stride */ 0,
                /* k_descale_head_stride */ 0,
                /* v_descale_batch_stride */ 0,
                /* v_descale_head_stride */ 0,
                /* b */ batch_size as u32,
                /* b_k */
                if self.block_table.is_some() {
                    block_table_batch_size as u32
                } else {
                    batch_size as u32
                },
                /* h */ num_heads as u32,
                /* h_k */ num_heads_k as u32,
                /* d */ head_size_og as u32,
                /* dv */ head_size_v as u32,
                /* seqlen_q */ self.max_seqlen_q as u32,
                /* seqlen_k */ self.max_seqlen_k as u32,
                /* total_q */ total_q as u32,
                /* total_k */ total_k as u32,
                /* softmax_scale*/ self.softmax_scale,
                /* is_bf16 */ is_bf16,
                /* is_e4m3 */ is_e4m3,
                /* window_size_left */ window_size_left,
                /* window_size_right */ window_size_right,
                /* attention_chunk */ 0,
                /* page_size */ page_block_size as i32,
                /* page_table_batch_stride*/ block_table_batch_stride as u32,
                /* num_pages */ num_pages as i32,
                /* num_splits */ 1,
                /* softmax_lseaccum_ptr */ std::ptr::null_mut(),
                /* oaccum_ptr */ std::ptr::null_mut(),
                /* oaccum_split_stride */ 0u32,
                /* oaccum_batch_stride */ 0u32,
                /* oaccum_row_stride */ 0u32,
                /* oaccum_head_stride */ 0u32,
                /* lseaccum_split_stride */ 0u32,
                /* lseaccum_batch_stride */ 0u32,
                /* lseaccum_head_stride */ 0u32,
                /* softcap */ self.softcap.unwrap_or(0.0),
                /* pack_gqa */ -1,
                *dev.cu_stream() as i64,
            )
        }

        let dst = candle::CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, out_shape))
    }
}

impl candle::CustomOp3 for FlashAttnVarLen {
    fn name(&self) -> &'static str {
        "flash-attn-varlen"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for flash-attn")
    }

    fn cuda_fwd(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        k: &candle::CudaStorage,
        k_l: &Layout,
        v: &candle::CudaStorage,
        v_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        match q.dtype() {
            candle::DType::F16 => self.cuda_fwd_t::<f16, f16>(q, q_l, k, k_l, v, v_l, false, false),
            candle::DType::BF16 => {
                self.cuda_fwd_t::<bf16, bf16>(q, q_l, k, k_l, v, v_l, true, false)
            }
            candle::DType::U8 => self.cuda_fwd_t::<u8, bf16>(q, q_l, k, k_l, v, v_l, true, true),
            dt => candle::bail!("flash-attn is only supported for f16/bf16/u8 ({dt:?})"),
        }
    }
}

#[allow(clippy::too_many_arguments)]
/// Flash-attention v3 layer with variable-length batching.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(total_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `seqlens_q` - The cumulative lengths of the sequences in the batch, used to index in q.
/// * `seqlens_k` - The cumulative lengths of the sequences in the batch, used to index in k and v.
/// * `max_seqlen_q` - The maximum query sequence length for q in the batch.
/// * `max_seqlen_k` - The maximum query sequence length for k and v in the batch.
///
/// `seqlens_q` and `seqlens_k` contain `batch_size + 1` elements, typically `0`, `seqlen_1`,
/// `seqlen_1 + seqlen_2`, etc.
///
/// The resulting tensor has dimensions `(batch_size, seqlen_q, num_heads_q, head_size_v)`.
pub fn flash_attn_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    block_table: &Option<Tensor>,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let window_size_left = None;
    let window_size_right = if causal { Some(0) } else { None };

    let op = FlashAttnVarLen {
        softmax_scale,
        max_seqlen_q,
        max_seqlen_k,
        seqlens_q: seqlens_q.clone(),
        seqlens_k: seqlens_k.clone(),
        block_table: block_table.clone(),
        window_size_left,
        window_size_right,
        softcap: None,
    };
    q.apply_op3(k, v, op)
}

pub fn flash_attn_varlen_softcap(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    block_table: &Option<Tensor>,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    softcap: Option<f32>,
    causal: bool,
) -> Result<Tensor> {
    let window_size_left = None;
    let window_size_right = if causal { Some(0) } else { None };

    let op = FlashAttnVarLen {
        softmax_scale,
        max_seqlen_q,
        max_seqlen_k,
        seqlens_q: seqlens_q.clone(),
        seqlens_k: seqlens_k.clone(),
        block_table: block_table.clone(),
        window_size_left,
        window_size_right,
        softcap,
    };
    q.apply_op3(k, v, op)
}

#[allow(clippy::too_many_arguments)]
/// Flash-attention v3 layer with variable-length batching.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(total_q, num_heads_q, head_size)`.
/// * `k` - Key tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `v` - Value tensor with shape `(total_kv, num_heads_kv, head_size)`.
/// * `seqlens_q` - The cumulative lengths of the sequences in the batch, used to index in q.
/// * `seqlens_k` - The cumulative lengths of the sequences in the batch, used to index in k and v.
/// * `max_seqlen_q` - The maximum query sequence length for q in the batch.
/// * `max_seqlen_k` - The maximum query sequence length for k and v in the batch.
/// * `window_size_left` - Limit left attention to value tokens.
/// * `window_size_right` - Limit right attention to value tokens.
///
/// `seqlens_q` and `seqlens_k` contain `batch_size + 1` elements, typically `0`, `seqlen_1`,
/// `seqlen_1 + seqlen_2`, etc.
///
/// The resulting tensor has dimensions `(total_q, num_heads_q, head_size_v)`.
///
/// # Causal mask
///
/// `window_size_left=None` with `window_size_right=Some(0)` applies a causal mask to the result
/// of  `Q @ K^T`
pub fn flash_attn_varlen_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    block_table: &Option<Tensor>,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor> {
    let op = FlashAttnVarLen {
        softmax_scale,
        max_seqlen_q,
        max_seqlen_k,
        seqlens_q: seqlens_q.clone(),
        seqlens_k: seqlens_k.clone(),
        block_table: block_table.clone(),
        window_size_left,
        window_size_right,
        softcap: None,
    };
    q.apply_op3(k, v, op)
}

pub fn flash_attn_varlen_windowed_softcap(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    block_table: &Option<Tensor>,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    softcap: Option<f32>,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor> {
    let op = FlashAttnVarLen {
        softmax_scale,
        max_seqlen_q,
        max_seqlen_k,
        seqlens_q: seqlens_q.clone(),
        seqlens_k: seqlens_k.clone(),
        block_table: block_table.clone(),
        window_size_left,
        window_size_right,
        softcap,
    };
    q.apply_op3(k, v, op)
}

#[cfg(feature = "flash-decoding")]
fn num_splits_heuristic(
    batch_nheads_mblocks: i32,
    num_sms: i32,
    num_n_blocks: i32,
    max_splits_input: i32,
) -> i32 {
    if batch_nheads_mblocks as f32 >= 0.8 * num_sms as f32 {
        return 1;
    }

    let max_splits = max_splits_input.min(num_sms).min(num_n_blocks);
    let mut max_efficiency = 0.0f32;
    let mut efficiency = Vec::with_capacity(max_splits as usize);

    let ceildiv = |a: i32, b: i32| (a + b - 1) / b;

    let is_split_eligible = |num_splits: i32| {
        num_splits == 1
            || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1)
    };

    for num_splits in 1..=max_splits {
        if !is_split_eligible(num_splits) {
            efficiency.push(0.0);
        } else {
            let n_waves = (batch_nheads_mblocks * num_splits) as f32 / num_sms as f32;
            let eff = n_waves / n_waves.ceil();
            if eff > max_efficiency {
                max_efficiency = eff;
            }
            efficiency.push(eff);
        }
    }

    for num_splits in 1..=max_splits {
        if !is_split_eligible(num_splits) {
            continue;
        }
        if efficiency[(num_splits - 1) as usize] >= 0.85 * max_efficiency {
            return num_splits;
        }
    }

    1
}

#[cfg(feature = "flash-decoding")]
pub fn get_num_splits(
    batch_size: usize,
    num_heads: usize,
    head_size: usize,
    max_seqlen_k: usize,
    max_seqlen_q: usize,
    num_sm: usize,
) -> usize {
    let block_n = if head_size <= 64 {
        256
    } else if head_size <= 128 {
        128
    } else {
        64
    };

    let num_n_blocks = (max_seqlen_k + block_n - 1).div_ceil(block_n);
    let num_m_blocks = (max_seqlen_q + 64 - 1).div_ceil(64);
    // 128 threads per block hard-coded factor
    let num_splits = num_splits_heuristic(
        (batch_size * num_heads * num_m_blocks) as i32,
        (num_sm * 2) as i32,
        num_n_blocks as i32,
        128i32,
    );

    assert!(num_splits <= 128, "num_splits > 128 not supported");

    num_splits as usize
}

#[cfg(feature = "flash-decoding")]
// A global, thread-safe cache of SM count per device ID
static SM_COUNT_CACHE: Lazy<Mutex<HashMap<i32, i32>>> = Lazy::new(|| Mutex::new(HashMap::new()));

#[cfg(feature = "flash-decoding")]
pub fn get_multiprocessor_count(device: &candle::CudaDevice) -> Result<i32> {
    use candle::cuda_backend::cudarc::driver::sys;
    let device_id = device.cu_device();
    // Lock the cache for access
    let mut cache = SM_COUNT_CACHE.lock().unwrap();

    if let Some(&cached_value) = cache.get(&device_id) {
        return Ok(cached_value);
    }

    // Not cached: query
    let mut value: i32 = 0;
    unsafe {
        sys::lib()
            .cuDeviceGetAttribute(
                &mut value as *mut _,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                *device_id,
            )
            .result()
            .map_err(|e| candle::Error::Msg(format!("cuDeviceGetAttribute failed: {e:?}")))?;
    }

    // Insert result into cache
    cache.insert(*device_id, value);
    Ok(value)
}

#[cfg(feature = "flash-decoding")]
struct FlashAttnCache {
    pub softmax_scale: f32,
    pub block_table: Option<Tensor>,
    pub context_lens: Option<Tensor>,
    pub softcap: Option<f32>,
    pub seqlenq_ngroups_swapped: bool,
    pub q_batch_stride: usize,
}

#[cfg(feature = "flash-decoding")]
impl FlashAttnCache {
    fn cuda_fwd_t<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
        OutT: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        k_cache: &candle::CudaStorage,
        k_cache_l: &Layout,
        v_cache: &candle::CudaStorage,
        v_cache_l: &Layout,
        is_bf16: bool,
        is_e4m3: bool,
    ) -> Result<(candle::CudaStorage, Shape)> {
        let dev = q.device();

        // println!("q {:?}, k_cache {:?}, v_cache {:?}", q_l.shape(), k_cache_l.shape(), v_cache_l.shape());
        let q = q.as_cuda_slice::<T>()?;
        let k_cache = k_cache.as_cuda_slice::<T>()?;
        let v_cache = v_cache.as_cuda_slice::<T>()?;
        let q = q.slice(q_l.start_offset()..);
        let k_cache = k_cache.slice(k_cache_l.start_offset()..);
        let v_cache = v_cache.slice(v_cache_l.start_offset()..);

        let q_stride = q_l.stride();
        let k_stride = k_cache_l.stride();
        let v_stride = v_cache_l.stride();

        let q_rank = q_stride.len();
        let k_rank = k_stride.len();
        let v_rank = v_stride.len();

        //flash attn q expect [batch_size, seqlen_q, num_heads, head_dim]
        if q_rank != 4 || k_rank != 4 || v_rank != 4 {
            candle::bail!(
                "flash-attn-with_kvcache expects input tensors of rank 4 (q: {q_rank}, k: {k_rank}, v: {v_rank}"
            )
        }
        if q_stride[q_rank - 1] != 1 {
            candle::bail!("the last dim of q must be contiguous {q_stride:?}")
        }
        if k_stride[k_rank - 1] != 1 {
            candle::bail!("the last dim of k_cache must be contiguous {k_stride:?}")
        }
        if v_stride[v_rank - 1] != 1 {
            candle::bail!("the last dim of v_cache must be contiguous {v_stride:?}")
        }

        //flash attention expect kv cache
        // kv_cache = [num_blocks, block_size, num_kv_heads, head_size]
        let (batch_size, seqlen_q, num_heads, head_size_og) = q_l.shape().dims4()?;
        let (num_blocks, block_size, num_heads_k, head_size_k) = k_cache_l.shape().dims4()?;
        let (v_num_blocks, v_block_size, v_num_heads_k, head_size_v) = v_cache_l.shape().dims4()?;
        if (num_blocks, block_size, num_heads_k) != (v_num_blocks, v_block_size, v_num_heads_k) {
            candle::bail!(
                "shape mismatch k_cache {:?} and v_cache {:?}",
                k_cache_l.shape(),
                v_cache_l.shape()
            )
        }
        if head_size_k != head_size_og {
            candle::bail!(
                "head size mismatch q {:?} and k_cache {:?}",
                q_l.shape(),
                k_cache_l.shape()
            )
        }

        if head_size_og > 256 {
            candle::bail!("only supports head dimension at most 256 (got {head_size_og})")
        }
        if head_size_og % 8 != 0 {
            // TODO: Handle head sizes that are not a multiple of 8 via some padding.
            candle::bail!("only supports head sizes that are a multiple of 8 (got {head_size_og})")
        }
        if num_heads % num_heads_k != 0 {
            candle::bail!("number of k/v heads {num_heads_k} must divide number of heads in query {num_heads}")
        }

        let (
            block_table_ptr,
            block_table_batch_stride,
            block_table_batch_size,
            max_num_blocks_per_seq,
        ) = if let Some(block_table) = &self.block_table {
            // println!("block table: {:?}", block_table.to_device(&candle::Device::Cpu)?.to_vec2::<u32>()?);
            let max_num_blocks_per_seq = block_table.dim(1)?;
            let block_table_batch_size = block_table.dim(0)?;
            let (block_table, block_table_layout) = block_table.storage_and_layout();
            let block_table_batch_stride = block_table_layout.stride()[0];
            let block_table = match &*block_table {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
                _ => candle::bail!("block_table must be a cuda tensor"),
            };

            let block_table = block_table.slice(block_table_layout.start_offset()..);

            (
                *block_table.device_ptr() as *const core::ffi::c_void,
                block_table_batch_stride,
                block_table_batch_size,
                max_num_blocks_per_seq,
            )
        } else {
            (std::ptr::null(), 0usize, 0usize, 0usize)
        };

        let context_lens_ptr = if let Some(context_lens) = &self.context_lens {
            // println!("context_lens: {:?}", context_lens.to_device(&candle::Device::Cpu)?.to_vec1::<u32>()?);

            let (context_lens, context_lens_layout) = context_lens.storage_and_layout();
            let context_lens = match &*context_lens {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<i32>()?,
                _ => candle::bail!("context_lens must be a cuda tensor"),
            };

            let context_lens = context_lens.slice(context_lens_layout.start_offset()..);

            *context_lens.device_ptr() as *const core::ffi::c_int
        } else {
            std::ptr::null() as *const core::ffi::c_int
        };

        let out_shape = Shape::from((batch_size, seqlen_q, num_heads, head_size_v));
        let out_l = Layout::contiguous(&out_shape);
        let o_stride = out_l.stride();

        let seqlen_k = max_num_blocks_per_seq * block_size;

        let elem_count = out_shape.elem_count();
        let dst = unsafe { dev.alloc::<OutT>(elem_count) }.w()?;
        let softmax_lse = dev
            .alloc_zeros::<f32>(num_heads * batch_size * seqlen_q, false)
            .w()?;

        let is_bf16 = if is_bf16 { 1 } else { 0 };
        let is_e4m3 = if is_e4m3 { 1 } else { 0 };

        let mut o_batch_stride = o_stride[0] as u32;

        if self.seqlenq_ngroups_swapped {
            o_batch_stride *= seqlen_q as u32;
        }

        unsafe {
            let q_ptr = *q.device_ptr() as *const core::ffi::c_void;
            let k_cache_ptr = *k_cache.device_ptr() as *const core::ffi::c_void;
            let v_cache_ptr = *v_cache.device_ptr() as *const core::ffi::c_void;
            let dst_ptr = *dst.device_ptr() as *mut core::ffi::c_void;
            let softmax_lse_ptr = *softmax_lse.device_ptr() as *mut core::ffi::c_void;
            ffi::run_mha(
                q_ptr,
                k_cache_ptr,
                v_cache_ptr,
                block_table_ptr,
                dst_ptr,
                softmax_lse_ptr,
                /* cu_seqlens_q_ptr */ std::ptr::null(),
                /* cu_seqlens_k_ptr */ std::ptr::null(),
                /* seqused_q_ptr */ std::ptr::null(),
                /* seqused_k_ptr */ context_lens_ptr,
                /* leftpad_k_ptr */ std::ptr::null(),
                /* kv_batch_idx_ptr */ std::ptr::null(),
                /* q_descale_ptr */ std::ptr::null(),
                /* k_descale_ptr */ std::ptr::null(),
                /* v_descale_ptr */ std::ptr::null(),
                /* q_batch_stride */ self.q_batch_stride as u32,
                /* k_batch_stride */ k_stride[0] as u32,
                /* v_batch_stride */ v_stride[0] as u32,
                /* o_batch_stride */ o_batch_stride,
                /* q_row_stride   */ q_stride[1] as u32,
                /* k_row_stride   */ k_stride[1] as u32,
                /* v_row_stride   */ v_stride[1] as u32,
                /* o_row_stride   */ o_stride[1] as u32,
                /* q_head_stride  */ q_stride[2] as u32,
                /* k_head_stride  */ k_stride[2] as u32,
                /* v_head_stride  */ v_stride[2] as u32,
                /* o_head_stride  */ o_stride[2] as u32,
                /* v_dim_stride */ v_stride[v_rank - 1] as u32,
                /* q_descale_batch_stride */ 0,
                /* q_descale_head_stride */ 0,
                /* k_descale_batch_stride */ 0,
                /* k_descale_head_stride */ 0,
                /* v_descale_batch_stride */ 0,
                /* v_descale_head_stride */ 0,
                /* b */ batch_size as u32,
                /* b_k */
                if block_table_ptr.is_null() {
                    batch_size as u32
                } else {
                    block_table_batch_size as u32
                },
                /* h */ num_heads as u32,
                /* h_k */ num_heads_k as u32,
                /* d */ head_size_og as u32,
                /* dv */ head_size_v as u32,
                /* softmax_scale*/ self.softmax_scale,
                /* seqlen_q */ seqlen_q as u32,
                /* seqlen_k */ seqlen_k as u32,
                /* total_q */ (batch_size * seqlen_q) as u32,
                /* total_k */ (batch_size * seqlen_k) as u32,
                /* is_bf16 */ is_bf16,
                /* is_e4m3 */ is_e4m3,
                /* window_size_left */ -1,
                /* window_size_right */ -1,
                /* attention_chunk */ 0,
                /* page_size */ block_size as i32,
                /* page_table_batch_stride */ block_table_batch_stride as u32,
                /* num_pages */ num_blocks as i32,
                /* num_splits */ 1,
                /* softmax_lseaccum_ptr */ std::ptr::null_mut(),
                /* oaccum_ptr */ std::ptr::null_mut(),
                /* oaccum_split_stride */ 0u32,
                /* oaccum_batch_stride */ 0u32,
                /* oaccum_row_stride */ 0u32,
                /* oaccum_head_stride */ 0u32,
                /* lseaccum_split_stride */ 0u32,
                /* lseaccum_batch_stride */ 0u32,
                /* lseaccum_head_stride */ 0u32,
                /* softcap */ self.softcap.unwrap_or(0.0),
                /* pack_gqa */ -1,
                *dev.cu_stream() as i64,
            )
        }

        // panic!("finished first run");
        let dst = candle::CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, out_shape))
    }
}

#[cfg(feature = "flash-decoding")]
impl candle::CustomOp3 for FlashAttnCache {
    fn name(&self) -> &'static str {
        "flash-attn-with-kvcache"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for flash-attn")
    }

    fn cuda_fwd(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        k_cache: &candle::CudaStorage,
        k_cache_l: &Layout,
        v_cache: &candle::CudaStorage,
        v_cache_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        match q.dtype() {
            candle::DType::F16 => self.cuda_fwd_t::<f16, f16>(
                q, q_l, k_cache, k_cache_l, v_cache, v_cache_l, false, false,
            ),
            candle::DType::BF16 => self.cuda_fwd_t::<bf16, bf16>(
                q, q_l, k_cache, k_cache_l, v_cache, v_cache_l, true, false,
            ),
            candle::DType::U8 => self
                .cuda_fwd_t::<u8, bf16>(q, q_l, k_cache, k_cache_l, v_cache, v_cache_l, true, true),
            dt => candle::bail!("flash-attn is only supported for f16/bf16/u8 ({dt:?})"),
        }
    }
}

#[allow(clippy::too_many_arguments)]
/// Flash-attention with kv cache.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors k and v with fewer heads
/// than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(batch_size, seqlen_q, num_heads_q, head_size)`.
/// * `k_cache` - Key cache tensor with shape `[num_blocks, block_size, num_kv_heads, head_size]`.
/// * `v_cache` - Value cache tensor with shape `[num_blocks, block_size, num_kv_heads, head_size]`.
/// * `max_seqlen_q` - The maximum query sequence length for q in the batch.
/// * `max_seqlen_k` - The maximum query sequence length for k and v in the batch.
///
///
/// The resulting tensor has dimensions `(total_q, num_heads_q, head_size_v)`.
#[cfg(feature = "flash-decoding")]
pub fn flash_attn_with_kvcache(
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    context_lens: &Tensor,
    block_table: &Tensor,
    softmax_scale: f32,
) -> Result<Tensor> {
    let (batch_size, mut seqlen_q, num_heads, head_size_og) = q.dims4()?;
    let (_, _, num_heads_k, _) = k_cache.dims4()?;
    let (_, _, _, head_size_v) = v_cache.dims4()?;
    let mut q_batch_stride = q.stride()[0];

    let seqlenq_ngroups_swapped = false; //seqlen_q == 1 && num_heads > num_heads_k && head_size_og % 8 == 0;
    let q = if seqlenq_ngroups_swapped {
        let ngroups = num_heads / num_heads_k;
        seqlen_q = ngroups;
        q_batch_stride *= seqlen_q;
        q.reshape((batch_size, num_heads_k, ngroups, head_size_og))?
            .transpose(1, 2)?
    } else {
        q.to_owned()
    };

    let op = FlashAttnCache {
        softmax_scale,
        context_lens: Some(context_lens.to_owned()),
        block_table: Some(block_table.to_owned()),
        softcap: None,
        seqlenq_ngroups_swapped,
        q_batch_stride,
    };
    let o = q.apply_op3(k_cache, v_cache, op)?;
    let o = if seqlenq_ngroups_swapped {
        o.transpose(1, 2)?
            .reshape((batch_size, 1, num_heads_k * seqlen_q, head_size_v))?
    } else {
        o
    };
    Ok(o)
}
