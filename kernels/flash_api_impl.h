/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

#include "flash.h"
#include "static_switch.h"
#include "tile_size.h"
#include "heuristics.h"
#include "cuda_check.h"
#include "flash_fwd_launch_template.h"
#include "flash_fwd_combine_launch_template.h"

#define PREPARE_VARLEN_MAX_BATCHES_1CTA 992

namespace flash_api {
inline int round_multiple(int x, int m) { return (x + m - 1) / m * m; }

inline void fill_device_properties(Flash_fwd_params &params) {
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    params.arch = prop.major * 10 + prop.minor;
    params.num_sm = prop.multiProcessorCount;
}

inline int get_max_headdim() {
    #ifndef FLASHATTENTION_DISABLE_HDIM256
    return 256;
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM192
    return 192;
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM128
    return 128;
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM96
    return 96;
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM64
    return 64;
    #endif
    return 0;
}

inline int round_up_headdim(int head_size) {
    #ifndef FLASHATTENTION_DISABLE_HDIM64
    if (head_size <= 64) { return 64; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM96
    if (head_size <= 96) { return 96; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM128
    if (head_size <= 128) { return 128; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM192
    if (head_size <= 192) { return 192; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM256
    if (head_size <= 256) { return 256; }
    #endif
    return 256;
}

inline int round_up_headdimv(int head_size) {
    if (head_size <= 64) { return 64; }
    if (head_size <= 96) { return 96; }
    if (head_size <= 128) { return 128; }
    if (head_size <= 192) { return 192; }
    if (head_size <= 256) { return 256; }
    return 512;
}

inline bool get_pagedkv_tma(Flash_fwd_params const& params) {
    if (params.arch < 90 || !params.page_table || params.leftpad_k || params.knew_ptr) { return false; }
    auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, false /*paged_kv_non_TMA*/, params.softcap > 0.f);
    int const kBlockM = std::get<0>(kBlockMN_kernel_args_sm90);
    int const kBlockN = std::get<1>(kBlockMN_kernel_args_sm90);
    return params.page_size % kBlockN == 0 && params.seqlen_q * (params.h / params.h_k) > kBlockM;
}

inline bool get_pack_gqa(Flash_fwd_params const& params) {
    if (params.arch < 90 || (params.page_table && !params.pagedkv_tma) || params.num_splits > 1) { return true; }
    #ifdef FLASHATTENTION_DISABLE_PACKGQA
    return false;
    #else
    if (params.h == params.h_k) { return false; }
    auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, params.page_table && !params.pagedkv_tma, params.softcap > 0.f);
    int const kBlockM = std::get<0>(kBlockMN_kernel_args_sm90);
    return should_pack_gqa(params.cu_seqlens_q || params.seqused_q, params.seqlen_q, params.h / params.h_k, kBlockM);
    #endif
}

inline int get_num_splits(Flash_fwd_params const& params) {
    #ifdef FLASHATTENTION_DISABLE_SPLIT
    return 1;
    #else
    bool varlen = params.cu_seqlens_q || params.cu_seqlens_k || params.seqused_q || params.seqused_k || params.leftpad_k;
    auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, params.page_table && !params.pagedkv_tma, params.softcap > 0.f);
    auto kBlockMN_kernel_args_sm8x = tile_size_fwd_sm8x(params.arch == 86 || params.arch == 89, params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, params.page_table, varlen, params.softcap > 0.f, params.knew_ptr);
    int const kBlockM = params.arch >= 90 ? std::get<0>(kBlockMN_kernel_args_sm90) : std::get<0>(kBlockMN_kernel_args_sm8x);
    int const kBlockN = params.arch >= 90 ? std::get<1>(kBlockMN_kernel_args_sm90) : std::get<1>(kBlockMN_kernel_args_sm8x);
    int seqlen_q_packgqa = params.seqlen_q * (params.h / params.h_k);
    int const seqlen_k_loaded = !params.is_local
        ? params.seqlen_k
        : std::max(0, std::min(params.seqlen_k, params.window_size_right + params.window_size_left + 1 + kBlockM));
    int const num_n_blocks = (seqlen_k_loaded + kBlockN - 1) / kBlockN;
    int const num_m_blocks = (seqlen_q_packgqa + kBlockM - 1) / kBlockM;
    int const size_one_kv_head = params.seqlen_k * (params.d + params.dv) * (params.is_e4m3 ? 1 : 2);
    int total_mblocks = (params.num_splits_dynamic_ptr ? 1 : params.b) * params.h_k * num_m_blocks;
    return num_splits_heuristic(total_mblocks, params.num_sm, num_n_blocks, num_m_blocks, size_one_kv_head, params.is_causal || params.is_local, 128);
    #endif
}

template <int Arch, int Split, bool PagedKVNonTMA, bool PackGQA, bool Has_softcap>
void run_mha_fwd_constexpr(Flash_fwd_params &params, cudaStream_t stream) {
    if (!params.is_e4m3) {
        if (params.is_bf16) {
            #ifndef FLASHATTENTION_DISABLE_HDIM64
            if (params.d <= 64) {
                #ifndef FLASHATTENTION_DISABLE_HDIMDIFF64
                if constexpr (Arch == 90) {
                    if (params.dv > 256) {
                        return run_mha_fwd_<Arch, cutlass::bfloat16_t, 64, 512, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                    } else if (params.dv > 64) {
                        return run_mha_fwd_<Arch, cutlass::bfloat16_t, 64, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                    }
                }
                #endif
                return run_mha_fwd_<Arch, cutlass::bfloat16_t, 64, 64, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
            }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM96
            if (params.d <= 96) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, 96, 96, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM128
            if (params.d <= 128) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, 128, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM192
            if (params.d <= 192) {
                #ifndef FLASHATTENTION_DISABLE_HDIMDIFF192
                if constexpr (Arch == 90) {
                    if (params.dv <= 128) {
                        return run_mha_fwd_<Arch, cutlass::bfloat16_t, 192, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                    }
                }
                #endif
                return run_mha_fwd_<Arch, cutlass::bfloat16_t, 192, 192, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
            }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM256
            if (params.d <= 256) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, 256, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
            #endif
        } else {
            #ifndef FLASHATTENTION_DISABLE_FP16
            #ifndef FLASHATTENTION_DISABLE_HDIM64
            if (params.d <= 64) {
                #ifndef FLASHATTENTION_DISABLE_HDIMDIFF64
                if constexpr (Arch == 90) {
                    if (params.dv > 256) {
                        return run_mha_fwd_<Arch, cutlass::half_t, 64, 512, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                    } else if (params.dv > 64) {
                        return run_mha_fwd_<Arch, cutlass::half_t, 64, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                    }
                }
                #endif
                return run_mha_fwd_<Arch, cutlass::half_t, 64, 64, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
            }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM96
            if (params.d <= 96) { return run_mha_fwd_<Arch, cutlass::half_t, 96, 96, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM128
            if (params.d <= 128) { return run_mha_fwd_<Arch, cutlass::half_t, 128, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM192
            if (params.d <= 192) {
                #ifndef FLASHATTENTION_DISABLE_HDIMDIFF192
                if constexpr (Arch == 90) {
                    if (params.dv <= 128) {
                        return run_mha_fwd_<Arch, cutlass::half_t, 192, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                    }
                }
                #endif
                return run_mha_fwd_<Arch, cutlass::half_t, 192, 192, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
            }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM256
            if (params.d <= 256) { return run_mha_fwd_<Arch, cutlass::half_t, 256, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
            #endif
            #endif
        }
    } else {
        #ifndef FLASHATTENTION_DISABLE_FP8
        #ifndef FLASHATTENTION_DISABLE_HDIM64
        if (params.d <= 64) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 64, 64, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
        #endif
        #ifndef FLASHATTENTION_DISABLE_HDIM96
        if (params.d <= 96) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 96, 96, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
        #endif
        #ifndef FLASHATTENTION_DISABLE_HDIM128
        if (params.d <= 128) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 128, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
        #endif
        #ifndef FLASHATTENTION_DISABLE_HDIM192
        if (params.d <= 192) {
            #ifndef FLASHATTENTION_DISABLE_HDIMDIFF192
            if (params.dv <= 128) {
                return run_mha_fwd_<90, cutlass::float_e4m3_t, 192, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
            }
            #endif
            return run_mha_fwd_<90, cutlass::float_e4m3_t, 192, 192, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
        }
        #endif
        #ifndef FLASHATTENTION_DISABLE_HDIM256
        if (params.d <= 256) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 256, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
        #endif
        #endif
    }
}

template <int Arch, bool Has_softcap>
void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    #ifndef FLASHATTENTION_DISABLE_SPLIT
    SPLIT_SWITCH(params.num_splits > 1, Split, [&] {
    #else
    constexpr bool Split = false;
    #endif
        if (params.page_table && !params.pagedkv_tma) {
            constexpr bool PagedKVNonTMA = true;
            if (params.pack_gqa) {
                constexpr bool PackGQA = true;
                run_mha_fwd_constexpr<Arch, Split, PagedKVNonTMA, PackGQA, Has_softcap>(params, stream);
            } else {
                constexpr bool PackGQA = false;
                run_mha_fwd_constexpr<Arch, Split, PagedKVNonTMA, PackGQA, Has_softcap>(params, stream);
            }
        } else {
            constexpr bool PagedKVNonTMA = false;
            if (params.pack_gqa) {
                constexpr bool PackGQA = true;
                run_mha_fwd_constexpr<Arch, Split, PagedKVNonTMA, PackGQA, Has_softcap>(params, stream);
            } else {
                constexpr bool PackGQA = false;
                run_mha_fwd_constexpr<Arch, Split, PagedKVNonTMA, PackGQA, Has_softcap>(params, stream);
            }
        }
    #ifndef FLASHATTENTION_DISABLE_SPLIT
    });
    #endif
}

inline void run_mha_fwd_combine(Flash_fwd_params &params, cudaStream_t stream, bool enable_pdl=false) {
    #ifndef FLASHATTENTION_DISABLE_SPLIT
    if (params.is_fp32) {
        if (params.dv <= 64) {
            run_mha_fwd_combine_<float, float, 64>(params, stream, enable_pdl);
        } else {
            run_mha_fwd_combine_<float, float, 128>(params, stream, enable_pdl);
        }
    } else if (params.is_bf16) {
        if (params.dv <= 64) {
            run_mha_fwd_combine_<cutlass::bfloat16_t, float, 64>(params, stream, enable_pdl);
        } else {
            run_mha_fwd_combine_<cutlass::bfloat16_t, float, 128>(params, stream, enable_pdl);
        }
    } else {
        if (params.dv <= 64) {
            run_mha_fwd_combine_<cutlass::half_t, float, 64>(params, stream, enable_pdl);
        } else {
            run_mha_fwd_combine_<cutlass::half_t, float, 128>(params, stream, enable_pdl);
        }
    }
    #else
    (void)params;
    (void)stream;
    (void)enable_pdl;
    #endif
}

#define FA_CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "flashattn: %s\n", msg); return; } } while (0)

template <int Arch, bool Has_softcap>
void run_mha_impl(
    void *q_ptr,
    void *k_ptr,
    void *v_ptr,
    void *page_table_ptr,
    void *o_ptr,
    void *softmax_lse_ptr,
    int32_t *cu_seqlens_q_ptr,
    int32_t *cu_seqlens_k_ptr,
    int32_t *seqused_q_ptr,
    int32_t *seqused_k_ptr,
    int32_t *leftpad_k_ptr,
    int32_t *kv_batch_idx_ptr,
    float *q_descale_ptr,
    float *k_descale_ptr,
    float *v_descale_ptr,
    uint32_t q_batch_stride,
    uint32_t k_batch_stride,
    uint32_t v_batch_stride,
    uint32_t o_batch_stride,
    uint32_t q_row_stride,
    uint32_t k_row_stride,
    uint32_t v_row_stride,
    uint32_t o_row_stride,
    uint32_t q_head_stride,
    uint32_t k_head_stride,
    uint32_t v_head_stride,
    uint32_t o_head_stride,
    uint32_t v_dim_stride,
    uint32_t q_descale_batch_stride,
    uint32_t q_descale_head_stride,
    uint32_t k_descale_batch_stride,
    uint32_t k_descale_head_stride,
    uint32_t v_descale_batch_stride,
    uint32_t v_descale_head_stride,
    uint32_t b,
    uint32_t b_k,
    uint32_t h,
    uint32_t h_k,
    uint32_t d,
    uint32_t dv,
    uint32_t seqlen_q,
    uint32_t seqlen_k,
    uint32_t total_q,
    uint32_t total_k,
    float softmax_scale,
    int is_bf16,
    int is_e4m3,
    int window_size_left,
    int window_size_right,
    int attention_chunk,
    int page_size,
    uint32_t page_table_batch_stride,
    int num_pages,
    int num_splits,
    void *softmax_lseaccum_ptr,
    void *oaccum_ptr,
    uint32_t oaccum_split_stride,
    uint32_t oaccum_batch_stride,
    uint32_t oaccum_row_stride,
    uint32_t oaccum_head_stride,
    uint32_t lseaccum_split_stride,
    uint32_t lseaccum_batch_stride,
    uint32_t lseaccum_head_stride,
    float softcap,
    int pack_gqa,
    int64_t cu_stream
) {
    Flash_fwd_params params{};

    params.q_ptr = q_ptr;
    params.k_ptr = k_ptr;
    params.v_ptr = v_ptr;
    params.o_ptr = o_ptr;
    params.softmax_lse_ptr = softmax_lse_ptr;

    params.q_batch_stride = q_batch_stride;
    params.k_batch_stride = k_batch_stride;
    params.v_batch_stride = v_batch_stride;
    params.o_batch_stride = o_batch_stride;
    params.q_row_stride = q_row_stride;
    params.k_row_stride = k_row_stride;
    params.v_row_stride = v_row_stride;
    params.o_row_stride = o_row_stride;
    params.q_head_stride = q_head_stride;
    params.k_head_stride = k_head_stride;
    params.v_head_stride = v_head_stride;
    params.o_head_stride = o_head_stride;
    params.v_dim_stride = v_dim_stride;

    params.cu_seqlens_q = cu_seqlens_q_ptr;
    params.cu_seqlens_k = cu_seqlens_k_ptr;
    params.seqused_q = seqused_q_ptr;
    params.seqused_k = seqused_k_ptr;
    params.leftpad_k = leftpad_k_ptr;
    params.kv_batch_idx = kv_batch_idx_ptr;

    params.page_table = page_table_ptr ? static_cast<int *>(page_table_ptr) : nullptr;
    params.page_table_batch_stride = page_table_batch_stride;
    params.page_size = page_size;
    params.num_pages = num_pages;

    params.q_descale_ptr = q_descale_ptr;
    params.k_descale_ptr = k_descale_ptr;
    params.v_descale_ptr = v_descale_ptr;
    params.q_descale_batch_stride = q_descale_batch_stride;
    params.q_descale_head_stride = q_descale_head_stride;
    params.k_descale_batch_stride = k_descale_batch_stride;
    params.k_descale_head_stride = k_descale_head_stride;
    params.v_descale_batch_stride = v_descale_batch_stride;
    params.v_descale_head_stride = v_descale_head_stride;

    params.b = static_cast<int>(b);
    params.b_k = static_cast<int>(b_k);
    params.h = static_cast<int>(h);
    params.h_k = static_cast<int>(h_k);
    params.seqlen_q = static_cast<int>(seqlen_q);
    params.seqlen_k = static_cast<int>(seqlen_k);
    params.total_q = static_cast<int>(total_q);
    params.total_k = static_cast<int>(total_k);
    params.d = static_cast<int>(d);
    params.dv = static_cast<int>(dv);
    params.d_rounded = round_up_headdim(params.d);
    params.dv_rounded = params.dv == params.d ? params.d_rounded : round_up_headdimv(params.dv);
    params.seqlen_q_rounded = round_multiple(params.seqlen_q, 128);
    params.seqlen_k_rounded = round_multiple(params.seqlen_k, 128);

    params.seqlen_knew = 0;
    params.total_knew = 0;
    params.knew_ptr = nullptr;
    params.vnew_ptr = nullptr;
    params.cu_seqlens_knew = nullptr;

    params.scale_softmax = softmax_scale;
    params.softcap = Has_softcap ? softcap : 0.0f;

    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0f));
    params.rp_dropout = 1.0f;

    params.is_bf16 = is_bf16 != 0;
    params.is_fp32 = false;
    params.is_e4m3 = is_e4m3 != 0;

    params.attention_chunk = attention_chunk;
    params.is_causal = window_size_left < 0 && window_size_right == 0 && attention_chunk == 0;
    params.is_local = (window_size_left >= 0 || window_size_right >= 0 || attention_chunk >= 1) && !params.is_causal;
    if (window_size_left < 0) { window_size_left = params.seqlen_k - 1; }
    if (window_size_right < 0) { window_size_right = params.seqlen_q - 1; }
    if (attention_chunk > 0) {
        window_size_left = std::min(window_size_left, attention_chunk - 1);
        window_size_right = std::min(window_size_right, attention_chunk - 1);
    }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    params.oaccum_ptr = oaccum_ptr;
    params.softmax_lseaccum_ptr = softmax_lseaccum_ptr;
    params.oaccum_split_stride = oaccum_split_stride;
    params.oaccum_batch_stride = oaccum_batch_stride;
    params.oaccum_row_stride = oaccum_row_stride;
    params.oaccum_head_stride = oaccum_head_stride;
    params.lseaccum_split_stride = lseaccum_split_stride;
    params.lseaccum_batch_stride = lseaccum_batch_stride;
    params.lseaccum_head_stride = lseaccum_head_stride;

    params.qv_ptr = nullptr;
    params.rotary_cos_ptr = nullptr;
    params.rotary_sin_ptr = nullptr;
    params.seqlens_rotary = nullptr;
    params.is_rotary_interleaved = false;

    fill_device_properties(params);
    params.arch = Arch;

    bool is_varlen = params.cu_seqlens_q || params.cu_seqlens_k || params.seqused_q || params.seqused_k || params.leftpad_k;
    bool use_prepare_varlen = is_varlen;

    params.skip_scheduler_metadata_computation = false;
    params.prepare_varlen_pdl = use_prepare_varlen && params.b <= PREPARE_VARLEN_MAX_BATCHES_1CTA;

    params.num_splits_dynamic_ptr = use_prepare_varlen ? reinterpret_cast<int *>(1) : nullptr;
    params.pagedkv_tma = get_pagedkv_tma(params);
    params.num_splits = num_splits <= 0 ? get_num_splits(params) : num_splits;
    params.pack_gqa = pack_gqa < 0 ? get_pack_gqa(params) : (pack_gqa != 0);

    if (params.num_splits > 1 && (!params.oaccum_ptr || !params.softmax_lseaccum_ptr)) {
        FA_CHECK(false, "num_splits > 1 requires oaccum and softmax_lseaccum buffers");
    }

    params.varlen_sort_batches = !params.is_local;
    params.head_swizzle = params.is_causal || params.is_local;

    int *metadata = nullptr;
    size_t metadata_size = 0;
    bool scheduler_needs_semaphore = params.arch >= 90 || params.num_splits > 1;
    if (use_prepare_varlen || scheduler_needs_semaphore) {
        if (use_prepare_varlen) {
            int b_rounded = round_multiple(params.b, 4);
            int num_prepare_batch_vectors = 2;
            if (params.varlen_sort_batches) { num_prepare_batch_vectors += 1; }
            if (params.head_swizzle) { num_prepare_batch_vectors += 1; }
            int head_swizzle_offset = b_rounded * (params.varlen_sort_batches ? 3 : 2);
            int tile_count_semaphore_offset = b_rounded * num_prepare_batch_vectors;
            metadata_size = static_cast<size_t>(tile_count_semaphore_offset + (scheduler_needs_semaphore ? 1 : 0));
            CHECK_CUDA(cudaMalloc(&metadata, metadata_size * sizeof(int)));
            CHECK_CUDA(cudaMemset(metadata, 0, metadata_size * sizeof(int)));
            params.num_splits_dynamic_ptr = metadata;
            params.num_m_blocks_ptr = metadata + b_rounded;
            params.varlen_batch_idx_ptr = params.varlen_sort_batches ? metadata + b_rounded * 2 : nullptr;
            params.num_nheads_in_l2_ptr = params.head_swizzle ? metadata + head_swizzle_offset : nullptr;
            params.tile_count_semaphore = scheduler_needs_semaphore ? metadata + tile_count_semaphore_offset : nullptr;
            params.tile_count_semaphore_offset = tile_count_semaphore_offset;
        } else {
            metadata_size = 1;
            CHECK_CUDA(cudaMalloc(&metadata, metadata_size * sizeof(int)));
            CHECK_CUDA(cudaMemset(metadata, 0, metadata_size * sizeof(int)));
            params.tile_count_semaphore = metadata;
        }
    }

    if (use_prepare_varlen) {
        auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, params.page_table && !params.pagedkv_tma, params.softcap > 0.f);
        auto kBlockMN_kernel_args_sm8x = tile_size_fwd_sm8x(params.arch == 86 || params.arch == 89, params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, params.page_table, is_varlen && params.num_splits > 1, params.softcap > 0.f, params.knew_ptr);
        int const kBlockM = params.arch >= 90 ? std::get<0>(kBlockMN_kernel_args_sm90) : std::get<0>(kBlockMN_kernel_args_sm8x);
        int const kBlockN = params.arch >= 90 ? std::get<1>(kBlockMN_kernel_args_sm90) : std::get<1>(kBlockMN_kernel_args_sm8x);
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(cu_stream);
        prepare_varlen_num_blocks(params, stream, params.pack_gqa, kBlockM, kBlockN, false /*enable_pdl*/);
        CHECK_CUDA_KERNEL_LAUNCH();
    }

    if (params.d > get_max_headdim()) {
        FA_CHECK(false, "head dimension exceeds max supported size");
    }

    if (params.total_q > 0 && (params.seqlen_k + params.total_knew) > 0 && params.h_k > 0) {
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(cu_stream);
        run_mha_fwd<Arch, Has_softcap>(params, stream);
        if (params.num_splits > 1 && params.oaccum_ptr && params.softmax_lseaccum_ptr) {
            run_mha_fwd_combine(params, stream, true /*enable_pdl*/);
        }
    }

    if (metadata) {
        CHECK_CUDA(cudaFree(metadata));
    }
}

}  // namespace flash_api

#undef FA_CHECK
