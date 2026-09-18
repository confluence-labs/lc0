/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "neural/backends/cuda/cuda_common.h"

// Fused MHA implementation from cutlass example #41
#include "fused_multi_head_attention/kernel_forward.h"
#include "utils/exception.h"

// Hero FFN fusion: CUTLASS fp16 gemm with a fused mish epilogue (folds the
// separate addBias-mish kernel into the up-projection gemm — ~5% on hero).
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination_generic.h"

namespace lczero {
namespace NS_BACKEND {

// mish(x) = x * tanh(softplus(x)); CUTLASS epilogue activation functor
template <typename T, int N>
struct MishActivation {
  CUTLASS_HOST_DEVICE cutlass::Array<T, N> operator()(cutlass::Array<T, N> const& x) const {
    cutlass::Array<T, N> y;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      float v = float(x[i]);
      float sp = v > 20.f ? v : logf(1.f + expf(v));
      y[i] = T(v * tanhf(sp));
    }
    return y;
  }
  CUTLASS_HOST_DEVICE T operator()(T const& s) const {
    float v = float(s); float sp = v > 20.f ? v : logf(1.f + expf(v));
    return T(v * tanhf(sp));
  }
};

// C[m,dff] = mish(A[m,d] @ Wup[dff,d]^T). Wup viewed ColumnMajor(ld=d) = [d,dff]=Wup^T.
// Output layout matches the cuBLAS up-gemm (row r's dff values contiguous) so the
// existing cuBLAS down-gemm reads it unchanged.
void cutlassFFNUpMish(const void* A, const void* Wup, void* C, int m, int d, int dff,
                      cudaStream_t stream) {
  using Elem = cutlass::half_t; using Acc = float;
  using Epilogue = cutlass::epilogue::thread::LinearCombinationGeneric<
      MishActivation, Elem, 128 / cutlass::sizeof_bits<Elem>::value, Acc, Acc>;
  using Gemm = cutlass::gemm::device::Gemm<
      Elem, cutlass::layout::RowMajor, Elem, cutlass::layout::ColumnMajor,
      Elem, cutlass::layout::RowMajor, Acc, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>, Epilogue>;
  Gemm op;
  typename Gemm::Arguments args({m, dff, d}, {(Elem const*)A, d}, {(Elem const*)Wup, d},
                                {(Elem*)C, dff}, {(Elem*)C, dff}, {1.f, 0.f});
  if (op(args, nullptr, stream) != cutlass::Status::kSuccess)
    throw Exception("cutlassFFNUpMish gemm failed");
}

template <typename ElementType, bool bias>
void fusedMHACutlass(void* output, void* q, void* k, void* v, void* skip,
                     int batch_size, int num_heads, int depth,
                     cudaStream_t stream, bool broadcast_bias) {
  ElementType* mha_q = (ElementType*)q;
  ElementType* mha_k = (ElementType*)k;
  ElementType* mha_v = (ElementType*)v;

  constexpr int kQueriesPerBlock = 64;
  constexpr int kKeysPerBlock = 64;
  constexpr bool kSingleValueIteration = true;

  using Attention =
      AttentionKernel<ElementType,          // scalar_t
                      cutlass::arch::Sm80,  // ArchTag
                      true,                 // Memory is aligned
                      kQueriesPerBlock, kKeysPerBlock, kSingleValueIteration,
                      false,  // Supports dropout
                      bias    // Supports bias
                      >;
  static_assert(
      !Attention::kNeedsOutputAccumulatorBuffer,
      "Unhandled case in cutlass MHA: needs output accumulator buffer");

  typename Attention::Params p;
  {  // set parameters
    p.query_ptr = mha_q;
    p.key_ptr = mha_k;
    p.value_ptr = mha_v;
    p.logsumexp_ptr = nullptr;  // Only needed for bw
    p.output_accum_ptr = nullptr;
    p.output_ptr = (ElementType*)output;
    p.attn_bias_ptr = (ElementType*)skip;

    p.scale = 1.0f / sqrt((float)depth);

    p.num_heads = num_heads;
    p.num_batches = batch_size;
    p.head_dim = depth;
    p.head_dim_value = depth;
    p.num_queries = 64;
    p.num_keys = 64;

    // All tensors are in BMHK shapes
    p.q_strideH = depth;
    p.k_strideH = depth;
    p.v_strideH = depth;
    p.q_strideM = depth * num_heads;
    p.k_strideM = depth * num_heads;
    p.v_strideM = depth * num_heads;
    p.q_strideB = p.q_strideM * 64;
    p.k_strideB = p.k_strideM * 64;
    p.v_strideB = p.v_strideM * 64;
    p.o_strideM = p.head_dim_value * p.num_heads;

    p.bias_strideH = 64 * 64;
    p.bias_strideM = 64;
    // broadcast_bias: the same (H,64,64) bias for every batch element (hero's
    // static geometry bias is batch-independent) -> strideB=0, no N-broadcast.
    p.bias_strideB = broadcast_bias ? 0 : num_heads * p.bias_strideH;
  }

  constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
  int smem_bytes = sizeof(typename Attention::SharedStorage);
  if (smem_bytes > 0xc000) {
    ReportCUDAErrors(cudaFuncSetAttribute(
        kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  }
  if (!Attention::check_supported(p)) {
    throw Exception("Unhandled case in cutlass MHA: check_supported failed.");
  }

  kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes, stream>>>(p);

  ReportCUDAErrors(cudaGetLastError());
}

template <typename DataType>
void fusedMHA(void* output, void* mha_q, void* mha_k, void* mha_v, void* skip,
              int batch_size, int num_heads, int depth, cudaStream_t stream,
              bool broadcast_bias) {
  if constexpr (std::is_same<DataType, float>::value) {
    throw Exception("Fused MHA is not supported for FP32.");
  }
#if LC0_CUDA_BF16_SUPPORTED
  else if constexpr (std::is_same<DataType, __nv_bfloat16>::value) {
    if (skip == nullptr) {
      fusedMHACutlass<cutlass::bfloat16_t, false>(
          output, mha_q, mha_k, mha_v, skip, batch_size, num_heads, depth,
          stream, broadcast_bias);
    } else {
      fusedMHACutlass<cutlass::bfloat16_t, true>(
          output, mha_q, mha_k, mha_v, skip, batch_size, num_heads, depth,
          stream, broadcast_bias);
    }
  }
#endif
  else if constexpr (std::is_same<DataType, half>::value) {
    if (skip == nullptr) {
      fusedMHACutlass<cutlass::half_t, false>(
          output, mha_q, mha_k, mha_v, skip, batch_size, num_heads, depth,
          stream, broadcast_bias);
    } else {
      fusedMHACutlass<cutlass::half_t, true>(
          output, mha_q, mha_k, mha_v, skip, batch_size, num_heads, depth,
          stream, broadcast_bias);
    }
  } else {
    throw Exception("Unsupported data type for Fused MHA.");
  }
}

template void fusedMHA<half>(void* output, void* mha_q, void* mha_k,
                             void* mha_v, void* skip, int batch_size,
                             int num_heads, int depth, cudaStream_t stream,
                             bool broadcast_bias);

#if LC0_CUDA_BF16_SUPPORTED
template void fusedMHA<__nv_bfloat16>(void* output, void* mha_q, void* mha_k,
                                      void* mha_v, void* skip, int batch_size,
                                      int num_heads, int depth,
                                      cudaStream_t stream, bool broadcast_bias);
#endif

template void fusedMHA<float>(void* output, void* mha_q, void* mha_k,
                              void* mha_v, void* skip, int batch_size,
                              int num_heads, int depth, cudaStream_t stream,
                              bool broadcast_bias);

}  // namespace cudnn_backend
}  // namespace lczero
