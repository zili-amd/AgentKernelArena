# Wiki search check: e7f8e84 -> 467eaa2

## triton2triton/rocmbench/hard/gemm (variant)

| Query | Before (top 5) | After (top 5; * = new page) |
|---|---|---|
| Triton fp16 GEMM MI300X tuning BLOCK_M BLOCK_N num_stages matrix_instr_nonkdim | technique-triton-amd-kernel-tuning, technique-moe-block-aligned-grouped-gemm, PR-3372, kernel-rocmbench-triton-tflops, technique-flash-attention-forward-loop | technique-triton-amd-kernel-tuning, technique-moe-block-aligned-grouped-gemm, PR-3372, kernel-rocmbench-triton-tflops, technique-gemm-tile-scheduling* |
| persistent GEMM stream-K split-K wave quantization 304 CUs | technique-fine-grained-quantization, PR-3881, PR-27111, kernel-gemm-dense-matmul, PR-394 | technique-gemm-tile-scheduling*, doc-rocm-triton-streamk*, doc-stream-k-paper*, technique-fine-grained-quantization, technique-hip-mfma-gemm-tiling* |
| GEMM output tiles not a multiple of CU count tail wave | technique-occupancy-tuning, technique-hip-latency-hiding, pattern-low-occupancy, technique-triton-amd-kernel-tuning, kernel-rocmbench-triton-tflops | technique-hip-mfma-gemm-tiling*, technique-gemm-tile-scheduling*, doc-rocm-triton-streamk*, technique-occupancy-tuning, technique-hip-latency-hiding |
| XCD-aware program id remap GEMM L2 locality | technique-xcd-aware-workgroup-remap, doc-aiter-pid-preprocessing, technique-moe-block-aligned-grouped-gemm, blog-lds-bank-conflict-ck-tile, kernel-fused-gemm-epilogue | technique-xcd-aware-workgroup-remap, doc-aiter-pid-preprocessing, doc-rocm-triton-streamk*, technique-gemm-tile-scheduling*, technique-moe-block-aligned-grouped-gemm |

## triton2triton/rocmbench/medium/naive_softmax (variant)

| Query | Before (top 5) | After (top 5; * = new page) |
|---|---|---|
| softmax Triton row reduction bandwidth bound MI300X | technique-triton-bandwidth-bound-kernels, kernel-rocmbench-triton-tflops, technique-flash-attention-forward-loop, technique-triton-amd-kernel-tuning, doc-xgmi-rccl-blog | technique-row-reduction-kernels*, technique-triton-bandwidth-bound-kernels, kernel-rocmbench-triton-tflops, technique-flash-attention-forward-loop, technique-triton-amd-kernel-tuning |
| online softmax single pass max and sum normalizer | technique-flash-attention-forward-loop, kernel-rocmbench-triton-tflops, kernel-softmax, kernel-fused-gemm-epilogue, PR-2073 | technique-row-reduction-kernels*, doc-online-softmax-paper*, technique-flash-attention-forward-loop, kernel-rocmbench-triton-tflops, kernel-softmax |
| tl.max tl.sum reduction lowering wave64 | PR-7321, PR-48788, PR-2047, kernel-softmax, kernel-rocmbench-triton-tflops | technique-row-reduction-kernels*, PR-7321, PR-48788, PR-2047, doc-hip-cpp-language-extensions* |
| softmax long rows vocab 32000 block size num_warps | technique-flash-attention-forward-loop, PR-1978, kernel-rocmbench-triton-tflops, technique-moe-block-aligned-grouped-gemm, technique-triton-amd-kernel-tuning | technique-row-reduction-kernels*, technique-flash-attention-forward-loop, doc-hip-cpp-language-extensions*, PR-1978, kernel-rocmbench-triton-tflops |

## hip2hip/gpumode_large/SimpleMatmulModule

| Query | Before (top 5) | After (top 5; * = new page) |
|---|---|---|
| HIP fp32 GEMM MFMA gfx942 v_mfma_f32_32x32x2f32 | kernel-gemm-dense-matmul, hw-fp8-fnuz-ocp, hw-mfma-operand-layout-cdna3, technique-hip-latency-hiding, technique-triton-amd-kernel-tuning | technique-hip-mfma-gemm-tiling*, kernel-gemm-dense-matmul, hw-fp8-fnuz-ocp, hw-mfma-operand-layout-cdna3, technique-hip-latency-hiding |
| xf32 precision fp32 matmul MI300X | blog-matrix-cores-cdna, kernel-gemm-dense-matmul, doc-hip-low-fp-types, pattern-low-occupancy, doc-datasheet-mi300x | technique-hip-mfma-gemm-tiling*, blog-matrix-cores-cdna, kernel-gemm-dense-matmul, doc-hip-low-fp-types, pattern-low-occupancy |
| HIP GEMM LDS tiling register blocking wave tile | technique-hip-latency-hiding, blog-lds-bank-conflict-ck-tile, technique-moe-block-aligned-grouped-gemm, doc-hip-programming-model, hw-mfma-operand-layout-cdna3 | technique-hip-mfma-gemm-tiling*, technique-gemm-tile-scheduling*, technique-hip-latency-hiding, technique-row-reduction-kernels*, blog-lds-bank-conflict-ck-tile |
| FP32 vector peak vs matrix peak MI300X | blog-matrix-cores-cdna, kernel-gemm-dense-matmul, technique-triton-amd-kernel-tuning, hw-fp8-fnuz-ocp, doc-datasheet-mi300x | technique-hip-mfma-gemm-tiling*, blog-matrix-cores-cdna, kernel-gemm-dense-matmul, technique-triton-amd-kernel-tuning, hw-fp8-fnuz-ocp |

## hip2hip/gpumode_large/SoftmaxModule

| Query | Before (top 5) | After (top 5; * = new page) |
|---|---|---|
| HIP warp shuffle reduction wave64 __shfl_xor | doc-hip-low-fp-types, PR-3491, technique-hip-latency-hiding, doc-hip-programming-model, PR-7321 | technique-row-reduction-kernels*, doc-hip-cpp-language-extensions*, doc-hip-amd-warp-functions*, doc-hip-low-fp-types, technique-hip-mfma-gemm-tiling* |
| block reduce shared memory warpSize 64 | PR-1978, PR-37132, doc-handbook-workload, doc-hip-programming-model, PR-1604 | doc-hip-cpp-language-extensions*, technique-row-reduction-kernels*, PR-1978, PR-37132, doc-handbook-workload |
| softmax HIP row reduction vectorized float4 loads | technique-hip-latency-hiding, kernel-softmax, technique-flash-attention-forward-loop, technique-triton-bandwidth-bound-kernels, kernel-rocmbench-triton-tflops | technique-row-reduction-kernels*, technique-hip-mfma-gemm-tiling*, technique-hip-latency-hiding, kernel-softmax, doc-hip-amd-warp-functions* |
| DPP cross-lane reduction gfx942 | PR-962, PR-48788, PR-1258, PR-50455, kernel-normalization-layernorm-rmsnorm | technique-row-reduction-kernels*, doc-hip-amd-warp-functions*, doc-hip-cpp-language-extensions*, PR-962, PR-48788 |

