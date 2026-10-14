# Wiki search check: 467eaa2 -> 0523818

## triton2triton/rocmbench/hard/gemm (variant)

| Query | Before (top 5) | After (top 5; * = new page) |
|---|---|---|
| Triton fp16 GEMM MI300X tuning BLOCK_M BLOCK_N num_stages matrix_instr_nonkdim | technique-triton-amd-kernel-tuning, technique-moe-block-aligned-grouped-gemm, PR-3372, kernel-rocmbench-triton-tflops, technique-gemm-tile-scheduling | technique-triton-amd-kernel-tuning, technique-moe-block-aligned-grouped-gemm, kernel-rocmbench-triton-tflops, PR-3372, technique-gemm-tile-scheduling |
| persistent GEMM stream-K split-K wave quantization 304 CUs | technique-gemm-tile-scheduling, doc-rocm-triton-streamk, doc-stream-k-paper, technique-fine-grained-quantization, technique-hip-mfma-gemm-tiling | technique-gemm-tile-scheduling, doc-rocm-triton-streamk, doc-stream-k-paper, technique-split-k*, technique-fine-grained-quantization |
| GEMM output tiles not a multiple of CU count tail wave | technique-hip-mfma-gemm-tiling, technique-gemm-tile-scheduling, doc-rocm-triton-streamk, technique-occupancy-tuning, technique-hip-latency-hiding | technique-hip-mfma-gemm-tiling, technique-gemm-tile-scheduling, doc-rocm-triton-streamk, technique-occupancy-tuning, pattern-low-mfma-utilization* |
| XCD-aware program id remap GEMM L2 locality | technique-xcd-aware-workgroup-remap, doc-aiter-pid-preprocessing, doc-rocm-triton-streamk, technique-gemm-tile-scheduling, technique-moe-block-aligned-grouped-gemm | technique-xcd-aware-workgroup-remap, doc-aiter-pid-preprocessing, doc-rocm-triton-streamk, technique-gemm-tile-scheduling, technique-moe-block-aligned-grouped-gemm |

## triton2triton/rocmbench/medium/naive_softmax (variant)

| Query | Before (top 5) | After (top 5; * = new page) |
|---|---|---|
| softmax Triton row reduction bandwidth bound MI300X | technique-row-reduction-kernels, technique-triton-bandwidth-bound-kernels, kernel-rocmbench-triton-tflops, technique-flash-attention-forward-loop, technique-triton-amd-kernel-tuning | technique-row-reduction-kernels, technique-triton-bandwidth-bound-kernels, kernel-rocmbench-triton-tflops, pattern-bandwidth-bound*, technique-flash-attention-forward-loop |
| online softmax single pass max and sum normalizer | technique-row-reduction-kernels, doc-online-softmax-paper, technique-flash-attention-forward-loop, kernel-rocmbench-triton-tflops, kernel-softmax | technique-row-reduction-kernels, doc-online-softmax-paper, technique-flash-attention-forward-loop, kernel-rocmbench-triton-tflops, kernel-softmax |
| tl.max tl.sum reduction lowering wave64 | technique-row-reduction-kernels, PR-7321, PR-48788, PR-2047, doc-hip-cpp-language-extensions | technique-row-reduction-kernels, PR-7321, PR-48788, PR-2047, kernel-rocmbench-triton-tflops |
| softmax long rows vocab 32000 block size num_warps | technique-row-reduction-kernels, technique-flash-attention-forward-loop, doc-hip-cpp-language-extensions, PR-1978, kernel-rocmbench-triton-tflops | technique-row-reduction-kernels, technique-flash-attention-forward-loop, doc-hip-cpp-language-extensions, PR-1978, kernel-rocmbench-triton-tflops |

## hip2hip/gpumode_large/SimpleMatmulModule

| Query | Before (top 5) | After (top 5; * = new page) |
|---|---|---|
| HIP fp32 GEMM MFMA gfx942 v_mfma_f32_32x32x2f32 | technique-hip-mfma-gemm-tiling, kernel-gemm-dense-matmul, hw-fp8-fnuz-ocp, hw-mfma-operand-layout-cdna3, technique-hip-latency-hiding | technique-hip-mfma-gemm-tiling, kernel-gemm-dense-matmul, hw-fp8-fnuz-ocp, hw-mfma-operand-layout-cdna3, technique-hip-latency-hiding |
| xf32 precision fp32 matmul MI300X | technique-hip-mfma-gemm-tiling, blog-matrix-cores-cdna, kernel-gemm-dense-matmul, doc-hip-low-fp-types, pattern-low-occupancy | technique-hip-mfma-gemm-tiling, kernel-gemm-dense-matmul, blog-matrix-cores-cdna, doc-hip-low-fp-types, pattern-low-occupancy |
| HIP GEMM LDS tiling register blocking wave tile | technique-hip-mfma-gemm-tiling, technique-gemm-tile-scheduling, technique-hip-latency-hiding, technique-row-reduction-kernels, blog-lds-bank-conflict-ck-tile | technique-hip-mfma-gemm-tiling, technique-gemm-tile-scheduling, technique-hip-latency-hiding, technique-row-reduction-kernels, blog-lds-bank-conflict-ck-tile |
| FP32 vector peak vs matrix peak MI300X | technique-hip-mfma-gemm-tiling, blog-matrix-cores-cdna, kernel-gemm-dense-matmul, technique-triton-amd-kernel-tuning, hw-fp8-fnuz-ocp | technique-hip-mfma-gemm-tiling, blog-matrix-cores-cdna, kernel-gemm-dense-matmul, technique-triton-amd-kernel-tuning, hw-fp8-fnuz-ocp |

## hip2hip/gpumode_large/SoftmaxModule

| Query | Before (top 5) | After (top 5; * = new page) |
|---|---|---|
| HIP warp shuffle reduction wave64 __shfl_xor | technique-row-reduction-kernels, doc-hip-cpp-language-extensions, doc-hip-amd-warp-functions, doc-hip-low-fp-types, technique-hip-mfma-gemm-tiling | technique-row-reduction-kernels, doc-hip-cpp-language-extensions, doc-hip-amd-warp-functions, doc-hip-low-fp-types, technique-hip-mfma-gemm-tiling |
| block reduce shared memory warpSize 64 | doc-hip-cpp-language-extensions, technique-row-reduction-kernels, PR-1978, PR-37132, doc-handbook-workload | doc-hip-cpp-language-extensions, technique-row-reduction-kernels, PR-1978, pattern-bandwidth-bound*, PR-37132 |
| softmax HIP row reduction vectorized float4 loads | technique-row-reduction-kernels, technique-hip-mfma-gemm-tiling, technique-hip-latency-hiding, kernel-softmax, doc-hip-amd-warp-functions | technique-row-reduction-kernels, technique-hip-mfma-gemm-tiling, technique-hip-latency-hiding, kernel-softmax, doc-hip-amd-warp-functions |
| DPP cross-lane reduction gfx942 | technique-row-reduction-kernels, doc-hip-amd-warp-functions, doc-hip-cpp-language-extensions, PR-962, PR-48788 | technique-row-reduction-kernels, doc-hip-amd-warp-functions, doc-hip-cpp-language-extensions, PR-962, PR-48788 |

## Replay of logged calls

13 of 31 search calls change their top 5; 8 of 16 grep calls change their file set.

New pages entering a search top 5: pattern-bandwidth-bound (7), pattern-low-mfma-utilization (2), hw-agpr (2), technique-software-pipelining (1), technique-mfma-layout-selection (1), hw-infinity-cache (1)

Pages leaving a search top 5: hw-fp8-fnuz-ocp (3), technique-hip-latency-hiding (2), pr-3372 (1), technique-hip-mfma-gemm-tiling (1), technique-lds-bank-conflict-avoidance (1), pr-7820 (1), hw-mfma-operand-layout-cdna3 (1), technique-xcd-aware-workgroup-remap (1), doc-datasheet-mi300a (1), technique-gemm-tile-scheduling (1), doc-clang-amdgpu-builtins (1), blog-lds-bank-conflict-ck-tile (1), kernel-rocmbench-triton-tflops (1), technique-row-reduction-kernels (1)

New pages added to grep results: hw-agpr (3), hw-wavefront64 (2), technique-pipeline-stages (2), hw-infinity-cache (2), pattern-bandwidth-bound (1)

### pilot r0 kf_wiki

| Role | Tool | Query or pattern | Before (top 5) | After (top 5; * = new page) |
|---|---|---|---|---|
| analysis | search | `fp32 GEMM register tiling LDS bandwidth bottleneck MI300X` | technique-hip-mfma-gemm-tiling, technique-moe-block-aligned-grouped-gemm, kernel-gemm-dense-matmul, technique-triton-bandwidth-bound-kernels, pr-3372 | technique-hip-mfma-gemm-tiling, technique-moe-block-aligned-grouped-gemm, pattern-bandwidth-bound*, kernel-gemm-dense-matmul, technique-triton-bandwidth-bound-kernels |
| analysis | search | `MFMA fp32 v_mfma_f32_16x16x4_f32 matrix core throughput` | technique-hip-mfma-gemm-tiling, blog-matrix-cores-cdna, hw-mfma, doc-amd-matrix-instruction-calculator, hw-mfma-operand-layout-cdna3 | technique-hip-mfma-gemm-tiling, blog-matrix-cores-cdna, hw-mfma, doc-amd-matrix-instruction-calculator, hw-mfma-operand-layout-cdna3 |
| orchestration | search | `fp32 SGEMM register tiling LDS double buffering gfx942` | technique-hip-latency-hiding, technique-hip-mfma-gemm-tiling, technique-lds-bank-conflict-avoidance, hw-mfma-operand-layout-cdna3, kernel-gemm-dense-matmul | technique-hip-latency-hiding, technique-hip-mfma-gemm-tiling, technique-lds-bank-conflict-avoidance, hw-mfma-operand-layout-cdna3, kernel-gemm-dense-matmul |
| specialist memory | search | `LDS layout padding XOR swizzle ds_read_b128 register tiling fp32 GEMM` | technique-lds-bank-conflict-avoidance, technique-hip-mfma-gemm-tiling, technique-moe-block-aligned-grouped-gemm, blog-lds-bank-conflict-ck-tile, hw-mfma-operand-layout-cdna3 | technique-lds-bank-conflict-avoidance, technique-hip-mfma-gemm-tiling, technique-moe-block-aligned-grouped-gemm, blog-lds-bank-conflict-ck-tile, hw-mfma-operand-layout-cdna3 |
| specialist compute | search | `fp32 MFMA 16x16x4 f32 gfx942 matrix core GEMM` | technique-hip-mfma-gemm-tiling, blog-matrix-cores-cdna, hw-mfma-operand-layout-cdna3, technique-triton-amd-kernel-tuning, hw-fp8-fnuz-ocp | technique-hip-mfma-gemm-tiling, blog-matrix-cores-cdna, kernel-gemm-dense-matmul, hw-mfma-operand-layout-cdna3, technique-triton-amd-kernel-tuning |
| specialist memory | search | `MI300X XCD round-robin workgroup scheduling L2 locality tile swizzle` | technique-xcd-aware-workgroup-remap, technique-gemm-tile-scheduling, doc-aiter-pid-preprocessing, technique-moe-block-aligned-grouped-gemm, doc-rocm-triton-streamk | technique-xcd-aware-workgroup-remap, technique-gemm-tile-scheduling, doc-aiter-pid-preprocessing, technique-moe-block-aligned-grouped-gemm, doc-rocm-triton-streamk |
| specialist algorithm | search | `fp32 GEMM register tiling thread tile LDS gfx942 SGEMM` | technique-hip-mfma-gemm-tiling, kernel-gemm-dense-matmul, technique-gemm-tile-scheduling, technique-moe-block-aligned-grouped-gemm, blog-lds-bank-conflict-ck-tile | technique-hip-mfma-gemm-tiling, kernel-gemm-dense-matmul, technique-gemm-tile-scheduling, technique-moe-block-aligned-grouped-gemm, blog-lds-bank-conflict-ck-tile |
| specialist algorithm | search | `split-K stream-K tile selection workgroup count CU occupancy launch` | technique-gemm-tile-scheduling, pattern-low-occupancy, technique-occupancy-tuning, technique-triton-amd-kernel-tuning, technique-hip-latency-hiding | technique-gemm-tile-scheduling, pattern-low-occupancy, technique-occupancy-tuning, technique-triton-amd-kernel-tuning, pattern-low-mfma-utilization* |
| specialist compute | search | `occupancy VGPR waves per SIMD gfx942 launch_bounds register budget` | technique-triton-amd-kernel-tuning, technique-hip-latency-hiding, pattern-low-occupancy, hw-mfma-operand-layout-cdna3, technique-hip-mfma-gemm-tiling | technique-triton-amd-kernel-tuning, technique-hip-latency-hiding, pattern-low-occupancy, hw-mfma-operand-layout-cdna3, hw-agpr* |
| specialist compute | search | `double buffering LDS software pipelining s_waitcnt barrier gfx942 global prefetch` | technique-hip-latency-hiding, technique-hip-mfma-gemm-tiling, pattern-lds-bound, doc-llvm-amdgpu-usage, technique-lds-bank-conflict-avoidance | technique-hip-latency-hiding, technique-hip-mfma-gemm-tiling, technique-software-pipelining*, pattern-lds-bound, doc-llvm-amdgpu-usage |
| specialist compute | grep | `(512 VGPR|VGPRs per|waves_per_eu|occupancy formula|accvgpr|512-entry)` | 6 files | 8 files; added hw-agpr*, hw-wavefront64*; removed — |
| specialist memory | grep | `(cache line|cacheline|128[- ]byte line|64 B line)` | 0 files | 0 files; added —; removed — |
| specialist memory | grep | `direct-to-LDS|global_load_lds|buffer_load_dword lds` | 11 files | 11 files; added —; removed — |
| specialist algorithm | grep | `(1024 thread blocks|≥1024|saturate 304)` | 1 files | 1 files; added —; removed — |
| orchestration lane 1 synthesis | search | `launch_bounds waves_per_eu VGPR occupancy gfx942 register tiling GEMM` | technique-triton-amd-kernel-tuning, technique-hip-latency-hiding, technique-hip-mfma-gemm-tiling, pattern-low-occupancy, hw-mfma-operand-layout-cdna3 | technique-triton-amd-kernel-tuning, technique-hip-latency-hiding, technique-hip-mfma-gemm-tiling, pattern-low-occupancy, hw-mfma-operand-layout-cdna3 |
| orchestration lane 1 synthesis | grep | `(64 ?KB|65536).{0,60}LDS|LDS.{0,40}(64 ?KB)` | 2 files | 3 files; added technique-pipeline-stages*; removed — |
| orchestration lane 2 synthesis | search | `MFMA operand layout fp32 16x16x4 CDNA3 lane mapping` | hw-mfma-operand-layout-cdna3, doc-amd-matrix-instruction-calculator, technique-hip-mfma-gemm-tiling, blog-matrix-cores-cdna, pr-7820 | hw-mfma-operand-layout-cdna3, doc-amd-matrix-instruction-calculator, technique-hip-mfma-gemm-tiling, technique-mfma-layout-selection*, blog-matrix-cores-cdna |
| orchestration lane 2 synthesis | search | `MFMA GEMM tiling HIP intrinsic AGPR accumulator` | technique-hip-mfma-gemm-tiling, hw-mfma-operand-layout-cdna3, doc-llvm-amdgpu-usage, technique-hip-latency-hiding, hw-fp8-fnuz-ocp | technique-hip-mfma-gemm-tiling, hw-mfma-operand-layout-cdna3, hw-agpr*, doc-llvm-amdgpu-usage, kernel-gemm-dense-matmul |

### pilot r1 kf_wiki

| Role | Tool | Query or pattern | Before (top 5) | After (top 5; * = new page) |
|---|---|---|---|---|
| analysis | search | `fp32 GEMM optimization MI300X gfx942 register tiling shared memory` | technique-hip-mfma-gemm-tiling, kernel-gemm-dense-matmul, technique-hip-latency-hiding, blog-matrix-cores-cdna, hw-mfma-operand-layout-cdna3 | technique-hip-mfma-gemm-tiling, kernel-gemm-dense-matmul, technique-hip-latency-hiding, pattern-bandwidth-bound*, blog-matrix-cores-cdna |
| analysis | grep | `v_pk_fma_f32|packed math|packed FP32|pk_fma` | 2 files | 2 files; added —; removed — |
| orchestration | search | `fp32 GEMM register tiling LDS double buffering gfx942` | technique-hip-mfma-gemm-tiling, technique-hip-latency-hiding, kernel-gemm-dense-matmul, technique-moe-block-aligned-grouped-gemm, pr-3372 | technique-hip-mfma-gemm-tiling, technique-hip-latency-hiding, kernel-gemm-dense-matmul, technique-moe-block-aligned-grouped-gemm, pr-3372 |
| orchestration | search | `tile scheduling workgroup count XCD swizzle split-K MI300X` | technique-gemm-tile-scheduling, technique-xcd-aware-workgroup-remap, doc-aiter-pid-preprocessing, doc-rocm-triton-streamk, technique-moe-block-aligned-grouped-gemm | technique-gemm-tile-scheduling, technique-xcd-aware-workgroup-remap, doc-aiter-pid-preprocessing, doc-rocm-triton-streamk, technique-moe-block-aligned-grouped-gemm |
| specialist memory | search | `GEMM LDS tiling global load float4 vectorization bank conflicts fp32` | technique-hip-mfma-gemm-tiling, blog-lds-bank-conflict-ck-tile, pattern-lds-bound, technique-triton-bandwidth-bound-kernels, technique-lds-bank-conflict-avoidance | technique-hip-mfma-gemm-tiling, blog-lds-bank-conflict-ck-tile, pattern-lds-bound, technique-triton-bandwidth-bound-kernels, technique-lds-bank-conflict-avoidance |
| specialist memory | search | `MI300X L2 cache bandwidth Infinity Cache MALL per XCD` | doc-whitepaper-cdna3, doc-whitepaper-cdna4, doc-datasheet-mi300x, technique-xcd-aware-workgroup-remap, doc-datasheet-mi300a | hw-infinity-cache*, pattern-bandwidth-bound*, doc-whitepaper-cdna3, doc-whitepaper-cdna4, doc-datasheet-mi300x |
| specialist memory | grep | `(Infinity Cache|L2 .*(bandwidth|TB/s)|peak.*TB/s)` | 10 files | 12 files; added hw-infinity-cache*, pattern-bandwidth-bound*; removed — |
| specialist algorithm | search | `fp32 SGEMM register tiling v_pk_fma_f32 packed math gfx942` | technique-hip-mfma-gemm-tiling, hw-mfma-operand-layout-cdna3, technique-hip-latency-hiding, kernel-gemm-dense-matmul, hw-fp8-fnuz-ocp | technique-hip-mfma-gemm-tiling, hw-mfma-operand-layout-cdna3, technique-hip-latency-hiding, kernel-gemm-dense-matmul, hw-fp8-fnuz-ocp |
| specialist algorithm | grep | `FORGE_SWEEP|sweep_const` | 0 files | 0 files; added —; removed — |
| specialist compute | grep | `(LDS bandwidth|128 bytes/clock|bytes per clock|ds_read_b128|LDS_IDX_ACTIVE)` | 4 files | 5 files; added hw-lds; removed — |
| specialist compute | search | `packed fp32 v_pk_fma_f32 register tiling SGEMM gfx942` | technique-hip-mfma-gemm-tiling, hw-mfma-operand-layout-cdna3, kernel-gemm-dense-matmul, technique-hip-latency-hiding, hw-fp8-fnuz-ocp | technique-hip-mfma-gemm-tiling, hw-mfma-operand-layout-cdna3, kernel-gemm-dense-matmul, technique-hip-latency-hiding, hw-fp8-fnuz-ocp |
| specialist compute | search | `occupancy VGPR budget waves per SIMD gfx942 512 registers` | technique-triton-amd-kernel-tuning, pattern-low-occupancy, technique-hip-latency-hiding, hw-mfma-operand-layout-cdna3, doc-handbook-workload | technique-triton-amd-kernel-tuning, pattern-low-occupancy, technique-hip-latency-hiding, hw-mfma-operand-layout-cdna3, doc-handbook-workload |
| specialist compute | grep | `(banks|LDS).*(per clock|bandwidth|cycle)` | 2 files | 3 files; added technique-pipeline-stages*; removed — |
| specialist compute | grep | `(512 VGPR|vgprs?/wave|waves per SIMD|occupancy formula|256 ArchVGPR)` | 3 files | 5 files; added hw-agpr*, hw-wavefront64*; removed — |
| specialist memory | grep | `(L2 (read )?bandwidth|channels|B/clk|bytes per clock|17 TB)` | 17 files | 17 files; added —; removed — |

### pilot r2 kf_wiki

| Role | Tool | Query or pattern | Before (top 5) | After (top 5; * = new page) |
|---|---|---|---|---|
| analysis | search | `LDS bandwidth per CU and FP32 vector peak MI300X gfx942` | technique-hip-mfma-gemm-tiling, pattern-low-occupancy, technique-triton-bandwidth-bound-kernels, technique-triton-amd-kernel-tuning, technique-gemm-tile-scheduling | technique-hip-mfma-gemm-tiling, pattern-low-occupancy, technique-triton-bandwidth-bound-kernels, pattern-bandwidth-bound*, technique-triton-amd-kernel-tuning |
| analysis | search | `dense fp32 matmul kernel achieved TFLOPS register tiling naive shared memory GEMM speedup` | kernel-rocmbench-triton-tflops, kernel-gemm-dense-matmul, kernel-fused-gemm-epilogue, kernel-vllm-decode-paged-attention, kernel-normalization-layernorm-rmsnorm | kernel-rocmbench-triton-tflops, kernel-gemm-dense-matmul, kernel-fused-gemm-epilogue, kernel-vllm-decode-paged-attention, kernel-normalization-layernorm-rmsnorm |
| analysis | search | `fp32 MFMA 16x16x4 f32 GEMM tiling throughput` | technique-hip-mfma-gemm-tiling, kernel-gemm-dense-matmul, doc-clang-amdgpu-builtins, hw-fp8-fnuz-ocp, kernel-rocmbench-triton-tflops | technique-hip-mfma-gemm-tiling, pattern-low-mfma-utilization*, kernel-gemm-dense-matmul, pattern-bandwidth-bound*, kernel-rocmbench-triton-tflops |
| analysis | search | `register micro-tiling LDS bytes per FLOP shared memory GEMM thread tile` | technique-hip-mfma-gemm-tiling, technique-gemm-tile-scheduling, technique-moe-block-aligned-grouped-gemm, doc-rocm-triton-streamk, blog-lds-bank-conflict-ck-tile | technique-hip-mfma-gemm-tiling, technique-gemm-tile-scheduling, pattern-bandwidth-bound*, technique-moe-block-aligned-grouped-gemm, doc-rocm-triton-streamk |
| orchestration | search | `fp32 GEMM register micro-tiling LDS bytes per FLOP MI300X` | technique-hip-mfma-gemm-tiling, technique-moe-block-aligned-grouped-gemm, technique-gemm-tile-scheduling, kernel-rocmbench-triton-tflops, technique-row-reduction-kernels | technique-hip-mfma-gemm-tiling, technique-moe-block-aligned-grouped-gemm, technique-gemm-tile-scheduling, pattern-bandwidth-bound*, kernel-gemm-dense-matmul |
| specialist algorithm | search | `fp32 GEMM register micro-tiling HIP tiling SGEMM` | technique-hip-mfma-gemm-tiling, kernel-gemm-dense-matmul, technique-hip-latency-hiding, kernel-fused-gemm-epilogue, kernel-rocmbench-triton-tflops | technique-hip-mfma-gemm-tiling, kernel-gemm-dense-matmul, technique-hip-latency-hiding, kernel-fused-gemm-epilogue, kernel-rocmbench-triton-tflops |
| specialist memory | search | `LDS staging global_load_dwordx4 double buffering GEMM tiling fp32` | technique-hip-mfma-gemm-tiling, technique-hip-latency-hiding, kernel-gemm-dense-matmul, technique-moe-block-aligned-grouped-gemm, pr-3372 | technique-hip-mfma-gemm-tiling, technique-hip-latency-hiding, kernel-gemm-dense-matmul, technique-moe-block-aligned-grouped-gemm, pr-3372 |
| specialist compute | search | `v_pk_fma_f32 packed fp32 FMA gfx942 dual issue peak vector fp32` | technique-hip-mfma-gemm-tiling, kernel-gemm-dense-matmul, hw-fp8-fnuz-ocp, hw-mfma-operand-layout-cdna3, technique-triton-amd-kernel-tuning | technique-hip-mfma-gemm-tiling, kernel-gemm-dense-matmul, hw-fp8-fnuz-ocp, hw-mfma-operand-layout-cdna3, technique-triton-amd-kernel-tuning |
| specialist compute | search | `register tiling SGEMM micro-tile 128x128 LDS double buffering HIP` | technique-hip-latency-hiding, technique-hip-mfma-gemm-tiling, technique-row-reduction-kernels, doc-hip-programming-model, hw-mfma-operand-layout-cdna3 | technique-hip-latency-hiding, technique-hip-mfma-gemm-tiling, technique-row-reduction-kernels, doc-hip-programming-model, hw-mfma-operand-layout-cdna3 |
| specialist compute | grep | `pk_fma|packed math|v_pk_|dual.issue` | 1 files | 1 files; added —; removed — |
| specialist compute | grep | `512.*(VGPR|register file)|waves per SIMD|accum_offset|AGPR.*unified` | 3 files | 4 files; added hw-agpr*; removed — |
| specialist compute | search | `occupancy tuning VGPR budget waves per SIMD gfx942 formula` | technique-triton-amd-kernel-tuning, pattern-low-occupancy, technique-hip-latency-hiding, technique-occupancy-tuning, hw-mfma-operand-layout-cdna3 | technique-triton-amd-kernel-tuning, pattern-low-occupancy, technique-hip-latency-hiding, technique-occupancy-tuning, hw-mfma-operand-layout-cdna3 |
| specialist memory | grep | `(L2|Infinity Cache|LLC).{0,60}(TB/s|GB/s|bandwidth)` | 1 files | 2 files; added hw-infinity-cache*; removed — |
| specialist algorithm | grep | `pk_fma|packed (FP32|fp32)|v_pk_mul` | 1 files | 1 files; added —; removed — |

