# Wiki leakage audit: infinity-wiki gfx942-technique-context @ 467eaa2

Classes: **self-reference** (the page names a test task, with baseline data only),
**solution-available** (the page gives a test task's fix or parameters),
**transfer** (general technique; no test task named).

## Pages added or changed in 467eaa2

| Page | Class | Why |
|---|---|---|
| technique-row-reduction-kernels | transfer | Online normalizer, HBM sweep counts, wave64 reduction primitive; no softmax kernel, no task named, no task parameters. Directly relevant to both softmax tasks. |
| technique-hip-mfma-gemm-tiling | transfer | FP32 vs. XF32 MFMA, vector vs. matrix peak, tile levels; one MFMA call as the only code. No task named. Directly relevant to the HIP GEMM task. |
| technique-gemm-tile-scheduling | transfer | Wave quantization, persistent / split-K / stream-K; worked examples use 1,000 and 400 tiles, not test shapes. Relevant to both GEMM tasks. |
| doc-rocm-triton-streamk | transfer | Quotes only the scheduling structure of AMD's stream-K GEMM, not its MAC loop. |
| kernel-rocmbench-triton-tflops (softmax section) | self-reference | Was solution-available: it explained why `naive_softmax` is slower than `softmax`. Now records both baselines only. |

## Existing pages that name the test tasks

| Page | Task | Class |
|---|---|---|
| kernel-rocmbench-triton-tflops | `hard/gemm`, `medium/naive_softmax` | self-reference: baseline TFLOP/s for the original shapes and the fraction of FP16 peak at the largest one |
| kernel-gemm-dense-matmul | `gpumode/SimpleMatmulModule` | self-reference: listed in the task table; small-shape numbers not tabulated |
| kernel-softmax | `gpumode/SoftmaxModule` | self-reference: latency of the original 4³–64³ cases; its open-questions note names single-pass vs. two-pass as unanalysed |

The HIP tasks run as `hip2hip/gpumode_large/*` with production-size cases, so
the original small-shape numbers on these pages do not describe the measured
workload.
