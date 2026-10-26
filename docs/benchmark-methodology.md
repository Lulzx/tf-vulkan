# Benchmark methodology (spec §11)

Warm pipeline first (report cold separately), ≥30 samples when affordable,
median + p10/p90, paired A/B order, outputs consumed via sync so async work
is not mistaken for completion. Report kernel, submission, H2D/D2H,
allocator, and whole-call time separately; Vulkan timestamps only when
calibrated. Every result carries correctness + fallback counters, strict
placement, and matched precision. `tfvk_bench` implements this for MatMul
on the native harness (15 samples); TF A/B via `benchmarks/matmul.py`.
