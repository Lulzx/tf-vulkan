# IREE spike (task 009 / gate F6) — desktop harness result

- `iree-compile` (pip, 20241104.1068) compiles `matmul.mlir` (linalg.matmul
  4x4 f32) to a VMFB for `--iree-hal-target-backends=vulkan-spirv`: PASS.
- `iree-run-module --device=vulkan` FAILS: the pip `iree-runtime` wheel
  registers only `local-sync`/`local-task` CPU drivers; no Vulkan HAL.
- Conclusion: F6 execution requires an IREE runtime built from source
  with the Vulkan HAL enabled (or a vendor runtime build). The VMFB
  artifact and compile path are proven; do not claim IREE execution.
- Android implication: budget a source build of the pinned IREE runtime
  for arm64 with Vulkan HAL before scheduling task 011.
