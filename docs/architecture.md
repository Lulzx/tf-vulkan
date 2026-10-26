# Architecture

Implements spec §3/§6. One in-process plugin (`libtf_vulkan.so`) loaded by
TensorFlow; native Vulkan eager engine first, optional IREE compiled regions
second. TensorFlow owns Python, autograd, and CPU kernels.

## Runtime (`src/runtime`)
- `device`: Vulkan instance/physical-device selection, feature probe
  (timelineSemaphore, scalarBlockLayout, synchronization2, FP16 gate),
  Tier A/B/C classification (§5), persistent pipeline cache keyed by
  `pipelineCacheUUID` + vendor/device (§12).
- `allocator`: pooled `VkBuffer` allocator. Budget from
  `VK_EXT_memory_budget`, else 30% of heap (§6.1). Buffers recycle only
  after their last-use completion token passes (§6.2). UMA-preferred
  (`DEVICE_LOCAL|HOST_VISIBLE`) memory selection.
- `stream`: deferred submission (§6.3). Ops record into one open command
  buffer with compute-to-compute pipeline barriers. Mandatory flush
  triggers: `BlockHostUntilDone`/`synchronize`, `RecordEvent`, `MemcpyDtoh`
  on pending buffers, 64-dispatch threshold, workgroup watchdog threshold.
  Per-batch ring descriptor pools reset after their fence signals; scalars
  go via push constants (≤128 B checked).

## Kernels (`src/kernels`, `src/shaders`)
Offline-compiled SPIR-V only — no runtime `shaderc`/`glslang` (contrast
desktop Vulten, §1). K0: Identity/binary pointwise, BiasAdd, naive+tiled
MatMul, single-axis Sum/Mean/Max. K1: stable row Softmax, Sigmoid/Tanh/
Exp/Log/Sqrt, DivNoNan, Pow, Maximum, SquaredDifference. CPU float32/
float64 references in `cpu_reference.cpp` are the correctness oracle.

## TensorFlow bridge (`src/tensorflow_plugin`)
`SE_InitPlugin` + `TF_InitKernel` against pinned TF C headers only
(`TFVK_BUILD_TF_PLUGIN=ON`). Device type `GPU`, subtype `VULKAN` (D1
provisional; `TF_VULKAN_DEVICE_TYPE` overrides pending F1b).

## Compiled graph path (`src/graph`, `src/iree_bridge`)
Gated research (F6/F7). Partitioning rules and the sha256 cache key are
implemented and tested in `tf_vulkan.partition_ops`/`cache_key`; lowering
and VMFB execution are not claimed until task 010 passes.
