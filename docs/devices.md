# Devices and tiers (spec §5)

| Tier | Requirement | Path |
|---|---|---|
| A | Vulkan 1.3 + timelineSemaphore + scalarBlockLayout + synchronization2 + compute queue | Native eager + optional IREE (Descriptor-Set ABI default) |
| B | Compute queue, 1.3 baseline incomplete | Native SPIR-V only |
| C | No supported compute | Plugin disabled, clear diagnostic, CPU TF remains |

Probe a device with `tfvk_probe` (add `--json`); record model, SoC, driver,
Android release, Vulkan features, memory types, FP16, and the measured
watchdog limit here per device. Never infer support from marketing names.
Target matrix (§11): modern Adreno, Mali, one older Vulkan phone (negative
test), desktop harness for debugging only.

## Recorded device: Android emulator (NOT a hardware gate pass)
- sdk_gphone64_arm64, Android 36 arm64-v8a image, SwiftShader (LLVM) Vulkan 1.3.0
- `tfvk_probe`: Tier A features (timelineSemaphore, scalarBlockLayout,
  synchronization2 present; shaderFloat16 absent; pushDescriptor absent)
- `tfvk_tests` (arm64 NDK build, adb): ALL PASS — MatMul odd shapes, chain,
  pointwise/threshold, Conv2D, FP16 correctly skipped, 10k alloc cycles
- SwiftShader is a software rasterizer: this validates the Android
  toolchain, Bionic linking, Vulkan loader, and kernel correctness on
  Android. It does NOT satisfy the spec's real-device gates (§4/§11):
  Adreno/Mali performance, watchdog limits, and UMA behavior still need
  physical phones.

## Recorded device: Google Pixel 9 (REAL DEVICE, Tensor G4 / Mali-G715)
- Pixel 9, Android 17 (API 37), arm64-v8a, build DP11.261002.005
- GPU: Mali-G715 MC7, Vulkan 1.4.349, driver 234881024
- Features: compute queue, timelineSemaphore, scalarBlockLayout,
  synchronization2, shaderFloat16, pushDescriptor all present;
  VK_EXT_memory_budget absent (budget falls back to 30% of 11.1 GiB heap)
- UMA: memory type 1 is DEVICE_LOCAL|HOST_VISIBLE|COHERENT (type 3 cached)
- `tfvk_tests` (NDK arm64): ALL PASS incl. FP16 matmul+cast, Conv2D,
  10k alloc cycles; `tfvk_fuzz_shapes`: PASS (60 shapes)
- `tfvk_bench` (vs single-thread CPU reference in the same binary):
  128^3 1.09 ms (2.4x), 512^3 7.0 ms (16.5x), 1024^3 34.9 ms (54.7x)
- This satisfies F3/F4 on real hardware. F0/F1/F5 still require the host
  app's TensorFlow Android build running on this phone.
