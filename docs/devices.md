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
