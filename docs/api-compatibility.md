# API compatibility manifest (tasks 000/001/001b)

## Verified: desktop harness, TensorFlow 2.21.0 (pip wheel, Python 3.13, macOS arm64)

This is the F0/F1 record for the development harness only. The Android
host-app build still needs its own manifest before Android claims (§4).

| Field | Value |
|---|---|
| TF version | 2.21.0 (wheel); headers shipped in `tensorflow/include/` |
| F0 import/eager/tf.function | PASS (desktop) |
| Plugin load | PASS via `tensorflow.python.framework.load_library.load_pluggable_device_library` (public `tf.load_library.load_pluggable_device_library` does not exist in 2.21) |
| Device registration | PASS: `/physical_device:GPU:0`, type `GPU`, name (subtype) `VULKAN` (D1 provisional) |
| Device memory reported | TF logs "0 MB memory" at creation; allocator stats callback is implemented but TF queries it lazily — revisit on Android |
| `TF_AssignVariable` / `TF_AssignUpdateVariable` / `TF_VariableInputLockHolder` | Declared in `kernels_experimental.h` but **not exported** by the 2.21 macOS wheel (nm-verified). Plugin resolves `TF_AssignVariable` via `dlsym` and registers `AssignVariableOp` only when present. Variable assign/add/sub work through TF core DEVICE_DEFAULT kernels on our mapped memory |
| `Identity`, `Reshape` | Already registered for GPU by TF core at the same priority — plugin must NOT register them (duplicate-registration error, verified) |
| `StatelessRandomGetKeyCounter` | 2 outputs in 2.21 (key, counter); no `alg` output |
| `TF_AllocateOutput` | Already sets the output; calling `TF_SetOutput` afterwards aborts (verified) |
| D1 device type | **F1b (desktop harness): both variants work.** `GPU`/`VULKAN` lists under `list_physical_devices('GPU')` (Keras/ecosystem checks see it); custom `VULKAN` (via `TF_VULKAN_DEVICE_TYPE=VULKAN`) lists only under `VULKAN` and needs explicit placement everywhere. Decision: keep `GPU`/`VULKAN` provisional per spec; re-run F1b on Android |
| Vulten (001b/F1c) | Open. This implementation is clean-room from the spec op list against the real 2.21 C ABI; audit Vulten before P2 Android work |
| IREE version / Vulkan baseline | Harness device reports Tier A features. IREE spike (task 009): pip iree-compiler compiles VMFB for vulkan-spirv, but pip iree-runtime has no Vulkan HAL driver — F6 execution needs a source-built runtime (src/iree_bridge/spike/) |

## Android (partial, emulator)
Native runtime cross-compiles with NDK r27c (arm64-v8a, API 28,
c++_static; `cmake/android-arm64.cmake`) and the full kernel suite
passes on an Android 36 arm64 emulator (SwiftShader Vulkan). See
docs/devices.md. Still TBD on a real device + host app: TF version/commit,
Python ABI, plugin load mechanism, `RTLD_GLOBAL`/SONAME result, Bionic
namespace result, Adreno/Mali feature tiers.

## Training notes (desktop harness)
- Gradient parity for MatMul+Sum verified exactly; Keras SGD and Adam
  `apply_gradients` work on device (100-step SGD loss decreases).
- Root cause of the earlier optimizer no-op (found and fixed): TF core
  kernels (assign family) read/write UMA mapped memory without stream
  synchronization, racing our deferred GPU dispatches (§6.2 hazard).
  The plugin therefore applies the P0 per-op sync policy at the TF
  boundary (`SyncOnExit` in plugin.cpp); the native runtime keeps
  deferred submission. `ResourceApply*` ops themselves are still not
  registered (variable C-API helpers are not exported by this build);
  Keras optimizers work through the assign path instead.

## FP16 (K1.5, desktop harness)
Harness device (MoltenVK/Apple M4 Pro) reports `shaderFloat16` +
`storageBuffer16BitAccess` after the real feature probe; FP16 MatMul
(f32 accumulate), pointwise, BiasAdd, and f32<->f16 Cast are implemented
and validated, and a Keras `mixed_float16` Dense model runs on device.
FP16 is never used implicitly for FP32 tensors (§6.4).
