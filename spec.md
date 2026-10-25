# TF Vulkan (`tf-vulkan`): Full TensorFlow GPU Acceleration via Vulkan (Android-First)

**Status:** Proposed engineering specification · **Revision:** 1.4 · **Date:** 2026-10-25  
**Project name:** TF Vulkan (`tf-vulkan`)  
**Scope:** **Android-first.** Android (`arm64-v8a`) is the first and only supported target platform for now; desktop Linux/Windows Vulkan is used only as a development/CI harness until Android milestones are met.  
**Owner:** _TBD_ · **Reviewers:** _TBD (TF runtime, Vulkan/GPU, Android packaging)_ · **Glossary:** Appendix A  
**Primary objective:** Execute ordinary TensorFlow Python programs inside an Android app that embeds CPython and full TensorFlow (the **host app**) on Android GPUs through Vulkan, without converting applications to TensorFlow Lite.

### Changes in revision 1.4

| # | Change | Sections |
|---|---|---|
| 1 | Renamed project to TF Vulkan (`tf-vulkan`) with an explicit **Android-first** scope note (and pre-publication GitHub/PyPI name-check requirement) | 1, 2, 3, 6.5, 8, 13, 15 |
| 2 | Added prior-art analysis of **Vulten** (`https://github.com/jswag180/Vulten`, Apache-2.0): 25 existing kernels (including `MatMul`, `Relu`, `SoftMax`, `BiasAdd`, `ReadVariableOp`, `AssignVariable`, `ResourceApplyAdam`), custom `VULK` device type data point for D1, ~90% host-transfer bottleneck validating our residency/deferred-submission design, and runtime `shaderc`/`glslang` vs precompiled SPIR-V trade-off | 1, 5, 7, 19, Appendix A |
| 3 | Added early evaluation gate (`F1c`) and backlog task (`001b`) to evaluate porting/adapting Vulten (and contributing upstream) vs using it as a reference implementation before writing plugin boilerplate from scratch | 1, 4, 14, 17, 18 |

### Changes in revision 1.3

| # | Change | Sections |
|---|---|---|
| 1 | Added `tensorflow/c/kernels_experimental.h` (`TF_AssignVariable`, `TF_AssignUpdateVariable`, `TF_VariableInputLockHolder`) to the F1 symbol checklist and K0.5/P3 resource-variable requirements; clarified `DEVICE_DEFAULT` variable ops vs plugin-registered update ops | 5, 7, 10, 19 |
| 2 | Fixed UMA mapped-`memcpy` vs deferred command-buffer hazard (RAW/WAR/WAW) and specified exact `SP_StreamExecutor` flush triggers (`block_host_until_done`, `synchronize_all_activity`, `record_event`, `memcpy_dtoh`) to prevent deadlocks and stale reads | 6.2, 6.3, 16 |
| 3 | Specified descriptor management (`VkPushConstants` for scalar metadata, ring-buffered resettable `VkDescriptorPool` or `VK_KHR_push_descriptor`) for low-overhead eager dispatch | 6.3, 7 |
| 4 | Pinned exact IREE Vulkan 1.3 baseline requirements (`timelineSemaphore`, `scalarBlockLayout`, `synchronization2`, compute queue), executable ABIs (Descriptor-Set vs `bufferDeviceAddress`), and phased interop (`VkDevice` wrapping + `iree_hal_allocator_import_buffer`) | 5, 8 |
| 5 | Documented classic `PluggableDevice` (StreamExecutor C API) vs `NextPluggableDevice` (PJRT C API) rationale and F0 check | 1, 5, Appendix A |
| 6 | Added Android Bionic linker namespace, `RTLD_GLOBAL`, and `SONAME` resolution rules for embedded CPython + plugin `dlopen` | 4, 5 |
| 7 | Fixed reference list numbering and expanded glossary | 19, Appendix A |

### Changes in revision 1.2

Project renamed to TF Vulkan Android (`tf-vulkan-android`); references to a specific host application replaced with the generic **host app**.

### Changes in revision 1.1

| # | Change | Sections |
|---|---|---|
| 1 | Device type is now an explicit decision (D1): register as TensorFlow type `GPU` with subtype `VULKAN`, following the Metal and DirectML precedents; custom `VULKAN` type is the fallback | 1, 2, 4, 6.1, 15, 18 |
| 2 | Read-only resource-variable residency moved from training (P3) into the inference path (P1→P2) | 2, 7, 10, 14 |
| 3 | Allocator rewritten for unified-memory (UMA) mobile GPUs; budget from `VK_EXT_memory_budget` | 6.1, 6.2 |
| 4 | Deferred command submission specified for P1; per-op fence wait limited to P0 | 6.3 |
| 5 | Concrete plugin C entry points and load mechanisms named; F1 is now a symbol checklist | 3, 4, 5, 8 |
| 6 | Host-compile round-trip workflow defined for compiled artifacts | 8 |
| 7 | New gate F0: full TensorFlow runs in the host app and exposes the plugin C APIs | 4, 14, 17, 18 |
| 8 | P0 acceptance test strengthened (random inputs, odd shapes, float64 reference) | 15 |
| 9 | Opt-in FP16 path planned before P4 | 2, 6.4, 7, 14 |
| 10 | Android failure modes: GPU watchdog, app lifecycle, persistent pipeline cache | 12, 16 |
| 11 | Schedule reconciled with backlog totals | 2, 14 |
| 12 | Plumbing kernels listed; references version-pinned; glossary added | 7, 19, Appendix A |

> **Design decision:** Build a TensorFlow PluggableDevice with a minimal native Vulkan eager-operation path, then an optional IREE-backed graph-compilation path. Keep standard TensorFlow semantics and gradients in the TensorFlow host runtime. Do not attempt to port all of TensorFlow to Vulkan.

## 1. Executive decision

### Chosen design

1. **Keep the host app's existing full TensorFlow and Python.** No TensorFlow fork initially. TensorFlow owns Python APIs, eager/graph semantics, gradients, variable bookkeeping, shape inference, and CPU kernels.
2. **Implement a Vulkan PluggableDevice** exposing a coherent device memory/execution contract via TensorFlow's supported C extension points (`SE_InitPlugin` StreamExecutor C API, `TF_InitKernel` kernel registration; later `TF_InitGraph` and `TF_InitProfiler`). The plugin is loaded into *the same native process* as TensorFlow. **Decision D1 (provisional):** register device type `GPU` with platform subtype `VULKAN`, so the device appears as `/GPU:0`. See §5 "Device type".
3. **Make single-op eager execution work first:** small Vulkan compute shaders for data movement, pointwise, reduction, and a tiled FP32 MatMul. This avoids requiring a compiler invocation for each eager op.
4. **Add graph compilation second:** use TensorFlow-supported graph optimization / function capture where feasible, lower carefully selected pure tensor subgraphs into a version-pinned MLIR/StableHLO-compatible representation, compile with IREE on the host build machine or in an optional development environment, and execute cached VMFB modules via the IREE runtime's Vulkan HAL. **Graph capture and lowering are research risks, not assumed capabilities of the PluggableDevice API.**
5. **Make training a separate milestone:** preserve TensorFlow's autograd, execute supported forward and backward tensor operations on Vulkan, then add optimizer kernels. Never promise that arbitrary Keras models train at first release.
6. **Gate by hardware capabilities.** IREE's documented Vulkan path requires a Vulkan 1.3 feature baseline. Baseline compatibility for lower Vulkan versions is a separate native-shader path, not something IREE automatically supplies.

### Scope note: Why `tf-vulkan` with an Android-first policy

Most of this architecture—the TensorFlow `PluggableDevice` C APIs (`SE_InitPlugin`, `TF_InitKernel`), the SPIR-V compute kernels, and the optional IREE compilation path—is platform-agnostic and works the same way on desktop Linux or Windows as on Android. The Android-specific layers are packaging/Bionic linker namespaces, unified-memory (UMA) optimization, GPU watchdog chunking, and Android app lifecycle management.

Naming the project **`tf-vulkan`** avoids baking a single OS into the package and library names if desktop GPUs are supported later. However, to prevent false expectations:
- **Android is the first and only supported platform for now.** Desktop Linux Vulkan is used solely as a debugging and CI harness (§11), not a supported end-user target for AMD/Intel/NVIDIA desktop GPUs during P0–P3.
- Before publishing any public repository or wheel, verify that `tf-vulkan` (and `tf_vulkan`) is available and unambiguous on both **GitHub** and **PyPI**.

### Prior art: Vulten (`jswag180/Vulten`)

An existing open-source project, **[Vulten](https://github.com/jswag180/Vulten)** (Apache-2.0 license), already implements a desktop-targeted Vulkan `PluggableDevice` for TensorFlow (targeting Vulkan 1.2 via Vulkan Memory Allocator `VMA` and `Vulkan-Hpp`). Its design provides critical empirical lessons and reusable code for `tf-vulkan`:

1. **Existing kernel and C-API scaffolding:** Vulten already wires up `SE_InitPlugin` and `TF_InitKernel` and implements 25 ops (`MatMul`, `Relu`, `ReluGrad`, `SoftMax`, `BiasAdd`, `BiasAddGrad`, `Add`, `AddV2`, `Addn`, `Sub`, `Mul`, `Div`, `DivNoNan`, `Pow`, `Exp`, `Sqrt`, `Sum`, `Cast`, `Identity`, `IdentityN`, `ReadVariableOp`, `AssignVariable`, `AssignAddVariable`, `AssignSubVariable`, and `ResourceApplyAdam`). That covers a substantial fraction of the ops planned for P1–P3, including variable residency and Adam updates.
2. **Device type data point (`VULK:0`):** Vulten registers a custom device type `VULK` (`with tf.device('VULK:0'):`), giving a concrete reference point for Decision D1 (§5).
3. **Validation of our residency and deferred-submission focus:** Vulten's author notes candidly that it is not fast because missing ops cause frequent fallbacks and **~90% of execution time is spent transferring tensors to and from the host**. This directly validates why this specification treats plumbing ops (§7 K0), device-resident resource variables (§7 K0.5), UMA buffer mapping (§6.2), and deferred command submission (§6.3) as first-order requirements rather than late optimizations.
4. **Desktop vs. Android differences to address if porting:** Vulten links `shaderc_combined`, `glslang`, and `SPIRV-Tools` to compile GLSL shaders at runtime and links directly against desktop `_pywrap_tensorflow_internal.so`. On Android, `tf-vulkan` should precompile shaders to SPIR-V at build time (to avoid shipping `shaderc`/`glslang` in the APK and paying runtime shader compile costs) and adapt CMake/linker rules for the host app's Android TensorFlow build.

**Action before writing new plugin code (Task `001b` / Gate `F1c`):** Audit Vulten's codebase against the host app's TensorFlow version. Decide whether to (a) port/fork Vulten to Android (and contribute portable improvements back upstream) or (b) start from a clean room using Vulten as the primary reference implementation.

### Why this is preferable

| Strategy | Benefit | Fatal limitation / cost | Decision |
|---|---|---|---|
| Port/adapt Vulten (`jswag180/Vulten`) + add Android UMA, deferred submission, and optional IREE | Reuses Apache-2.0 `SE_InitPlugin`/`TF_InitKernel` scaffolding and ~25 ops | Needs precompiled SPIR-V (replace runtime `shaderc`), plumbing ops, deferred queue submission, and Android ABI alignment | **Evaluate first (`001b` / `F1c`)** |
| Write PluggableDevice from scratch + direct shaders + optional IREE | Clean-slate control over UMA allocator and command batching | Duplicates boilerplate that Vulten already solved | Fallback if Vulten's runtime structure is harder to adapt than rewrite |
| Rewrite all of TensorFlow ops directly in Vulkan | Full control | Large operator surface, fusion and maintenance burden | Minimal eager path only |
| Convert everything to TensorFlow Lite | Already mature for inference | Not ordinary full TF Python; no general training path | Baseline, not product |
| Use IREE alone | Efficient ahead-of-time graphs | Doesn't expose generic `tf.device('/GPU:0')` eager execution by itself | Graph engine |
| NextPluggableDevice (PJRT C API) | OpenXLA ecosystem alignment | Assumes an XLA/PJRT compiler runtime; poorly suited to lightweight single-op mobile eager dispatch without full XLA on Android | Reject for P0–P2; monitor if host TF drops classic `SE_InitPlugin` |
| Fork TensorFlow/XLA to add Vulkan | Deepest integration | Massive build/maintenance exposure | Only if plugin limitations are proven blockers |

## 2. Target behavior and non-goals

### User-visible target

```python
import tensorflow as tf

print(tf.config.list_physical_devices("GPU"))   # Vulkan plugin, subtype VULKAN

with tf.device("/GPU:0"):
    a = tf.random.normal((2048, 2048))
    b = tf.random.normal((2048, 2048))
    c = tf.matmul(a, b)

print(c.numpy().shape)
```

**Note:** This is the **target experience**, not an implemented API. TensorFlow op placement and randomness on device must be validated experimentally. Early examples should create tensors on CPU and explicitly copy or place them on GPU as needed. If D1 is reversed in favour of a custom device type, every `/GPU:0` in this spec becomes `/VULKAN:0`.

### Maturity levels

- **P0: proof of hardware execution.** One TensorFlow `MatMul` executes Vulkan commands and returns numerically correct results. Run with soft placement disabled; GPU timestamp/profiler confirms actual dispatch.
- **P1: credible eager backend.** MatMul, core pointwise and reductions, reshape/copy paths; GPU tensor residency across chains; deferred command submission; safe CPU fallback.
- **P2: useful Keras inference.** Dense network and selected CNN on representative Adreno/Mali devices; **model weights resident on the device as resource variables** (no per-call weight upload); fused subgraphs and measured speedups; opt-in FP16 measured against FP32.
- **P3: useful small-model training.** GradientTape works end-to-end on chosen MLP; supported gradients and optimizer operations stay GPU-resident where possible.
- **P4: broader ecosystem.** More ops, dynamic shape coverage, cache distribution, mixed precision, third-party hardware.

### Non-goals (first 6 months)

The six-month window realistically covers P0–P2 for one engineer; P3 fits only at the optimistic end of the §14 estimates.


- 100% TensorFlow op coverage; general control flow, arbitrary custom ops, distributed TensorFlow, DTensor, TPU/XLA equivalence.
- JIT compilation of arbitrary Python code, including non-TensorFlow NumPy expressions.
- Guaranteed speedup for tiny tensors or every device.
- Vulkan-based acceleration on hardware without required Vulkan compute features.
- Replacing TensorFlow Lite, NNAPI, or vendor inference runtimes for production mobile inference.
- Supporting desktop GPUs (AMD/Intel/NVIDIA on Linux or Windows) as an end-user product before Android milestones P0–P2 are proven.
- Quietly falling back to CPU while claiming successful GPU execution.

## 3. System boundaries and diagram

```text
Host Android app (arm64-v8a)
└── CPython + full TensorFlow shared libraries
    ├── Python: tf.*, tf.keras.*, tf.GradientTape
    ├── TensorFlow runtime: eager, graph, autograd, ops, CPU fallback
    └── libtf_vulkan.so: plugin loaded in-process
        ├── TensorFlow C plugin ABI: SE_InitPlugin, TF_InitKernel,
        │   TF_InitGraph (optional), TF_InitProfiler (optional)
        ├── Device allocator, stream executor, transfer/staging, fences
        ├── Eager engine (direct SPIR-V kernels, pipeline cache)
        ├── Optional graph capture / supported-subgraph compiler adapter
        ├── Compiled module registry + shape-keyed artifact cache
        └── Vulkan HAL boundary
            ├── Native Vulkan command path (initial, direct kernels)
            └── Optional IREE runtime Vulkan driver (compiled VMFB)
                └── Android Vulkan loader → Adreno / Mali / other GPU
```

**Hard rule:** TensorFlow-visible GPU buffers must have a single explicit owner and lifecycle. Native Vulkan and IREE may use distinct allocator models; do not assume buffers are automatically interchangeable. First compiled-mode implementation may incur explicit transfers. Profile and optimize interop only after confirming exported/imported buffer compatibility and synchronization semantics in the pinned IREE version.

## 4. Feasibility gates before writing the backend

The first sprint is deliberately designed to kill an unsound architecture early.

| Gate | Experiment | Pass condition | Failure action |
|---|---|---|---|
| F0 Full TF in host app | Import TensorFlow in the host app; record version, build origin, and whether the plugin C APIs were compiled in | `import tensorflow` works; eager and `tf.function` run on CPU; plugin loader code present | Stop: fix or rebuild the host app's TensorFlow before any plugin work |
| F1 TensorFlow plugin load | Minimal CPU-hosted test device plugin against **exact Android TF build**; check symbol checklist (§5) | TF calls `SE_InitPlugin` and `TF_InitKernel`; device listed without crash | Inspect/export required C symbols; consider version-pinned TF fork only after documenting blocker |
| F1b Device type (D1) | Register the test plugin once as type `GPU`/subtype `VULKAN`, once as custom type `VULKAN` (or `VULK` per Vulten) | Record list/placement behaviour, Keras GPU checks, default placement preference for each | Pick the variant with fewer blockers; update §2 examples |
| F1c Vulten audit & port spike | Evaluate `jswag180/Vulten` (Apache-2.0) against the pinned TF build and Android NDK; test cross-compiling its `SE_InitPlugin`/`TF_InitKernel` layer | Clear decision: port/adapt Vulten (replacing runtime `shaderc` with offline SPIR-V and adding UMA/deferred submission) vs clean-room implementation using Vulten as reference | Document decision in architecture notes before starting 003/004 |
| F2 Packaging | Load `.so` alongside TensorFlow inside the host app's embedded Python | `dlopen` succeeds across Android Bionic linker namespaces; no missing C++/protobuf/libc++/NDK/`TF_*` symbols | Set `RTLD_GLOBAL` on TF load or link against exact `libtensorflow_framework.so` SONAME; align NDK/STL |
| F3 Vulkan compute | Dispatch independent 256×256 FP32 kernel in the host app process | Correct values; Vulkan validation clean | Verify driver capabilities/app linkage |
| F4 Tensor buffer ownership | TF tensor allocated/copied to plugin device, released safely | Repeated 10k create/execute/destroy cycles without leaks or UAF | Redesign allocator contract |
| F5 TF kernel execution | Register and execute `MatMul` via `/GPU:0` (Vulkan subtype) | Device-side profiler event and CPU parity | Stop: plugin is not actually integrated |
| F6 IREE portability | Run precompiled IREE module on target phone | Correct GPU results with supported feature set | Native path remains viable; IREE optional |
| F7 Graph lowering | Extract one deterministic `tf.function` segment, lower, run, compare | Shape/dtype/semantics parity and amortizable compilation | Ship P1 eager backend; postpone P2 compiler |

**Go/no-go:** If F0–F5 fail on the chosen TensorFlow build after a bounded investigation, do not spend months implementing operators. Publish a compatibility report and decide whether a version-pinned TensorFlow fork is warranted. No API can be considered stable without testing against the actual released Android wheel/shared libraries.

## 5. Critical compatibility decisions

### TensorFlow ABI

TensorFlow PluggableDevice uses C extension APIs for device runtime, kernel registration, and optionally graph optimization and profiling. **ABI compatibility must be tested with each TensorFlow release and distribution.** Do not link against private C++ internals and assume forward compatibility. Pin exact TF source commit, headers, build features, compiler ABI, and the host app's TensorFlow binary. Maintain a compatibility manifest.

**Action:** At project start, inventory the exact TensorFlow wheel/build used by the host app, its exported symbols, Python ABI, libc++ linkage, supported `tensorflow/c/experimental/*` headers, and the plugin initialization mechanism. Some upstream extension APIs or documented examples may have changed or not be enabled on Android builds. No official full-TensorFlow wheels exist for Android, so the host app's build is presumably custom and may have been built without plugin support (gate F0).

**Plugin entry-point checklist (F1).** For the pinned TF commit, confirm each surface exists in headers and in the host app's binary (and confirm classic StreamExecutor `PluggableDevice` support was not stripped in favor of PJRT/`NextPluggableDevice`):

| Surface | Header (upstream path; verify at pinned commit) | Needed from |
|---|---|---|
| `SE_InitPlugin` — device, stream, memory, events | `tensorflow/c/experimental/stream_executor/stream_executor.h` | P0 |
| `TF_InitKernel` — kernel registration & context | `tensorflow/c/kernels.h`, `tensorflow/c/tf_tensor.h` | P0 |
| Resource variable C helpers (`TF_AssignVariable`, `TF_AssignUpdateVariable`, `TF_MaybeLockVariableInputMutexesInOrder`, `TF_GetInputTensorFromVariable`) | `tensorflow/c/kernels_experimental.h` | K0.5 (P2) & P3 |
| `TF_InitGraph` — C Grappler graph optimizer | `tensorflow/c/experimental/grappler/grappler.h` | compiled mode (§8) |
| `TF_InitProfiler` — pluggable profiler | `tensorflow/c/experimental/pluggable_profiler/pluggable_profiler.h` | P0 evidence (optional) |

**Load mechanisms and Android Bionic linker namespaces.** Record which works in the host app: (a) automatic loading of `.so` files from the `tensorflow-plugins/` directory under site-packages at `import tensorflow`; (b) explicit loading through `tf.load_library.load_pluggable_device_library('/path/to/libtf_vulkan.so')`. On Android 7.0+ (Bionic linker namespaces) with embedded CPython:
- Python imports extension modules with `RTLD_LOCAL` by default, which can hide `TF_*` symbols exported by `libtensorflow_framework.so` or `_pywrap_tensorflow_internal.so` when `dlopen` loads `libtf_vulkan.so`.
- Test and document whether the plugin requires (1) linking `DT_NEEDED` against the exact `SONAME` of the host app's `libtensorflow_framework.so` (or `:_pywrap_tensorflow_internal.so`, as Vulten does on desktop) in the same linker namespace / `RUNPATH`, or (2) setting `sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)` prior to `import tensorflow`.

### Device type (decision D1)

A PluggableDevice reports a device *type* (what users write in `tf.device`) and a platform *subtype*. Apple Metal and Microsoft DirectML both register type `GPU` with their own subtypes, whereas Vulten (`jswag180/Vulten`) registers a custom device type `VULK` (`tf.device('VULK:0')`). Trade-offs:

| Option | Gains | Costs |
|---|---|---|
| Type `GPU`, subtype `VULKAN` (**provisional choice**, Metal/DirectML precedent) | `list_physical_devices('GPU')`, Keras/tf.data GPU checks, and the placer's GPU preference work unchanged | Code that assumes "GPU means CUDA" may misbehave (rare on Android); diagnostics must expose the subtype |
| Custom type `VULKAN` or `VULK` (Vulten precedent) | Unambiguous; no collision with CUDA assumptions | Likely no automatic placement preference; ecosystem GPU checks miss the device; every example needs explicit `tf.device` placement (contributing to CPU fallbacks when ops are left unplaced) |

Resolve with F1b before PR #2.

### Vulkan and IREE baseline

IREE's published Vulkan deployment documentation (`iree.dev/guides/deployment-configurations/gpu-vulkan/`) specifies a hard **Vulkan 1.3** baseline with four mandatory capabilities:
1. **Compute queue** (`VK_QUEUE_COMPUTE_BIT`)
2. **`timelineSemaphore`** (`VkPhysicalDeviceVulkan12Features::timelineSemaphore`)
3. **`scalarBlockLayout`** (`VkPhysicalDeviceVulkan12Features::scalarBlockLayout`)
4. **`synchronization2`** (`VkPhysicalDeviceVulkan13Features::synchronization2`)

In addition, IREE supports two executable ABIs: **Descriptor-Set Dispatch** (the portable baseline ABI) and **Buffer Device Address (`bufferDeviceAddress`)** (an optional faster ABI when the driver exposes `bufferDeviceAddress` and the VMFB is compiled for it), plus opportunistic extensions (`VK_KHR_push_descriptor`, `VK_EXT_calibrated_timestamps`, and external memory extensions). Test actual `VkPhysicalDeviceFeatures2`/properties, extensions, limits, buffer alignment, timestamp support, subgroup support, and memory heaps (and cross-check with `iree-run-module --dump_devices` where available); never infer compatibility from Android version or GPU marketing name.

- **Tier A:** device passes pinned IREE Vulkan 1.3 requirements (`feature_tier[vulkan-1.3]: supported`); native eager + optional IREE compiled mode (using Descriptor-Set ABI by default, or BDA if validated).
- **Tier B:** Vulkan 1.1/1.2 compute is present but IREE's Vulkan 1.3 baseline is missing; native SPIR-V kernels only **if** feature-level requirements are verified.
- **Tier C:** no supported Vulkan compute or known-bad driver; GPU backend disabled, clear diagnostic, CPU TensorFlow remains available.

Native kernels should target a validated SPIR-V environment for the target driver's actual Vulkan capabilities, and request only necessary extensions. Do not silently assume FP16, subgroup matrix instructions, timeline semaphores, or storage 8-bit/16-bit support.

### Android packaging

- Initial supported ABI: `arm64-v8a`; Android API minimum chosen after verifying actual Vulkan feature availability and host app compatibility, not guessed in the spec.
- Build native artifacts reproducibly using Android NDK and a pinned CMake/Bazel/IREE toolchain.
- The host app must package or locate `libvulkan.so` dynamically and load plugin libraries from Android-supported application-owned locations.
- Resolve executable/writable directory restrictions, native library extraction, `dlopen` search paths, and `RTLD_*` interactions on **real, non-rooted devices**.
- Ship compiler tools on the host/workstation first. Never assume `iree-compile` can run within the host app on Android or that a full MLIR compiler fits the APK.
- Make binary sizes, cold start latency, storage, and download/update strategy first-class metrics.

## 6. Device runtime implementation

### 6.1 Device discovery and registration

Register the device (type per D1, subtype `VULKAN`) when GPU initialization succeeds. Advertise memory capacity conservatively: query `VkPhysicalDeviceMemoryBudgetPropertiesEXT` via `VK_EXT_memory_budget` where supported (widely present on Adreno 6xx/7xx+ and modern Mali Valhall/Immortalis drivers), and fall back to a conservative fraction (e.g. 25–40%) of the relevant `VkPhysicalDeviceMemoryProperties` heap size on older drivers that omit the extension, since GPU memory on phones is system memory and competes with Android's low-memory killer (LMK). Expose device name, GPU vendor/device IDs, driver version, Vulkan API/feature caps, and selected execution tier via the project's diagnostics. Avoid double-registration across Python interpreter reloads.

### 6.2 Allocator

Use `VkBuffer` + `VkDeviceMemory`/suitable device allocator for device-resident tensors. Maintain a reusable allocation pool, bounded by a configurable app budget, with per-allocation size/alignment and outstanding-use tracking. Handle allocation pressure/OOM gracefully; no hidden CPU copy presented as GPU storage.

**Unified memory and synchronization hazards.** Adreno and Mali GPUs share physical memory with the CPU and usually expose memory types that are both `DEVICE_LOCAL` and `HOST_VISIBLE`. The allocator must:

1. Enumerate memory types and heaps at startup and record them in diagnostics.
2. Prefer `DEVICE_LOCAL | HOST_VISIBLE` (coherent, or `HOST_CACHED` with explicit `vkFlushMappedMemoryRanges`/`vkInvalidateMappedMemoryRanges`, whichever benchmarks better) for tensor storage when present.
3. Implement host↔device copies via mapped `memcpy` + required cache flush/invalidate when the memory type allows it, **subject to strict stream-ordering rules under deferred submission (§6.3)**:
   - **Device-to-host (`memcpy_dtoh`):** Never execute an immediate CPU `memcpy` from a mapped `VkBuffer` that is targeted by unsubmitted commands in an open command buffer or by an in-flight GPU batch. First flush any open command buffer that touches the buffer, wait for the associated fence/timeline semaphore, call `vkInvalidateMappedMemoryRanges` (if non-coherent), and only then `memcpy` to the host destination.
   - **Host-to-device (`memcpy_htod`):** If the target buffer is freshly allocated or idle (zero pending GPU reads/writes in open or in-flight command buffers), perform a direct CPU `memcpy` + `vkFlushMappedMemoryRanges` immediately without flushing the stream. If the buffer has pending GPU work (e.g. in-place variable update), either allocate/rename a fresh buffer or synchronize before writing.
4. Evaluate `VK_EXT_external_memory_host` to import TensorFlow CPU tensor memory without copying; treat as optional, per-driver, and alignment-constrained (`minImportedHostPointerAlignment`).
5. Keep a staging-buffer path (`vkCmdCopyBuffer`) for drivers or heaps that expose no host-visible device-local type.

At P0, accept whichever path is simplest. At P1, make chained GPU ops stay resident. Explicitly specify read/write and ownership barriers; recycle allocations only after the relevant fence/timeline completion.

### 6.3 Streams, synchronization, errors

Begin with one compute queue and one command-submission worker. Map TensorFlow's `SP_StreamExecutor` operations to Vulkan sequencing; implement host-to-device, device-to-host, device-to-device copies and event/wait dependencies.

**Submission policy by milestone:**

- **P0:** one command buffer per op, submit, wait on a fence. Simple and debuggable.
- **P1 (required):** deferred submission. Ops record into an open command buffer on the stream. Ordering between consecutive dispatches in the same command buffer comes from `vkCmdPipelineBarrier` (`VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT`, `VK_ACCESS_SHADER_WRITE_BIT` → `VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT`). Without deferred submission, chained ops stay resident but each still pays a full `vkQueueSubmit` + kernel driver round trip.
- **Mandatory flush triggers (`SP_StreamExecutor` contract):** To prevent deadlocks in TensorFlow's PluggableDevice runtime, the open command buffer **must** be closed and submitted to `VkQueue` (attaching a fence or timeline semaphore value) upon:
  1. `block_host_until_done(device, stream, status)` and `synchronize_all_activity(device, status)` (flush and block until fence/semaphore completes).
  2. `record_event(device, stream, event, status)` (flush the open command buffer and associate the resulting fence/timeline value with `SP_Event`, so subsequent `wait_for_event` or host polling makes forward progress).
  3. `memcpy_dtoh` when reading a buffer written by recorded commands in the open batch.
  4. Reaching a bounded dispatch count (e.g. 32–64 dispatches) or estimated work threshold (to avoid GPU watchdog timeouts and keep the GPU fed while recording).
- Fence and timeline waits happen only at host synchronization points; allocation and descriptor-pool recycling are tied to the completion token of the flushed batch that last used the resource.
- **Descriptor and push-constant management:** Pass scalar kernel parameters (dimensions, strides, element counts, reduction axes, alpha/beta) via `vkCmdPushConstants` (fits comfortably within the 128-byte Vulkan minimum `maxPushConstantsSize`) to avoid allocating uniform buffers per op. Bind storage buffers via per-batch ring-buffered `VkDescriptorPool` instances reset in bulk (`vkResetDescriptorPool`) once the batch fence signals, or via `VK_KHR_push_descriptor` where supported.

Never hold Python GIL while waiting for long GPU execution if the Python entry path makes that avoidable. Preserve device lifetime across asynchronous work. Translate `VK_ERROR_DEVICE_LOST`, allocation failure, unsupported op, and invalid shapes into explicit TensorFlow status errors.

### 6.4 Tensor descriptors

Internally represent dtype, rank, dimensions, strides/layout, physical buffer, offset, size, and last-use token. Start with contiguous FP32 tensors. FP32 is the correctness baseline, but mobile GPU FP32 throughput is modest and most mobile ML performance comes from FP16. Plan an **opt-in FP16 path** (storage and arithmetic, gated on `shaderFloat16` and 16-bit storage features) tied to Keras mixed precision (`mixed_float16`), introduced with backlog item 016 before P4. Never use FP16 implicitly for FP32 tensors. Validate integer overflow in dimension products and range checks on buffer offsets.

### 6.5 Tracing and introspection

Provide a `tf_vulkan.status()`-style optional diagnostic helper (not a substitute for normal TF APIs) reporting real driver details, supported ops, selected tier, cache activity, GPU allocations, CPU fallbacks, and per-op transfer/dispatch time.

## 7. Kernel plan

> **Note on Vulten reuse:** Vulten (`jswag180/Vulten`, Apache-2.0) already implements C++ kernel registrations and shaders for 25 ops (`MatMul`, `Relu`, `ReluGrad`, `SoftMax`, `BiasAdd`, `BiasAddGrad`, `Add`, `AddV2`, `Addn`, `Sub`, `Mul`, `Div`, `DivNoNan`, `Pow`, `Exp`, `Sqrt`, `Sum`, `Cast`, `Identity`, `IdentityN`, `ReadVariableOp`, `AssignVariable`, `AssignAddVariable`, `AssignSubVariable`, `ResourceApplyAdam`). Where task `001b` confirms correctness and Android compatibility, reuse or adapt those implementations rather than rewriting them from scratch, while adding the missing plumbing ops (K0.0), UMA zero-copy/mapped paths, and precompiled SPIR-V pipelines.

### Phase K0: correctness foundations

0. **Plumbing kernels** needed for strict placement inside `tf.function`: `Const`, `Identity`, `NoOp`, `_Arg`/`_Retval` handling as required by the pinned build, and `Shape`/`Size`/`Rank` with **host-memory outputs**. Without these, strict-mode graphs fail before any compute kernel runs (and eager/graph execution suffers the ~90% host-transfer bottleneck observed in Vulten).
1. `Identity`, device copy, host/device transfers.
2. `AddV2`, `Sub`, `Mul`, `Relu`, `Relu6`, `BiasAdd` for contiguous FP32.
3. `MatMul` FP32, initially 2-D no transpose; then transpose flags, batched variants if justified.
4. `Sum` and `Mean` along selected axes with deterministic reduction order per implementation, documented FP tolerances.
5. `Reshape`/`Squeeze`/`ExpandDims` as safe metadata operations only when TensorFlow's device-memory and kernel API permits; otherwise explicit copy.

### Phase K0.5: read-only variable residency (required for P2)

Keras layer weights are resource variables. If their ops are not available on the device, weights live on CPU and **every forward pass uploads every weight**, which can erase any inference speedup.
- In TensorFlow's PluggableDevice runtime, core TensorFlow typically registers device-agnostic (`DEVICE_DEFAULT`) kernels for `VarHandleOp`, `ReadVariableOp`, `VarIsInitializedOp`, and `DestroyResourceOp`, while **variable assignment and mutation ops** (`AssignVariableOp`, `AssignAddVariableOp`, `AssignSubVariableOp`) must be registered by the plugin using the C API helpers in `tensorflow/c/kernels_experimental.h` (`TF_AssignVariable`, `TF_AssignUpdateVariable`, `TF_MaybeLockVariableInputMutexesInOrder`).
- Verify during F1/007b which `DEVICE_DEFAULT` variable ops are active in the host app's TF build, and implement `AssignVariableOp` via `TF_AssignVariable` (initially for weight initialization and checkpoint restore, referencing Vulten's `ReadVariableOp` and `AssignVariable` implementations) plus any missing read/handle ops.
- Validate that a Keras `Dense` model's weights stay on the device across repeated `model(x)` calls, using allocation and transfer counters (zero H2D weight transfers after initial load).

### Phase K1: Keras MLP

`Sigmoid`, `Tanh`, `Exp`, `Log`, `Maximum`, `SquaredDifference`, `Softmax`/stable log-sum-exp, additional reductions. General `AssignVariableOp`/`AssignAddVariableOp` (`TF_AssignUpdateVariable`) and optimizer resource kernels (`ResourceApplyGradientDescent`, `ResourceApplyAdam`—already prototyped in Vulten) for training belong to P3. Avoid declaring mutating resource ops supported before lifetime, mutex locking (`TF_VariableInputLockHolder`), and alias semantics are verified.

### Phase K1.5: opt-in FP16

FP16 variants of MatMul, pointwise, and reductions (FP32 accumulation where supported), selected only under mixed precision on devices that pass the feature gate. Benchmark against FP32 on every target device.

### Phase K2: CNN

`Conv2D` (NHWC baseline), `DepthwiseConv2dNative`, pooling, common data-layout transforms. Select one validated mobile CNN and avoid covering every convolution parameter combination prematurely.

### Shader strategy

- Development: precompile GLSL/HLSL or MLIR-generated kernels into SPIR-V with pinned tooling; ship binaries, not a runtime shader compiler (`shaderc`/`glslang` as used in desktop Vulten adds substantial binary size and cold-start overhead on Android).
- Eager MatMul: naive reference shader for correctness → tiled shared-memory shader → optional subgroup variants selected by capability.
- Kernel specialization: shape and dtype keys; bounded shader/pipeline variants; an LRU pipeline cache.
- Dispatch: batch/fuse pointwise ops when the graph engine exists; avoid per-op high-overhead shader JIT.
- Long dispatches: split large kernels so no single submission approaches the driver's GPU watchdog limit (see §12).
- Correctness: compare to CPU reference using per-op absolute/relative tolerances and explicit treatment of NaN, Inf, signed zero, reduction order, subnormal behavior, and denormal modes.

## 8. Compiled graph engine (optional and gated)

### Objective

Amortize kernel launches, eliminate temporaries and repeated CPU/GPU transitions, and use IREE's existing lowering and Vulkan code generation for **supported pure tensor graphs**.

### Proposed compilation pipeline

```text
tf.function / eligible TensorFlow graph
  → conservative partition into supported, pure tensor subgraphs
  → validated conversion to supported MLIR / StableHLO representation
  → signature + shape/layout/dtype/feature key
  → host-side IREE compilation (or controlled ahead-of-time build)
  → versioned VMFB artifact
  → Android IREE runtime, Vulkan HAL
  → explicit buffer bindings / output materialization
```

**Important caveat:** There is no assumed magic mapping from arbitrary TensorFlow GraphDef to StableHLO or guarantee that an arbitrary subgraph can be intercepted with existing plugin APIs. First produce a compatibility spike identifying supported graph optimization hooks, conversion tools, supported operations, shape constraints and resource semantics for the *pinned* versions. The expected interception route is the C Grappler API (`TF_InitGraph`): a plugin graph optimizer receives the `GraphDef` of each `tf.function` placed on the device and rewrites eligible regions into a single fused op (e.g. `_VulkanCompiledRegion`) whose registered kernel executes the matching VMFB through IREE. This applies to `tf.function` graphs only, not to op-by-op eager execution. If no reliable interception exists, use an **explicit** opt-in API (`tf_vulkan.compile(tf_function, example_inputs)`) before considering intrusive TF modifications.

### Partitioning rules

Compile only contiguous regions that are: pure tensor operations; known shape/dtype signatures or supported bounded dynamic dimensions; no host callbacks; no stateful or resource side effects; no unresolved control flow; and no unsupported kernels. Insert boundaries at stateful/unsupported ops. Validate result ordering, layouts, broadcasting, and error behavior.

### Caching

Key artifacts by `sha256(canonical graph, constant hashes, input shapes/dtypes/layouts, specialization parameters, TF version, converter/IREE version, Vulkan backend config, device feature fingerprint)`. Build atomically; checksum outputs; use storage quota + LRU eviction; invalidate on compiler/runtime ABI changes. Install only signed/locally verified packaged artifacts where executable module load paths require integrity checks. A VMFB is not a trusted universal portable binary; pin toolchain and verify runtime compatibility.

### Host-compile round trip

Compilation happens on a host, but the graph the cache key describes is the one TensorFlow builds on the phone. The workflow must close that loop:

1. **Capture (phone):** in development mode the plugin optimizer serializes each eligible region (canonical `GraphDef`, constants or their hashes, signatures, device feature fingerprint) to an export directory and records a cache miss.
2. **Compile (host):** a host tool running the *same* TF version and converter pins lowers the region and runs `iree-compile` for the recorded Vulkan target.
3. **Install (phone):** the VMFB plus manifest (cache key, toolchain versions, checksum) is pushed into the app's artifact store; the next run hits the cache.

**Alternative to evaluate in backlog item 010:** do the entire lowering on desktop TensorFlow from the same Python model (e.g. HLO export via `tf.function(...).experimental_get_compiler_ir`), and confirm the resulting signatures match what the phone captures. If they do, step 1 reduces to recording signatures.

### Compile policy

- Default mode: **precompiled** known example/model signatures for Android.
- Development mode: compile on desktop and install module on phone.
- Optional future mode: bounded asynchronous compile service or on-device compiler after a size/latency/security feasibility study.
- If unsupported: execute ordinary native eager ops or TensorFlow CPU ops, never silently claim graph acceleration.

### Shared device and allocator progression

Native eager Vulkan and IREE Vulkan HAL may own different `VkDevice` instances, queues, allocator metadata, or synchronization. Treat zero-copy interop as **unproven** until validated against the pinned IREE release, and follow a three-stage progression:

1. **Stage 1 (Initial P2 compiled mode — explicit staging/copies):** Copy inputs/outputs across the compiled-region boundary through mapped host-visible buffers. Prefer executing coarse compiled regions rather than interleaving per-op native and IREE dispatches.
2. **Stage 2 (Shared `VkDevice` context):** Initialize the IREE Vulkan HAL driver (`<iree/hal/drivers/vulkan/api.h>`) by wrapping the plugin's existing `VkInstance`, `VkPhysicalDevice`, and `VkDevice` handles rather than creating a second Vulkan logical device in-process. This avoids duplicate driver overhead and cross-device synchronization on mobile GPUs.
3. **Stage 3 (Zero-copy buffer import):** Use `iree_hal_allocator_import_buffer` with `iree_hal_external_buffer_t` (`IREE_HAL_EXTERNAL_BUFFER_TYPE_OPAQUE_FD` via `VK_KHR_external_memory_fd` / Android hardware buffers, or `HOST_ALLOCATION` on UMA) and a completion-tied `release_callback` so TensorFlow retains single-owner buffer lifetime (§3).

## 9. CPU fallback and behavior guarantees

**Correctness over illusion.** TensorFlow soft placement can make an op placed on the Vulkan device run on CPU when no device kernel exists. That is appropriate for usability but can conceal the absence of acceleration.

Modes:

- `strict`: disable soft placement; unsupported GPU kernels are explicit errors. **Required for all GPU proof tests.**
- `balanced`: allow supported subgraph execution with clear fallback counters and device placement logging.
- `cpu`: disable plugin execution.

Always emit warnings/diagnostics for repeated cross-device copies, expensive fallback islands, unsupported dtypes, and graph recompilations. In benchmarks, measure end-to-end latency including transfers and synchronizations; GPU-only kernel timings must be labeled separately.

## 10. Training support

Training works only when TensorFlow's automatic differentiation generates gradient operations that the backend can place and execute correctly. The GPU plugin does **not** need to reimplement automatic differentiation itself, but it must support relevant gradient ops, resource variable updates, and synchronization semantics.

### Minimum training demonstration

1. Two-layer FP32 Keras MLP, fixed input shape, small synthetic dataset.
2. Forward pass outputs match CPU within documented tolerances.
3. `tf.GradientTape` returns finite gradients with known values checked numerically.
4. Weights already device-resident via K0.5; GPU-accelerated backward MatMul/reductions; optimizer may initially update variables on CPU with transparently measured copies.
5. Then add supported SGD/Adam resource operations and validate repeated optimizer steps, saved checkpoints, and resume.

**No training claim** until gradients, optimization, and loss trajectory tests pass. Stateful operations must never be incorrectly compiled as pure graph fragments.

## 11. Benchmarking methodology

### Test devices

- One modern Adreno Android phone with confirmed Vulkan feature profile.
- One Mali Android phone with independently confirmed profile.
- One older Vulkan phone (negative/compatibility test).
- Linux desktop Vulkan GPU as a debugging reference, not representative mobile performance.

Record device model, SoC, driver build, Android release, Vulkan API/features, CPU governor if accessible, power mode, thermal state, battery condition and background load. Exact devices are selected only after feature probing; no claim of universal support.

### Workloads

| Class | Shapes/workload | Comparisons |
|---|---|---|
| Pointwise | 1K, 1M, 16M FP32 elements | TF CPU, native GPU |
| GEMM | 128², 512², 1024², 2048², tall/skinny | TF CPU, native GPU, IREE compiled |
| Eager chain | MatMul → BiasAdd → ReLU → MatMul | CPU vs direct kernels vs compiled |
| Inference MLP | 2–4 Dense layers, batch 1/8/32 | CPU vs Vulkan; supported TFLite as context only |
| CNN | small validated NHWC classifier, batch 1 | CPU, Vulkan, TFLite delegate |
| Training | 2-layer MLP 100 iterations | CPU vs Vulkan, loss and gradient parity |

### Measurement protocol

- Warm pipeline compilation; also record cold compile and first-inference latency separately.
- At least 30 samples per condition when affordable; report median, p10/p90, and confidence intervals / variability; paired alternating A/B order to reduce thermal confounds.
- Separate GPU kernel time, command submission, H2D/D2H transfers, allocator overhead, Python/TF overhead, and whole-call latency.
- Use Vulkan timestamp queries **only when supported and calibrated/validated**; otherwise CPU sync timing with label.
- Include wall power/energy proxy where feasible; monitor sustained performance and thermal throttling over multi-minute runs.
- Check that outputs are actually consumed (`.numpy()`/barrier where relevant) so execution isn't mistakenly timed asynchronously.
- Benchmark with strict placement and log every fallback. Do not benchmark native GPU and CPU at different precision without disclosure.

### Proposed success gates (targets, not promises)

- **P0:** 100% correctness on specified MatMul conformance inputs and repeatability under 10k allocations/dispatch cycles.
- **P1:** chained operations avoid CPU round trips, and at least one 1024²+ FP32 MatMul is measurably faster end-to-end than TF CPU on one target phone after warmup.
- **P2:** supported real MLP inference produces correct outputs and a meaningful end-to-end speedup vs TF CPU on at least one target phone; no strict minimum over a competitive TFLite delegate.
- **P3:** 100 consecutive training updates with finite loss and gradient parity, reproducible within declared numeric tolerance.

## 12. Security, safety, and reliability

- Treat arbitrary model dimensions and serialized artifacts as untrusted. Validate buffer sizes, offsets, shape products, and input layouts against overflow and allocation limits.
- No arbitrary SPIR-V module loading from random model input; compiled artifacts must come from trusted tooling or a clearly labeled advanced development mode.
- Run Vulkan validation layers in CI/development, not unconditionally in production.
- Defensive device-lost handling: mark backend unavailable, synchronize/cleanup where possible, surface TensorFlow exception, allow process restart. Do not pretend all GPU driver failures can be recovered in place.
- **GPU watchdog:** mobile drivers may reset the GPU and report device loss when a submission runs too long. Bound per-submission work (chunk large GEMMs and reductions), and measure the practical limit per driver during F3.
- **App lifecycle:** handle backgrounding, `onTrimMemory` pressure, and process death mid-submission. Flush and release caches on trim; on restart, treat all in-flight state as lost.
- **Pipeline cache persistence:** persist `VkPipelineCache` to app storage, keyed by driver version and `pipelineCacheUUID`, to cut cold-start latency; discard on mismatch.
- Use bounded caches, memory budgets, and an explicit GPU disable toggle. Avoid exhausting phone memory or persistent overheating.
- Do not require elevated Android permissions, rooting, or remote compilation services for the baseline package.
- Review third-party licenses (TensorFlow, IREE, Vulten [Apache-2.0], VMA, shader libraries) and notice requirements before distributing binaries; verify `tf-vulkan` / `tf_vulkan` name availability on GitHub and PyPI before publishing.

## 13. Repository structure

```text
tf-vulkan/
├── README.md
├── docs/
│   ├── architecture.md
│   ├── api-compatibility.md
│   ├── devices.md
│   ├── benchmark-methodology.md
│   └── troubleshooting.md
├── cmake/                   # pinned Android NDK / Vulkan toolchain
├── third_party/             # manifests, patches, hashes; avoid opaque vendoring
├── src/
│   ├── tensorflow_plugin/   # device and kernel registration via C APIs
│   ├── runtime/             # queues, buffers, allocator, events, transfers
│   ├── kernels/             # native eager shader dispatch
│   ├── shaders/             # SPIR-V source and build rules
│   ├── graph/               # optional verified graph extraction / lowering
│   ├── iree_bridge/         # optional VMFB execution path
│   └── diagnostics/
├── python/tf_vulkan/        # packaging, explicit compile API, diagnostics
├── tests/
│   ├── abi/
│   ├── conformance/
│   ├── integration/
│   ├── training/
│   └── fuzz/
├── benchmarks/
│   ├── matmul.py
│   ├── eager_chain.py
│   ├── mlp_inference.py
│   ├── cnn_inference.py
│   └── mlp_training.py
├── examples/
└── ci/
    ├── desktop-vulkan/
    └── android-adb/
```

## 14. Implementation sequence and issue backlog

**Timelines are rough effort envelopes for one experienced GPU/runtime engineer, assuming the host app's TensorFlow already works and compatible hardware is available. They are not delivery commitments.**

| ID | Task | Dependencies | Deliverable | Rough effort |
|---|---|---|---|---|
| 000 | F0: full TF in host app + plugin APIs compiled in | none | Pass/fail report | 1–3 days |
| 001 | Inventory host app TensorFlow Android ABI | 000 | Compatibility manifest + symbol dump | 2–4 days |
| 001b | Evaluate Vulten (`jswag180/Vulten`, Apache-2.0) for Android port vs reference (`F1c`) | 001 | Port/fork vs clean-room decision + license/code audit | 2–4 days |
| 002 | Android Vulkan probe | none | Device feature report + clear tier selection | 1–3 days |
| 003 | Minimal TF plugin registration + D1 experiment | 001, 001b | Device listed under chosen type; F1b report | 5–12 days |
| 004 | Vulkan context/queue/allocator/fence | 001b, 002 | Stable native compute runtime | 5–12 days |
| 005 | Device tensor alloc + transfers | 003,004 | Round-trip with no leak/UAF | 5–10 days |
| 006 | `MatMul` dispatch through TF | 005 | Strict-placement P0 demo | 5–12 days |
| 007 | Pointwise + reductions + plumbing kernels + deferred submission | 006 | Eager chain with residency | 3–6 weeks |
| 007b | Read-only variable residency (K0.5) | 007 | Keras weights stay on device across calls | 1–2 weeks |
| 008 | Benchmarks + Android diagnostics | 006 | Cross-device benchmark reports | 1–2 weeks |
| 009 | IREE VMFB standalone POC | 002 | Host compilation → phone execution | 3–7 days |
| 010 | TF graph capture/lowering feasibility | 003,009 | Verified one-subgraph round trip or blocker report | 1–3 weeks |
| 011 | Optional compiled-region execution | 005,010 | Cached graph execution + parity | 3–8 weeks |
| 012 | Dense inference and fallback policy | 007, 007b | P2 MLP milestone | 2–5 weeks |
| 013 | Conv2D + CNN integration | 012 | Small CNN workload | 3–8 weeks |
| 014 | Training + optimizer/resource semantics | 007,012 | P3 MLP training | 4–10 weeks |
| 016 | Opt-in FP16 path (K1.5) | 007, 008 | FP16 vs FP32 benchmarks per device | 2–4 weeks |
| 015 | Release engineering | 003–014, 016 as applicable | reproducible Android package, CI matrix, docs | 2–4 weeks |

**Totals (one engineer, no parallelism):** roughly 27–64 engineer-weeks for the full backlog (potentially compressed on 003–007b and 014 if Vulten's plugin scaffolding and kernels are ported cleanly in `001b`). The critical path to P2 (000 → 001 → 001b → 003 → 005 → 006 → 007 → 007b → 012, with 002/004 alongside) is roughly 10–22 weeks. Items 009–010 can run in parallel only with a second person, or else extend the calendar.

### Suggested milestone calendar

**Sprint 0: feasibility, ~2–3 weeks.** Establish 000–003 (including the `001b` Vulten evaluation) and standalone Vulkan path. Produce go/no-go artifacts; if TF plugin ABI isn't viable, stop and redesign.

**Sprint 1: real GPU MatMul, ~2–4 further weeks.** Land 004–006, strict placement, measured Vulkan dispatch, correctness and resource stability.

**Sprint 2: useful eager backend, ~5–10 further weeks.** Land 007, 007b, 008 and selected 012 operations. Aim for a Dense inference demo with device-resident weights.

**Parallel research:** 009–010 are independent feasibility spikes; do not block eager progress or assume success.

**Later:** 011/013/014 only after measured usefulness; prioritize whichever moves end-to-end workloads most.

## 15. Tests required before each milestone

### Test pyramid

- **Shader unit:** shader arithmetic vs CPU in isolation, boundary dimensions, transpose flags, non-multiples of workgroup size.
- **Memory:** CPU→GPU→CPU parity, each memory-type path (UMA mapped, staging, imported host memory if enabled), multiple queues if enabled, aligned and unaligned sizes, OOM, 10k lifetime cycles, ASan/UBSan where supported.
- **TensorFlow integration:** true `/GPU:0` placement on the Vulkan subtype, weight residency for Keras models, disabled soft placement, dtype/shape errors, repeated imports and unload behavior, eager vs `tf.function` parity.
- **Numerics:** FP32 reference with seeded values, NaN/Inf, extreme magnitudes, reduced precision disclosed, gradient checks.
- **Training:** 100-step regression/classification, stable loss trajectory, checkpoint serialization and resumption.
- **Compatibility:** driver feature gate, unsupported device diagnostics, old driver regression, TF version mismatch.
- **Performance:** microbench + model-level A/B, correctness and fallback counters attached to every result.

### Explicit P0 acceptance script (illustrative)

```python
import numpy as np
import tensorflow as tf

tf.config.set_soft_device_placement(False)
assert tf.config.list_physical_devices('GPU'), 'Plugin did not register'
# Plus: tf_vulkan.status() must report subtype VULKAN (per D1).

rng = np.random.default_rng(1234)
# Odd shapes: M, N, K not multiples of any plausible tile or workgroup size.
for (m, k, n) in [(1, 1, 1), (7, 13, 5), (509, 517, 503), (1024, 1024, 1024)]:
    a_np = rng.standard_normal((m, k), dtype=np.float32)
    b_np = rng.standard_normal((k, n), dtype=np.float32)
    # CPU creation avoids making random-kernel support a P0 prerequisite.
    a, b = tf.constant(a_np), tf.constant(b_np)

    with tf.device('/GPU:0'):
        result = tf.matmul(a, b)

    expected = a_np.astype(np.float64) @ b_np.astype(np.float64)
    # Tolerance scales with K because FP32 error grows with accumulation length.
    tol = 1e-5 * np.sqrt(k) * np.abs(expected).max()
    np.testing.assert_allclose(result.numpy(), expected, rtol=0, atol=tol)
    print(m, k, n, 'executed device:', result.device)
```

The identity-matrix test from revision 1.0 is dropped: multiplying by the identity is exact and does not exercise accumulation, so a broken inner loop or tiling bug could still pass.

**Additional mandatory evidence:** Vulkan queue submission capture or timestamp associated with the TensorFlow op, supported device-buffer allocation, and **zero CPU MatMul fallback**. The device string alone is not sufficient proof.

## 16. Risk register

| Risk | Impact | Mitigation / falsification |
|---|---|---|
| PluggableDevice binary incompatibility with Android TensorFlow | Critical | ABI manifest and load test before GPU work |
| Device/kernel C APIs insufficient for reliable required semantics | Critical | Minimal kernel and tensor lifetime spike; version-pinned fork only if necessary |
| IREE Vulkan requires features missing on target handset | High | Probe features; native-only tier; supported-device matrix |
| TensorFlow graph capture/lowering incompatible or unsupported | High | Independent short spike; explicit compiled API as fallback |
| IREE/native allocator incompatibility | High | Explicit transfer boundaries first; no speculative zero-copy |
| Missing gradients/resource operations | High | Training isolated as P3; support selected workload first |
| Cross-device data transfers erase speedup | High | GPU residency, fallback counters, whole-program benchmarks |
| Driver quirks/device loss | High | Multiple actual phones, feature gates, error handling |
| GPU thermal throttling | Medium | sustained tests, transparent performance reports |
| Compiler/runtime binaries too large for the host app | High | host AOT, strip binaries, minimal features, optional downloaded module |
| TF API drift | Medium | pinned supported versions and CI release matrix |
| Host app TF built without plugin C APIs | Critical | Gate F0; rebuild TF with plugin support or stop |
| Device type choice (D1) breaks placement or Keras checks | High | F1b experiment before PR #2 |
| Weights left on CPU erase inference speedup | High | K0.5 variable residency before P2; transfer counters |
| Per-op fence waits cap eager performance | High | Deferred submission required at P1 |
| GPU watchdog resets on long dispatches | High | Chunked dispatch; per-driver limit measured in F3 |
| FP32-only throughput too low for speedup gates | Medium | Opt-in FP16 (016); report FP32 and FP16 separately |
| Phone-captured graphs differ from host-lowered graphs | Medium | Round-trip workflow (§8); signature comparison in 010 |
| UMA mapped `memcpy` races with unsubmitted deferred command buffers | Critical | Flush-and-wait rule on `memcpy_dtoh` and `record_event` (§6.2, §6.3) |
| Missing `kernels_experimental.h` symbols (`TF_AssignVariable`) block K0.5/P3 | High | Included in F1 symbol checklist (§5); verify before P2 |
| Android Bionic linker namespace / `RTLD_LOCAL` hides `TF_*` symbols on `dlopen` | High | Test `RTLD_GLOBAL` and `libtensorflow_framework.so` `SONAME` linkage in F2 (§5) |

## 17. Decisions requiring experimental resolution

1. **What exact full TensorFlow build currently runs in the host app, and does it include the plugin C APIs?** Python version, `tensorflow.__version__`, package origin, C++ ABI and build flags (F0).
2. **Can it load a modern PluggableDevice plugin on Android, and by which load mechanism?** This is the main architecture gating question (F1).
3. **Should we port/fork Vulten (`jswag180/Vulten`, Apache-2.0) to Android and contribute back, or build from scratch using Vulten as a reference?** Evaluate in `001b` / `F1c` before writing plugin boilerplate.
4. **Does registering as type `GPU` with subtype `VULKAN` work better than a custom type (`VULKAN` or Vulten's `VULK`)?** (D1, F1b)
5. **What Vulkan API/features does the first phone actually expose?** Including IREE-required 1.3 baseline, memory types (UMA), FP16 support, and the practical watchdog limit.
6. **Do relevant TF extension hooks (`TF_InitGraph`) expose enough graph capture and rewriting for the selected TF build?** If not, retain eager backend and explicit `compile()`.
7. **How costly is a TensorFlow↔IREE buffer boundary?** Benchmark explicit copies before proposing shared allocations.
8. **Will the first supported Keras workload be MLP or CNN?** Choose based on actual operator coverage and user demand, not marketing.

## 18. The exact first pull request

**PR title:** `feat: probe Android TensorFlow plugin ABI and Vulkan compute capabilities`

Acceptance checklist:

- [ ] Confirm full TensorFlow imports and runs eager + `tf.function` on CPU inside the host app (F0).
- [ ] Print Python, TensorFlow, Android ABI, NDK/libc++ and loader configuration.
- [ ] Print exported TensorFlow plugin-related symbols against the §5 entry-point checklist and identify supported header revisions.
- [ ] Record which plugin load mechanism works (auto-load directory vs explicit loader).
- [ ] A standalone Android Vulkan compute program calculates 256×256 elementwise add correctly.
- [ ] Print GPU name, vendor/driver, Vulkan API version, required feature/extension matrix, memory types/heaps, and FP16 support.
- [ ] Minimal plugin `.so` loads into the same host app process as TensorFlow, or records precise missing symbol/error.
- [ ] Test on at least one real non-rooted Android phone.
- [ ] Save logs under a reproducible test fixture and publish a pass/fail table.
- [ ] No claim of TensorFlow acceleration yet.

**PR #2:** run `001b`/`F1c` (Vulten port vs reference evaluation) and `F1b`, settle D1, register the device, allocate/copy tensors.  
**PR #3:** integrate genuine Vulkan-backed `MatMul` with strict placement + profiler evidence.  
**PR #4:** chained pointwise ops + memory residency + deferred submission.  
**PR #5:** read-only variable residency for Keras weights.  
**Research RFC:** optional IREE subgraph compiler feasibility, isolated from main eager backend.

## 19. Reference implementation and reading list

Project/documentation entrypoints. Links point at moving targets (`master`, `latest`); before coding, replace each with the page or tag matching the pinned TensorFlow and IREE versions in the compatibility manifest.

1. **Vulten — existing Vulkan PluggableDevice for TensorFlow (primary prior art & candidate upstream base, Apache-2.0):** https://github.com/jswag180/Vulten
2. TensorFlow GPU device plugin docs: https://www.tensorflow.org/install/gpu_plugins
3. TensorFlow PluggableDevice tutorial: https://github.com/tensorflow/community/blob/master/rfcs/20200624-pluggable-device-for-tensorflow/tutorial.md
4. TensorFlow PluggableDevice architecture overview: https://blog.tensorflow.org/2021/06/pluggabledevice-device-plugins-for-TensorFlow.html
5. IREE official overview and model/import/backend matrix: https://iree.dev/
6. IREE Vulkan deployment requirements: https://iree.dev/guides/deployment-configurations/gpu-vulkan/
7. Khronos Vulkan + IREE deployment discussion: https://docs.vulkan.org/tutorial/latest/ML_Inference/Third_Party_Libraries/12_iree.html
8. TensorFlow Lite GPU source (use as baseline/reference, not equivalent full-TF implementation): https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/gpu/README.md
9. TensorFlow Lite GPU delegate limitations and CPU/GPU partition overhead: https://android.googlesource.com/platform/external/tensorflow/+/upstream-master/tensorflow/lite/g3doc/performance/gpu.md (old Android mirror; replace with current LiteRT GPU documentation)
10. Microsoft TensorFlow DirectML plugin (architectural precedent): https://github.com/microsoft/tensorflow-directml-plugin
11. Apple TensorFlow Metal (architectural precedent): https://developer.apple.com/metal/tensorflow-plugin/
12. Plugin C API headers at the pinned TF commit: `tensorflow/c/experimental/stream_executor/stream_executor.h`, `tensorflow/c/kernels.h`, `tensorflow/c/kernels_experimental.h`, `tensorflow/c/tf_tensor.h`, `tensorflow/c/experimental/grappler/grappler.h`, `tensorflow/c/experimental/pluggable_profiler/pluggable_profiler.h`

---

### Final recommendation

**Build the smallest honest full-TensorFlow Vulkan device first.** Evaluate Vulten (`jswag180/Vulten`) first so we do not reinvent working `SE_InitPlugin`/`TF_InitKernel` scaffolding or basic shaders unnecessarily, while addressing its desktop-centric assumptions (runtime `shaderc` compilation, custom `VULK` device type, and ~90% host-transfer overhead). A proven `tf.matmul` dispatched from TensorFlow Python through a PluggableDevice to an Android Vulkan queue is a more valuable engineering achievement than a large collection of standalone shaders or an unintegrated IREE demo. Add IREE only where graph capture, compilation, and transfer economics are experimentally validated. Ship a narrow backend that is correct, benchmarked, and clear about unsupported operations, then grow coverage around actual Keras workloads.

---

## Appendix A: Glossary

| Term | Meaning |
|---|---|
| BDA | Buffer Device Address (`bufferDeviceAddress`), an optional faster Vulkan executable ABI in IREE (§5) |
| D1 | Decision on the TensorFlow device type the plugin registers (§5) |
| HAL | IREE Hardware Abstraction Layer; the Vulkan HAL driver executes compiled modules on Vulkan |
| IREE | MLIR-based ML compiler and runtime used for the optional compiled-graph path |
| NextPluggableDevice | Newer OpenXLA/PJRT-based TensorFlow device plugin architecture, contrasted with classic StreamExecutor `PluggableDevice` (§1, §5) |
| PJRT | OpenXLA uniform device runtime C API used by JAX, PyTorch/XLA, and `NextPluggableDevice` |
| PluggableDevice | TensorFlow mechanism for loading out-of-tree device plugins through stable StreamExecutor/Kernel C APIs |
| Soft placement | TensorFlow behaviour that silently moves ops without a device kernel to CPU |
| SPIR-V | Binary shader intermediate representation consumed by Vulkan drivers |
| StableHLO | MLIR dialect for ML operations used as an input to IREE |
| Strict mode | Soft placement disabled; unsupported device ops raise errors (§9) |
| Tier A/B/C | Device capability tiers (§5) |
| UMA | Unified memory architecture: CPU and GPU share physical memory, as on Adreno and Mali |
| VMFB | IREE's compiled module format (VM FlatBuffer) |
| Vulten | Existing open-source desktop Vulkan `PluggableDevice` for TensorFlow (`jswag180/Vulten`, Apache-2.0) evaluated in task `001b` / gate `F1c` |
