# TF Vulkan (`tf-vulkan`)

Full TensorFlow GPU acceleration via Vulkan, **Android-first** (spec rev 1.4,
[spec.md](spec.md)). Android (`arm64-v8a`) is the first and only supported
target; the desktop Vulkan build in this repo is a development/CI harness
only — no desktop or Android speedup is claimed until the spec gates pass
on real hardware.

## Status (honest gate table)

| Gate | What | State on this checkout |
|---|---|---|
| F0 full TF in host app | `import tensorflow` + plugin C APIs in the host build | PASS on desktop (TF 2.21) and on Pixel 9 (TF 2.21 in adb-driven Ubuntu proot, CPU); Android host app still TBD |
| F1/F1b/F1c plugin load, D1, Vulten audit | Symbol checklist, device-type experiment, port-vs-clean-room | Desktop PASS: plugin loads, GPU/VULKAN registered; F1b/F1c open (docs/api-compatibility.md) |
| F2 packaging | Bionic namespaces, `RTLD_GLOBAL`, SONAME | Documented; needs a real device |
| F3 Vulkan compute | Standalone dispatch + feature/tier report | PASS on Pixel 9 (Mali-G715, Tier A) and desktop harness |
| F4 buffer ownership | Alloc/copy/release, 10k cycles | PASS on Pixel 9 + desktop harness |
| F5 TF MatMul on `/GPU:0` | Strict placement + profiler evidence | Desktop harness PASS: strict MatMul, chains, Keras MLP, gradients (tests/integration/test_tf_plugin.py); Android pending |
| F6/F7 IREE / graph lowering | VMFB POC, subgraph round trip | Not implemented; `tf_vulkan.compile()` raises explicitly |

## Build (desktop harness)

```sh
brew install vulkan-headers vulkan-loader molten-vk glslang spirv-tools  # macOS harness
cmake -S . -B build && cmake --build build -j
ctest --test-dir build
VK_ICD_FILENAMES=/opt/homebrew/share/vulkan/icd.d/MoltenVK_icd.json ./build/tfvk_probe
VK_ICD_FILENAMES=/opt/homebrew/share/vulkan/icd.d/MoltenVK_icd.json ./build/tfvk_bench
```

On Linux use your distro Vulkan loader/headers + `glslang-tools`.
Android builds use the pinned NDK toolchain in `cmake/` (forthcoming with
the host app's TF build; see docs/api-compatibility.md).

## Python

```sh
python3 -m venv .venv && .venv/bin/pip install numpy
PYTHONPATH=python .venv/bin/python -m tf_vulkan status
PYTHONPATH=python .venv/bin/python -m tf_vulkan.probe_tf
PYTHONPATH=python .venv/bin/python tests/conformance/test_python_api.py
```

Target experience (once F0–F5 pass on device):

```python
import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))  # GPU, subtype VULKAN
with tf.device("/GPU:0"):
    c = tf.matmul(a, b)
```

## Layout

Per spec §13: `src/runtime` (device/allocator/deferred stream),
`src/kernels` + `src/shaders` (offline SPIR-V eager kernels),
`src/tensorflow_plugin` (C-API entry points, built only against pinned TF
headers), `src/graph` + `src/iree_bridge` (gated research path),
`python/tf_vulkan` (status/probes), `tests/`, `benchmarks/`, `docs/`.

Before publishing: verify `tf-vulkan`/`tf_vulkan` availability on GitHub
and PyPI (spec §1/§12).
