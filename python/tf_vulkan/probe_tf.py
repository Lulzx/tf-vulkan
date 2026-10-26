"""F0/F1/F2 feasibility probe (spec §4, §5, §18 PR #1).

Prints a pass/fail table. Exit 0 only if F0 and F1 pass. Never claims
acceleration. Run: python -m tf_vulkan.probe_tf [--tf-lib PATH]
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import os
import subprocess
import sys

# F1 symbol checklist (spec §5). TF_* symbols must be exported by the host
# TensorFlow build; SE_InitPlugin/TF_InitKernel are exported by the plugin.
PLUGIN_SYMBOLS = ["SE_InitPlugin", "TF_InitKernel", "TF_InitGraph", "TF_InitProfiler"]
TF_SYMBOLS = [
    "TF_NewKernelBuilder", "TF_RegisterKernelBuilder", "TF_GetInput",
    "TF_AssignVariable", "TF_AssignUpdateVariable",
    "TF_VariableInputLockHolder", "TF_GetInputTensorFromVariable",
]
HEADERS = [
    "tensorflow/c/experimental/stream_executor/stream_executor.h",
    "tensorflow/c/kernels.h",
    "tensorflow/c/kernels_experimental.h",
    "tensorflow/c/tf_tensor.h",
    "tensorflow/c/experimental/grappler/grappler.h",
    "tensorflow/c/experimental/pluggable_profiler/pluggable_profiler.h",
]


def _nm_symbols(lib: str) -> set[str]:
    nm = "nm"
    try:
        out = subprocess.run([nm, "-gU", lib], capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            out = subprocess.run([nm, "-D", "--defined-only", lib],
                                 capture_output=True, text=True, timeout=60)
        syms = set()
        for line in out.stdout.splitlines():
            parts = line.split()
            if parts:
                syms.add(parts[-1].lstrip("_"))
        return syms
    except Exception:
        return set()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf-lib", default="", help="libtensorflow_framework / _pywrap .so to inspect")
    ap.add_argument("--header-root", default="", help="TF source dir to check headers under")
    args = ap.parse_args()
    rows: list[tuple[str, str, str]] = []

    # F0: full TF imports, eager + tf.function on CPU.
    tf = None
    try:
        import tensorflow as tf
        x = tf.constant([1.0, 2.0]) + 1.0
        f = tf.function(lambda v: tf.matmul(v, v))
        y = f(tf.eye(2))
        rows.append(("F0 full TF in host app", "PASS",
                     f"tf={tf.__version__} eager={x.numpy().tolist()} fn={y.numpy().shape}"))
    except Exception as exc:
        rows.append(("F0 full TF in host app", "FAIL", f"{type(exc).__name__}: {exc}"))

    # F1: symbol checklist against the actual TF binary.
    libs = [args.tf_lib] if args.tf_lib else []
    if tf is not None:
        base = os.path.dirname(tf.__file__)
        for pat in ("_pywrap_tensorflow_internal*.so", "_pywrap_tensorflow_internal*.dylib",
                    "libtensorflow_framework*.so", "libtensorflow_framework*.dylib"):
            libs += glob.glob(os.path.join(base, "**", pat), recursive=True)
    syms: set[str] = set()
    for lib in libs:
        syms |= _nm_symbols(lib)
    if libs:
        missing = [s for s in TF_SYMBOLS if s not in syms]
        rows.append(("F1 TF C symbols", "PASS" if not missing else "FAIL",
                     f"libs={[os.path.basename(l) for l in libs]} missing={missing}"))
    else:
        rows.append(("F1 TF C symbols", "FAIL", "no TF shared library found to inspect"))

    if args.header_root:
        missing_h = [h for h in HEADERS if not os.path.exists(os.path.join(args.header_root, h))]
        rows.append(("F1 headers at pinned commit", "PASS" if not missing_h else "FAIL",
                     f"missing={missing_h}"))
    else:
        rows.append(("F1 headers at pinned commit", "SKIP", "pass --header-root <tf source>"))

    # F1b/D1 + load mechanism note.
    rows.append(("F1b device type D1", "INFO",
                 "plugin registers GPU/VULKAN by default; TF_VULKAN_DEVICE_TYPE=VULKAN overrides"))
    rows.append(("F2 load mechanism", "INFO",
                 "try tensorflow-plugins/ auto-load, else "
                 "tf.load_library.load_pluggable_device_library(path); on Android test "
                 "RTLD_GLOBAL / DT_NEEDED SONAME (spec §5)"))

    for name, res, detail in rows:
        print(f"{res:4}  {name}: {detail}")
    failed = any(r == "FAIL" for _, r, _ in rows[:2])
    print("GO" if not failed else "NO-GO: fix F0/F1 before any plugin work (spec §4)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
