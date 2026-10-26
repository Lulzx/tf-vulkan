"""tf-vulkan: TensorFlow Vulkan PluggableDevice support (spec rev 1.4).

Android-first. This package never claims GPU acceleration it cannot prove:
`status()` reports the real probed tier and TensorFlow gate results.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess

__version__ = "0.1.0"

_PURE_OPS = {
    "MatMul", "Add", "AddV2", "Sub", "Mul", "Div", "DivNoNan", "Pow", "Maximum",
    "SquaredDifference", "Relu", "Relu6", "Sigmoid", "Tanh", "Exp", "Log",
    "Sqrt", "BiasAdd", "Sum", "Mean", "Softmax", "Identity", "Reshape",
    "Transpose", "ConcatV2", "Cast",
}
_STATEFUL_OPS = {
    "VarHandleOp", "ReadVariableOp", "AssignVariableOp", "AssignAddVariableOp",
    "AssignSubVariableOp", "ResourceApplyAdam", "ResourceApplyGradientDescent",
    "RandomNormal", "RandomUniform", "If", "While", "PyFunc", "Print",
}


def _find_probe() -> str | None:
    env = os.environ.get("TF_VULKAN_PROBE")
    if env and os.path.exists(env):
        return env
    for cand in ("tfvk_probe", os.path.join("build", "tfvk_probe")):
        found = shutil.which(cand) or (cand if os.path.exists(cand) else None)
        if found:
            return found
    return None


def vulkan_probe() -> dict:
    """Run the native Vulkan probe (task 002 / gate F3). Tier C when absent."""
    probe = _find_probe()
    if probe is None:
        return {"tier": "C", "reason": "tfvk_probe binary not built/found",
                "device": None}
    try:
        out = subprocess.run([probe, "--json"], capture_output=True, text=True, timeout=30)
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as exc:  # probe failure is a Tier C diagnostic, not a crash
        return {"tier": "C", "reason": f"probe failed: {exc}", "device": None}


def mode() -> str:
    """Fallback mode (spec §9): strict | balanced | cpu."""
    m = os.environ.get("TF_VULKAN_MODE", "balanced").lower()
    return m if m in ("strict", "balanced", "cpu") else "balanced"


def status() -> dict:
    """Diagnostics (spec §6.5). Real driver details; no implied acceleration."""
    tf_info: dict = {"available": False}
    try:
        import tensorflow as tf

        tf_info = {
            "available": True,
            "version": tf.__version__,
            "gpu_devices": [d.name for d in tf.config.list_physical_devices("GPU")],
            "vulkan_devices": [
                d.name for d in tf.config.list_physical_devices("GPU")
            ],
        }
    except Exception as exc:
        tf_info["error"] = str(exc)
    return {
        "version": __version__,
        "mode": mode(),
        "vulkan": vulkan_probe(),
        "tensorflow": tf_info,
        "device_type": os.environ.get("TF_VULKAN_DEVICE_TYPE", "GPU"),
        "device_subtype": "VULKAN",
        # Native eager kernels actually implemented in src/kernels (§7 K0/K1).
        # Graph-partition candidates are a superset and are not claimed here.
        "supported_ops": sorted({
            "MatMul", "Add", "AddV2", "Sub", "Mul", "Div", "DivNoNan", "Pow",
            "Maximum", "SquaredDifference", "Identity", "Relu", "Relu6",
            "Sigmoid", "Tanh", "Exp", "Log", "Sqrt", "BiasAdd", "Sum", "Mean",
            "Softmax", "Square", "ReluGrad", "BiasAddGrad", "Conv2D",
            "DepthwiseConv2dNative", "Conv2DBackpropFilter",
            "Conv2DBackpropInput", "MaxPool", "AvgPool", "Cast", "Tile",
            "Fill", "AddN", "FloorMod",
        }),
        "note": "GPU execution is claimed only for ops dispatched through the "
                "native plugin under strict placement (spec §9, §15).",
    }


def partition_ops(op_names: list[str]) -> list[dict]:
    """Partition an op list into compilable pure regions (spec §8).

    Boundaries are inserted at stateful, resource, control-flow, or
    unsupported ops. Stateful ops are never compiled as pure fragments.
    """
    regions: list[dict] = []
    current: list[str] = []
    for op in op_names:
        if op in _PURE_OPS and op not in _STATEFUL_OPS:
            current.append(op)
        else:
            if current:
                regions.append({"kind": "compiled_candidate", "ops": current})
                current = []
            regions.append({"kind": "boundary", "ops": [op],
                            "reason": "stateful" if op in _STATEFUL_OPS else "unsupported"})
    if current:
        regions.append({"kind": "compiled_candidate", "ops": current})
    return regions


def cache_key(graph_repr: str, *, input_shapes, input_dtypes, tf_version: str,
              converter_version: str, iree_version: str, device_fingerprint: str,
              constants_hash: str = "") -> str:
    """Artifact cache key (spec §8): sha256 over canonical graph + versions."""
    canonical = json.dumps({
        "graph": graph_repr, "shapes": input_shapes, "dtypes": input_dtypes,
        "tf": tf_version, "converter": converter_version, "iree": iree_version,
        "device": device_fingerprint, "constants": constants_hash,
    }, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def compile(tf_function, example_inputs=None):  # noqa: A001 - explicit API per spec §8
    """Explicit opt-in compiled-region API (spec §8 fallback route).

    Graph capture/lowering is a research spike (task 010, gate F7). Until it
    passes on the pinned TF/IREE versions this raises rather than silently
    claiming graph acceleration.
    """
    raise NotImplementedError(
        "tf_vulkan.compile: graph lowering not yet validated on a pinned "
        "TF/IREE build (gate F7, task 010). Use the eager plugin path or "
        "TensorFlow CPU; no graph acceleration is claimed.")


__all__ = ["status", "vulkan_probe", "mode", "partition_ops", "cache_key", "compile"]
