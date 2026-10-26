"""Python API tests: status honesty, partitioning, cache keys (spec §6.5/§8/§9)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
import tf_vulkan  # noqa: E402


def test_status_never_claims_without_probe():
    s = tf_vulkan.status()
    assert s["device_subtype"] == "VULKAN"
    assert s["vulkan"]["tier"] in ("A", "B", "C")
    assert "note" in s


def test_mode_validation(monkeypatch=None):
    assert tf_vulkan.mode() in ("strict", "balanced", "cpu")


def test_partition_inserts_boundaries_at_stateful_ops():
    regions = tf_vulkan.partition_ops(
        ["MatMul", "BiasAdd", "Relu", "AssignVariableOp", "MatMul", "NoSuchOp", "Softmax"])
    kinds = [r["kind"] for r in regions]
    assert kinds == ["compiled_candidate", "boundary", "compiled_candidate",
                     "boundary", "compiled_candidate"]
    assert regions[1]["reason"] == "stateful"
    assert regions[3]["reason"] == "unsupported"
    # Stateful ops are never inside a compiled candidate (spec §8/§10).
    for r in regions:
        if r["kind"] == "compiled_candidate":
            assert not (set(r["ops"]) & tf_vulkan._STATEFUL_OPS)


def test_cache_key_changes_with_versions_and_shapes():
    base = dict(input_shapes=[[1, 4]], input_dtypes=["float32"], tf_version="2.x",
                converter_version="c1", iree_version="i1", device_fingerprint="d1")
    k1 = tf_vulkan.cache_key("graph", **base)
    assert k1 == tf_vulkan.cache_key("graph", **base)
    assert k1 != tf_vulkan.cache_key("graph", **{**base, "iree_version": "i2"})
    assert k1 != tf_vulkan.cache_key("graph", **{**base, "input_shapes": [[2, 4]]})
    assert len(k1) == 64


def test_compile_is_explicit_not_silent():
    try:
        tf_vulkan.compile(lambda x: x)
        raise AssertionError("compile must not silently claim acceleration")
    except NotImplementedError as exc:
        assert "F7" in str(exc)


if __name__ == "__main__":
    test_status_never_claims_without_probe()
    test_mode_validation()
    test_partition_inserts_boundaries_at_stateful_ops()
    test_cache_key_changes_with_versions_and_shapes()
    test_compile_is_explicit_not_silent()
    print("python api tests PASS")
