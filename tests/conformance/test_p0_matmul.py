"""P0 acceptance script (spec §15). Requires TensorFlow + the built plugin
registered as GPU/VULKAN. Without TF it reports NO-GO instead of passing."""
import sys


def main() -> int:
    try:
        import numpy as np
        import tensorflow as tf
    except Exception as exc:
        print(f"NO-GO F0: TensorFlow unavailable ({exc}); P0 not run, not passed")
        return 2

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("NO-GO F1/F5: plugin did not register a GPU device; P0 not run")
        return 2

    tf.config.set_soft_device_placement(False)  # strict placement (spec §9)
    rng = np.random.default_rng(1234)
    for (m, k, n) in [(1, 1, 1), (7, 13, 5), (509, 517, 503), (1024, 1024, 1024)]:
        a_np = rng.standard_normal((m, k), dtype=np.float32)
        b_np = rng.standard_normal((k, n), dtype=np.float32)
        a, b = tf.constant(a_np), tf.constant(b_np)
        with tf.device("/GPU:0"):
            result = tf.matmul(a, b)
        expected = a_np.astype(np.float64) @ b_np.astype(np.float64)
        tol = 1e-5 * np.sqrt(k) * max(1.0, np.abs(expected).max())
        np.testing.assert_allclose(result.numpy(), expected, rtol=0, atol=tol)
        print(m, k, n, "executed device:", result.device)
    print("P0 PASS (device string alone is not proof; attach profiler/timestamp "
          "evidence per spec §15)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
