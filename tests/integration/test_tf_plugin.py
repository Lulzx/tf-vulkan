"""TensorFlow integration tests (spec §15, gates F1/F5, P0-P3 smoke).

Requires: TensorFlow (desktop harness `.venv-tf`) and the built plugin
(`build-plugin/libtf_vulkan.dylib|.so`). Reports NO-GO (exit 2) when either
is missing — never passes silently.

Run:
  VK_ICD_FILENAMES=<icd> .venv-tf/bin/python tests/integration/test_tf_plugin.py
"""
import os
import sys

PLUGIN = os.environ.get(
    "TF_VULKAN_PLUGIN",
    os.path.join(os.path.dirname(__file__), "..", "..", "build-plugin",
                 "libtf_vulkan.dylib"))


def main() -> int:
    try:
        import numpy as np
        import tensorflow as tf
        from tensorflow.python.framework.load_library import (
            load_pluggable_device_library)
    except Exception as exc:
        print(f"NO-GO F0: TensorFlow unavailable ({exc})")
        return 2
    if not os.path.exists(PLUGIN):
        print(f"NO-GO: plugin not built at {PLUGIN} "
              "(cmake -DTFVK_BUILD_TF_PLUGIN=ON -DTFVK_TF_INCLUDE_DIR=<tf include>)")
        return 2

    load_pluggable_device_library(PLUGIN)
    gpus = tf.config.list_physical_devices("GPU")
    assert gpus, "F1/F5: plugin did not register GPU device"
    print("registered:", gpus)
    tf.config.set_soft_device_placement(False)  # strict (§9, §15)

    rng = np.random.default_rng(1234)
    # P0 acceptance (§15): odd shapes, float64 reference, K-scaled tolerance.
    for (m, k, n) in [(1, 1, 1), (7, 13, 5), (509, 517, 503), (256, 256, 256)]:
        a_np = rng.standard_normal((m, k), dtype=np.float32)
        b_np = rng.standard_normal((k, n), dtype=np.float32)
        with tf.device("/GPU:0"):
            r = tf.matmul(tf.constant(a_np), tf.constant(b_np))
        expected = a_np.astype(np.float64) @ b_np.astype(np.float64)
        tol = 1e-5 * np.sqrt(k) * max(1.0, np.abs(expected).max())
        np.testing.assert_allclose(r.numpy(), expected, rtol=0, atol=tol)
        assert "GPU:0" in r.device, r.device
        print(f"P0 matmul {m}x{k}x{n} OK on {r.device}")

    # Batched MatMul with batch broadcast.
    with tf.device("/GPU:0"):
        ba = tf.constant(rng.standard_normal((3, 4, 5), dtype=np.float32))
        bb = tf.constant(rng.standard_normal((5, 2), dtype=np.float32))
        br = tf.matmul(ba, bb)
        a2 = tf.constant(rng.standard_normal((3, 5, 4), dtype=np.float32))
        b2 = tf.constant(rng.standard_normal((3, 5, 2), dtype=np.float32))
        bt = tf.matmul(a2, b2, transpose_a=True)
    np.testing.assert_allclose(br.numpy(), ba.numpy() @ bb.numpy(), atol=1e-4)
    np.testing.assert_allclose(
        bt.numpy(), np.transpose(a2.numpy(), (0, 2, 1)) @ b2.numpy(), atol=1e-4)
    assert br.shape == (3, 4, 2) and bt.shape == (3, 4, 2)
    print("batched MatMul + transpose OK:", br.shape)

    # Resident eager chain (§6.3/P1).
    with tf.device("/GPU:0"):
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        y = tf.nn.relu(tf.matmul(x, x) + tf.constant([1.0, -10.0]))
    np.testing.assert_allclose(y.numpy(), [[8.0, 0.0], [16.0, 12.0]])
    print("eager chain MatMul->Add(broadcast)->Relu OK")

    # K0.5 variable residency smoke: variable created and read on device.
    with tf.device("/GPU:0"):
        v = tf.Variable(rng.standard_normal((4, 4), dtype=np.float32))
        inp = tf.constant(rng.standard_normal((2, 4), dtype=np.float32))
        out = tf.matmul(inp, v)
        v.assign_sub(tf.ones((4, 4)) * 0.0)
    np.testing.assert_allclose(out.numpy(), inp.numpy() @ v.numpy(), atol=1e-4)
    print("variable matmul + assign_sub OK on", out.device)

    # Keras MLP inference (P2 smoke).
    with tf.device("/GPU:0"):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(4,)),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(3),
        ])
        res = model(tf.constant(rng.standard_normal((2, 4), dtype=np.float32)))
    assert res.shape == (2, 3) and "GPU:0" in res.device
    print("Keras Dense MLP inference OK:", res.shape, res.device)

    # tf.function (graph mode through the same kernels).
    @tf.function
    def fn(a, b):
        return tf.nn.relu(tf.matmul(a, b))

    with tf.device("/GPU:0"):
        r = fn(tf.constant(rng.standard_normal((8, 8), dtype=np.float32)),
               tf.constant(rng.standard_normal((8, 8), dtype=np.float32)))
    assert r.shape == (8, 8)
    print("tf.function MatMul+Relu OK")

    # Gradient parity vs analytic CPU values (§10/P3 smoke).
    with tf.device("/GPU:0"):
        w = tf.Variable(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(tf.matmul(w, w))
        grad = tape.gradient(loss, w)
    np.testing.assert_allclose(grad.numpy(), [[7.0, 11.0], [9.0, 13.0]], atol=1e-4)
    assert float(loss) == 54.0
    print("GradientTape MatMul+Sum gradient parity OK")

    # Conv2D NHWC (K2) vs CPU reference + tiny Keras CNN.
    with tf.device("/GPU:0"):
        cx = tf.constant(rng.standard_normal((1, 5, 5, 3), dtype=np.float32))
        cf = tf.constant(rng.standard_normal((3, 3, 3, 4), dtype=np.float32))
        cy = tf.nn.conv2d(cx, cf, strides=1, padding="SAME")
    with tf.device("/CPU:0"):
        cy_ref = tf.nn.conv2d(cx, cf, strides=1, padding="SAME")
    np.testing.assert_allclose(cy.numpy(), cy_ref.numpy(), atol=1e-4)
    with tf.device("/GPU:0"):
        cnn = tf.keras.Sequential([
            tf.keras.Input(shape=(5, 5, 3)),
            tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2),
        ])
        cout = cnn(cx)
    assert cout.shape == (1, 2) and "GPU:0" in cout.device
    print("Conv2D + tiny CNN OK:", cout.shape)

    # Pooling + depthwise + Conv2D gradients vs CPU (K2).
    with tf.device("/GPU:0"):
        mp = tf.nn.max_pool(cx, ksize=2, strides=2, padding="VALID")
        ap = tf.nn.avg_pool(cx, ksize=2, strides=2, padding="VALID")
        dwf = tf.constant(rng.standard_normal((3, 3, 3, 2), dtype=np.float32))
        dw = tf.nn.depthwise_conv2d(cx, dwf, strides=[1, 1, 1, 1], padding="SAME")
    with tf.device("/CPU:0"):
        mp_r = tf.nn.max_pool(cx, ksize=2, strides=2, padding="VALID")
        ap_r = tf.nn.avg_pool(cx, ksize=2, strides=2, padding="VALID")
        dw_r = tf.nn.depthwise_conv2d(cx, dwf, strides=[1, 1, 1, 1], padding="SAME")
    np.testing.assert_allclose(mp.numpy(), mp_r.numpy(), atol=1e-5)
    np.testing.assert_allclose(ap.numpy(), ap_r.numpy(), atol=1e-5)
    np.testing.assert_allclose(dw.numpy(), dw_r.numpy(), atol=1e-4)
    print("MaxPool/AvgPool/Depthwise OK")
    # Conv2D gradient parity (autodiff on both devices).
    grads = {}
    for dev in ("/CPU:0", "/GPU:0"):
        with tf.device(dev):
            xv = tf.Variable(cx.numpy())
            fv = tf.Variable(cf.numpy())
            with tf.GradientTape() as tape:
                loss = tf.reduce_sum(tf.nn.conv2d(xv, fv, strides=1, padding="SAME"))
            grads[dev] = [g.numpy() for g in tape.gradient(loss, [xv, fv])]
    np.testing.assert_allclose(grads["/GPU:0"][0], grads["/CPU:0"][0], atol=1e-3)
    np.testing.assert_allclose(grads["/GPU:0"][1], grads["/CPU:0"][1], atol=1e-3)
    print("Conv2D backprop input/filter gradient parity OK")

    # Opt-in FP16 (K1.5): gated on shaderFloat16, FP32 accumulation.
    with tf.device("/GPU:0"):
        fa = tf.constant(rng.standard_normal((32, 32), dtype=np.float32) * 0.5, dtype=tf.float16)
        fb = tf.constant(rng.standard_normal((32, 32), dtype=np.float32) * 0.5, dtype=tf.float16)
        fr = tf.matmul(fa, fb)
        assert fr.dtype == tf.float16
        err = np.abs(fr.numpy().astype(np.float32)
                     - fa.numpy().astype(np.float32) @ fb.numpy().astype(np.float32)).max()
        assert err < 0.01, err
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        try:
            f16_model = tf.keras.Sequential([
                tf.keras.Input(shape=(16,)),
                tf.keras.layers.Dense(8, activation="relu"),
                tf.keras.layers.Dense(4),
            ])
            fout = f16_model(tf.constant(rng.standard_normal((2, 16), dtype=np.float32)))
            assert fout.shape == (2, 4) and "GPU:0" in fout.device
        finally:
            tf.keras.mixed_precision.set_global_policy("float32")
    print("FP16 matmul (err %.4f) + mixed_float16 Dense OK" % err)

    print("TF INTEGRATION PASS (desktop harness; Android gates remain, §4)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
