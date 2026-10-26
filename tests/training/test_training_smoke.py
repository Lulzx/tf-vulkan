"""Training smoke (spec §10): fresh-process SGD loop on /GPU:0.

Loss must decrease over 20 steps. This is a smoke test, not the full P3
claim (which needs 100-step parity + checkpoint/resume, §15).
Run: VK_ICD_FILENAMES=<icd> .venv-tf/bin/python tests/training/test_training_smoke.py
"""
import os
import sys

PLUGIN = os.environ.get("TF_VULKAN_PLUGIN", os.path.join(
    os.path.dirname(__file__), "..", "..", "build-plugin", "libtf_vulkan.dylib"))


def main() -> int:
    try:
        import numpy as np
        import tensorflow as tf
        from tensorflow.python.framework.load_library import load_pluggable_device_library
    except Exception as exc:
        print(f"NO-GO F0: {exc}")
        return 2
    if not os.path.exists(PLUGIN):
        print(f"NO-GO: plugin not built at {PLUGIN}")
        return 2
    load_pluggable_device_library(PLUGIN)
    tf.config.set_soft_device_placement(False)
    rng = np.random.default_rng(7)
    with tf.device("/GPU:0"):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(4,)),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1),
        ])
        opt = tf.keras.optimizers.SGD(learning_rate=0.1)
        xs = tf.constant(rng.standard_normal((16, 4), dtype=np.float32))
        ys = tf.constant(rng.standard_normal((16, 1), dtype=np.float32))
        losses = []
        for _ in range(100):  # §15: 100 consecutive training updates
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.square(model(xs) - ys))
            grads = tape.gradient(loss, model.trainable_variables)
            assert all(g is not None for g in grads), "missing gradient"
            opt.apply_gradients(zip(grads, model.trainable_variables))
            losses.append(float(loss))
    print("losses first/last:", losses[0], losses[-1])
    assert all(map(lambda v: v == v and abs(v) != float("inf"), losses)), losses
    assert losses[-1] < losses[0], losses
    # Adam one-step sanity (optimizer state + update on device).
    with tf.device("/GPU:0"):
        v = tf.Variable(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
        adam = tf.keras.optimizers.Adam(learning_rate=0.1)
        adam.apply_gradients([(tf.constant([[0.5, 0.5], [0.5, 0.5]]), v)])
        assert abs(float(v.numpy()[0, 0]) - 0.9) < 1e-3, v.numpy()
        print("Adam one-step OK:", v.numpy().tolist())
    # CNN training: Conv2D backprops + pooling through a real optimizer.
    with tf.device("/GPU:0"):
        cnn = tf.keras.Sequential([
            tf.keras.Input(shape=(8, 8, 3)),
            tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(2),
        ])
        copt = tf.keras.optimizers.SGD(learning_rate=0.05)
        cx = tf.constant(rng.standard_normal((8, 8, 8, 3), dtype=np.float32))
        cy = tf.constant(rng.standard_normal((8, 2), dtype=np.float32))
        closses = []
        for _ in range(10):
            with tf.GradientTape() as tape:
                closs = tf.reduce_mean(tf.square(cnn(cx) - cy))
            cgrads = tape.gradient(closs, cnn.trainable_variables)
            assert all(g is not None for g in cgrads), "missing CNN gradient"
            copt.apply_gradients(zip(cgrads, cnn.trainable_variables))
            closses.append(float(closs))
    assert closses[-1] < closses[0], closses
    print("CNN training losses first/last:", closses[0], closses[-1])
    print("TRAINING SMOKE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
