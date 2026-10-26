"""MLP inference benchmark (spec §11, P2). TF-gated; weights must be device-resident (K0.5)."""
try:
    import tensorflow as tf
    m = tf.keras.Sequential([tf.keras.layers.Dense(128, activation="relu", input_shape=(64,)), tf.keras.layers.Dense(10)])
    m.summary()
    print("Time model(x) under /GPU:0 and verify zero H2D weight transfers after load (spec §7 K0.5).")
except Exception as e:
    print(f"NO-GO F0: {e}")
