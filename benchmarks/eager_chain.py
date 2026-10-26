"""Eager chain MatMul->BiasAdd->Relu->MatMul (spec §11). TF-gated."""
try:
    import tensorflow as tf
    print("devices:", tf.config.list_physical_devices("GPU"))
    print("Run under /GPU:0 with strict placement; attach fallback counters.")
except Exception as e:
    print(f"NO-GO F0: {e}")
