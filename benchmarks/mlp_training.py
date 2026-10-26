"""MLP training benchmark (spec §10/§11, P3). TF-gated GradientTape check."""
try:
    import tensorflow as tf
    print("Training claim requires gradient parity + 100-step loss test (spec §10); run tests/training first.")
except Exception as e:
    print(f"NO-GO F0: {e}")
