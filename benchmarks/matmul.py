"""GEMM benchmark entry (spec §11). Native numbers come from tfvk_bench;
with TF present this also times TF CPU for the A/B table."""
import subprocess, sys
r = subprocess.run(["tfvk_bench"] + sys.argv[1:], capture_output=True, text=True)
print(r.stdout or r.stderr)
try:
    import time, numpy as np, tensorflow as tf
    for s in (128, 512, 1024):
        a = tf.random.normal((s, s)); b = tf.random.normal((s, s))
        for _ in range(3): tf.matmul(a, b)
        t = time.perf_counter()
        for _ in range(10): tf.matmul(a, b).numpy()
        print(f"TF CPU MatMul {s}: {(time.perf_counter()-t)/10*1e3:.3f} ms")
except Exception as e:
    print(f"TF CPU comparison unavailable: {e}")
