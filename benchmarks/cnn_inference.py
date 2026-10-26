"""CNN benchmark (spec §11, K2). Conv2D NHWC forward is implemented;
run with TF + plugin under strict placement on /GPU:0."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests", "integration"))
print("Use tests/integration/test_tf_plugin.py for correctness; time a Keras "
      "Conv2D model under /GPU:0 with warmup + median of >=30 runs per §11.")
