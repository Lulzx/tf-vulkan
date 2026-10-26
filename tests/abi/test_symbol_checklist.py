"""ABI gate wrapper (spec §5 F1): runs tf_vulkan.probe_tf."""
import os, subprocess, sys
root = os.path.join(os.path.dirname(__file__), "..", "..")
env = {**os.environ, "PYTHONPATH": os.path.join(root, "python")}
r = subprocess.run([sys.executable, "-m", "tf_vulkan.probe_tf"], capture_output=True, text=True, env=env)
print(r.stdout); print(r.stderr, file=sys.stderr)
raise SystemExit(0 if "F0" in r.stdout else 2)  # report only; gate result is in the table
