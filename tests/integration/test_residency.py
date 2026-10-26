"""Residency gate (spec §7 K0.5/P1): chained ops must not round-trip to CPU.
Native evidence: tfvk_tests prints alloc/transfer counters."""
import subprocess
r = subprocess.run(["tfvk_tests"], capture_output=True, text=True)
print(r.stdout[-2000:])
raise SystemExit(r.returncode)
