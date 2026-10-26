import json
import sys

from . import status

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        print(json.dumps(status(), indent=2))
    else:
        print("usage: python -m tf_vulkan status | python -m tf_vulkan.probe_tf")
