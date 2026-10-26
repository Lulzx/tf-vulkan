#!/bin/bash
# Runs INSIDE the Ubuntu proot (launched from adb shell, see adb-proot.sh).
set -x
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export DEBIAN_FRONTEND=noninteractive
export HOME=/root
LOG=/sdcard/tf-vulkan-adb.log
exec > >(tee "$LOG") 2>&1
echo "=== tf-vulkan adb proot run: $(date) ==="
id; uname -a
# resolv.conf is prepared by the host-side launcher (DNS-over-TCP to 8.8.8.8)
APT="apt-get -o Acquire::ForceIPv4=true"

echo "=== DRM open test ==="
python3 -c "import os; fd=os.open('/dev/dri/renderD128', os.O_RDWR); print('renderD128 OPEN OK', fd)"

dpkg --configure -a || true
$APT update
$APT install -y python3-venv cmake g++ git libvulkan-dev glslang-tools \
  mesa-vulkan-drivers vulkan-tools || exit 1

echo "=== Vulkan via panvk ==="
export PAN_I_WANT_A_BROKEN_VULKAN_DRIVER=1
PANVK_ICD=$(ls /usr/share/vulkan/icd.d/*panfrost*.json | head -1)
echo "ICD: $PANVK_ICD"
VK_ICD_FILENAMES=$PANVK_ICD vulkaninfo --summary 2>&1 | head -40

python3 -m venv /root/tfenv
. /root/tfenv/bin/activate
pip install -U pip
pip install tensorflow numpy || exit 1
echo "=== F0 ==="
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices())"

cd /root
git clone https://github.com/Lulzx/tf-vulkan.git || (cd tf-vulkan && git pull)
cd tf-vulkan
TFINC=$(python -c "import tensorflow,os;print(os.path.join(os.path.dirname(tensorflow.__file__),'include'))")
cmake -S . -B build-phone -DTFVK_BUILD_TF_PLUGIN=ON -DTFVK_TF_INCLUDE_DIR=$TFINC || exit 1
cmake --build build-phone -j || exit 1

echo "=== probe ==="
VK_ICD_FILENAMES=$PANVK_ICD ./build-phone/tfvk_probe || true
echo "=== native tests ==="
VK_ICD_FILENAMES=$PANVK_ICD timeout 600 ./build-phone/tfvk_tests || true
echo "=== F5 integration (attempt) ==="
TF_VULKAN_PLUGIN=/root/tf-vulkan/build-phone/libtf_vulkan.so \
VK_ICD_FILENAMES=$PANVK_ICD \
python tests/integration/test_tf_plugin.py || true
echo "=== DONE ==="
