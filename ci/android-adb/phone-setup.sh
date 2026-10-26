#!/data/data/com.termux/files/usr/bin/bash
# tf-vulkan on-phone setup (spec §4 F0/F5 path via Termux + Ubuntu proot).
# Run in Termux:  bash /sdcard/tf-vulkan-phone-setup.sh
# Log lands in /sdcard/tf-vulkan-phone.log (readable via adb).
set -x
LOG=/sdcard/tf-vulkan-phone.log
exec > >(tee "$LOG") 2>&1
echo "=== tf-vulkan phone setup: $(date) ==="

pkg install -y proot-distro || exit 1
proot-distro install ubuntu || true

cat > /root/tf-vulkan-ubuntu.sh <<'EOS'
set -x
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3 python3-venv cmake g++ git libvulkan-dev \
  glslang-tools mesa-vulkan-drivers vulkan-tools file || exit 1

python3 -m venv /root/tfenv
. /root/tfenv/bin/activate
pip install -U pip
pip install tensorflow numpy || exit 1

echo "=== F0: TensorFlow import on phone (Ubuntu proot) ==="
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices())"

echo "=== Vulkan via Mesa panvk (Panthor DRM) ==="
export PAN_I_WANT_A_BROKEN_VULKAN_DRIVER=1
ls /usr/share/vulkan/icd.d/
vulkaninfo --summary 2>&1 | head -30

echo "=== Build tf-vulkan (native runtime + TF plugin) ==="
cd /root
git clone https://github.com/Lulzx/tf-vulkan.git || (cd tf-vulkan && git pull)
cd tf-vulkan
TFINC=$(python -c "import tensorflow,os;print(os.path.join(os.path.dirname(tensorflow.__file__),'include'))")
cmake -S . -B build-phone -DTFVK_BUILD_TF_PLUGIN=ON -DTFVK_TF_INCLUDE_DIR=$TFINC || exit 1
cmake --build build-phone -j || exit 1

echo "=== Probe (panvk) ==="
VK_ICD_FILENAMES=$(ls /usr/share/vulkan/icd.d/*panvk*.json | head -1) ./build-phone/tfvk_probe || true

echo "=== F5: strict MatMul through TF on phone GPU (attempt) ==="
TFVK_GRAPH_LOG=/sdcard/tf-graph.log \
VK_ICD_FILENAMES=$(ls /usr/share/vulkan/icd.d/*panvk*.json | head -1) \
TF_VULKAN_PLUGIN=/root/tf-vulkan/build-phone/libtf_vulkan.so \
python tests/integration/test_tf_plugin.py || true
echo "=== DONE ==="
EOS

proot-distro login ubuntu -- bash /root/tf-vulkan-ubuntu.sh
echo "=== setup finished: $(date) ==="
