"""Target user experience (spec §2). Runs only when the plugin registers GPU/VULKAN."""
import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))
tf.config.set_soft_device_placement(False)
with tf.device("/GPU:0"):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    print(tf.matmul(a, a).numpy())
