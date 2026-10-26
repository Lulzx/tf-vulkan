# Troubleshooting

- **Tier C / no device**: check loader + ICD (`VK_ICD_FILENAMES` on macOS
  MoltenVK), compute queue presence; run `tfvk_probe`.
- **Plugin does not load (Android)**: Bionic namespace / `RTLD_LOCAL`
  hiding `TF_*` symbols — test `RTLD_GLOBAL` before `import tensorflow`
  or link the exact `libtensorflow_framework.so` SONAME (spec §5).
- **Device listed but op ran on CPU**: soft placement. Use strict mode
  (`TF_VULKAN_MODE=strict`, `tf.config.set_soft_device_placement(False)`)
  and check fallback counters; the device string alone is not proof (§15).
- **Stale reads after dtoh**: deferred batch was not flushed — dtoh on a
  pending buffer must flush+wait (§6.2); report with stream flush counters.
- **GPU watchdog / device lost**: submission too large; lower
  `StreamConfig.max_workgroups`, chunk GEMMs (§12).
- **Pipeline cache ignored**: driver UUID/vendor mismatch discards it (§12).
