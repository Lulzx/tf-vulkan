# Graph capture / lowering (spec §8)

## F7 spike result (task 010, desktop harness, TF 2.21)
- `TF_InitGraph` is implemented in the plugin: TF **does** invoke our
  Grappler optimizer for `tf.function` graphs placed on the plugin
  device (verified: log records invocations, graph executes correctly).
- The optimizer is currently pass-through (byte-identical output) and
  logs serialized-graph size plus heuristic op-name counts. It proves
  the interception route; it does not yet rewrite regions.
- Partitioning rules and the sha256 cache key are implemented and
  unit-tested in `python/tf_vulkan` (`partition_ops`, `cache_key`).

## Still gated
- Rewriting eligible regions into `_VulkanCompiledRegion` + VMFB
  execution needs an IREE runtime with the Vulkan HAL (pip runtime
  lacks it; see src/iree_bridge/spike). Until then
  `tf_vulkan.compile()` raises explicitly and no graph acceleration
  is claimed.
