# Desktop Vulkan CI harness
No CI workflow is configured. Run locally: `cmake -S . -B build &&
cmake --build build -j && ctest --test-dir build` (lavapipe or a real
Vulkan device). Desktop results are debugging evidence only, never
mobile performance claims (§11).
