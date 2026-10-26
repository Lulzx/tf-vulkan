cmake_minimum_required(VERSION 3.20)
project(tf-vulkan VERSION 0.1.0 LANGUAGES CXX)
# TF Vulkan (tf-vulkan) — spec rev 1.4. Android-first; desktop Vulkan is a
# development/CI harness only (§1, §11). See docs/architecture.md.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(TFVK_BUILD_TF_PLUGIN "Build TF PluggableDevice plugin (requires pinned TF C headers)" OFF)
set(TFVK_TF_INCLUDE_DIR "" CACHE PATH "Path to TensorFlow source/include dir for TF C API headers")

find_package(Vulkan QUIET)
find_package(Python3 COMPONENTS Interpreter QUIET)
find_program(GLSLANG glslangValidator)
set(TFVK_HAS_VULKAN OFF)
if(Vulkan_FOUND AND GLSLANG AND Python3_FOUND)
  set(TFVK_HAS_VULKAN ON)
endif()

add_library(tfvk_runtime STATIC
  src/runtime/device.cpp src/runtime/allocator.cpp src/runtime/stream.cpp
  src/kernels/cpu_reference.cpp src/kernels/vulkan_kernels.cpp)
target_include_directories(tfvk_runtime PUBLIC src)

set(SHADER_NAMES matmul matmul_tiled binary unary bias_add reduce softmax grad conv2d pool depthwise conv_filter_grad conv_input_grad matmul_f16 unary_f16 binary_f16 bias_add_f16 cast_to_f16 cast_from_f16)
if(TFVK_HAS_VULKAN)
  target_compile_definitions(tfvk_runtime PUBLIC TFVK_HAS_VULKAN=1)
  target_link_libraries(tfvk_runtime PUBLIC Vulkan::Vulkan)
  # Offline SPIR-V (spec §7: ship binaries, never a runtime shader compiler).
  set(GEN_DIR ${CMAKE_BINARY_DIR}/generated)
  file(MAKE_DIRECTORY ${GEN_DIR})
  set(GEN_HEADERS "")
  foreach(name IN LISTS SHADER_NAMES)
    string(REPLACE "_" ";" parts ${name})
    set(sym "")
    foreach(p IN LISTS parts)
      string(SUBSTRING ${p} 0 1 first)
      string(TOUPPER ${first} first)
      string(SUBSTRING ${p} 1 -1 rest)
      set(sym "${sym}${first}${rest}")
    endforeach()
    if(name MATCHES "_f16$")
      set(SPV_ENV vulkan1.3)
    else()
      set(SPV_ENV vulkan1.1)
    endif()
    add_custom_command(
      OUTPUT ${GEN_DIR}/${name}_spv.h
      COMMAND ${GLSLANG} -V --target-env ${SPV_ENV} ${CMAKE_SOURCE_DIR}/src/shaders/${name}.comp
              -o ${GEN_DIR}/${name}.spv
      COMMAND Python3::Interpreter ${CMAKE_SOURCE_DIR}/tools/embed_spv.py
              ${GEN_DIR}/${name}.spv ${GEN_DIR}/${name}_spv.h k${sym}Spv
      DEPENDS ${CMAKE_SOURCE_DIR}/src/shaders/${name}.comp ${CMAKE_SOURCE_DIR}/tools/embed_spv.py)
    list(APPEND GEN_HEADERS ${GEN_DIR}/${name}_spv.h)
  endforeach()
  add_custom_command(
    OUTPUT ${GEN_DIR}/shaders_spv.h
    COMMAND ${CMAKE_COMMAND} -E cat ${GEN_HEADERS} > ${GEN_DIR}/shaders_spv.h
    DEPENDS ${GEN_HEADERS})
  add_custom_target(tfvk_shaders DEPENDS ${GEN_DIR}/shaders_spv.h)
  add_dependencies(tfvk_runtime tfvk_shaders)
  target_include_directories(tfvk_runtime PRIVATE ${GEN_DIR})
endif()

add_executable(tfvk_probe src/diagnostics/probe_main.cpp)
target_link_libraries(tfvk_probe PRIVATE tfvk_runtime)

add_executable(tfvk_bench benchmarks/bench_matmul.cpp)
target_link_libraries(tfvk_bench PRIVATE tfvk_runtime)

enable_testing()
add_executable(tfvk_tests tests/conformance/cpp_tests.cpp)
target_link_libraries(tfvk_tests PRIVATE tfvk_runtime)
add_test(NAME tfvk_tests COMMAND tfvk_tests)

add_executable(tfvk_fuzz_shapes tests/fuzz/shape_fuzz.cpp)
target_link_libraries(tfvk_fuzz_shapes PRIVATE tfvk_runtime)
add_test(NAME tfvk_fuzz_shapes COMMAND tfvk_fuzz_shapes)

# TensorFlow PluggableDevice plugin (gated on real TF headers, spec §5).
if(TFVK_BUILD_TF_PLUGIN)
  if(NOT TFVK_TF_INCLUDE_DIR)
    message(FATAL_ERROR "TFVK_BUILD_TF_PLUGIN needs TFVK_TF_INCLUDE_DIR (pinned TF headers, spec §5)")
  endif()
  add_library(tf_vulkan_plugin SHARED src/tensorflow_plugin/plugin.cpp)
  target_include_directories(tf_vulkan_plugin PRIVATE ${TFVK_TF_INCLUDE_DIR})
  target_link_libraries(tf_vulkan_plugin PRIVATE tfvk_runtime)
  if(APPLE)
    # TF_* symbols resolve from the host process at load time (§5 F2).
    target_link_options(tf_vulkan_plugin PRIVATE -undefined dynamic_lookup)
  endif()
  set_target_properties(tf_vulkan_plugin PROPERTIES OUTPUT_NAME tf_vulkan)
endif()

message(STATUS "tf-vulkan: Vulkan harness = ${TFVK_HAS_VULKAN} (glslang=${GLSLANG})")
message(STATUS "tf-vulkan: TF plugin = ${TFVK_BUILD_TF_PLUGIN} (F0/F1 gates must pass first, spec §4)")
