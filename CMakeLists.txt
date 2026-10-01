#-----------------------------------------------------------------------------------
# Bare-Metal LLM Inference Engine
#
# DESIGN NOTE (read this before editing):
#   This project is developed across TWO machines:
#     Machine A -- the laptop. No NVIDIA GPU. Where the C++ is written, where the
#                  CPU reference implementations and the tensor library are tested,
#                  and where CI runs.
#     Machine B -- a Windows box with an RTX 4090 (sm_89), exclusive access. Where
#                  every CUDA kernel is compiled, verified and benchmarked.
#   Git is the transport between them; see docs/01-dev-environment.md.
#
#   Therefore CUDA is *optional*. The whole project must configure, build, and pass
#   its CPU test suite with no NVIDIA toolchain present -- otherwise Machine A stops
#   being usable for development the moment a kernel is added. CUDA sources are added
#   only when nvcc is found, and everything that touches the GPU is #if-guarded.
#
#   See docs/adr/0001-cuda-optional-build.md for the reasoning.
#-----------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.20)

project(bare_metal_llm_engine
        VERSION 0.1.0
        DESCRIPTION "A llama.cpp-style LLM inference engine written from scratch in C++/CUDA"
        LANGUAGES CXX)

#-----------------------------------------------------------------------------------
# Global C++ settings
#-----------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Generates compile_commands.json so clangd / the VS Code C++ extension gives you
# correct autocomplete and error squiggles. Symlink it into the repo root if needed.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Put all binaries in one predictable place: build/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  # RelWithDebInfo is the right default for this project: we care about performance
  # numbers (so -O2) but we also need symbols and -lineinfo for Nsight profiling.
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
  message(STATUS "CMAKE_BUILD_TYPE not set; defaulting to RelWithDebInfo")
endif()

#-----------------------------------------------------------------------------------
# Options
#-----------------------------------------------------------------------------------
option(ENGINE_BUILD_TESTS    "Build the unit test suite"               ON)
option(ENGINE_BUILD_BENCH    "Build the benchmark executables"         ON)
option(ENGINE_BUILD_EXAMPLES "Build downstream integration examples"   ON)
option(ENGINE_WITH_CUDA      "Build CUDA kernels if nvcc is available" ON)
option(ENGINE_WERROR         "Treat compiler warnings as errors"       OFF)

# Force CUDA_CHECK_KERNEL() to synchronise even in optimised builds.
#
# WHY THIS IS A LIBRARY-WIDE OPTION AND NOT A PER-TARGET ONE -- this is a subtle point
# and getting it wrong wastes a debugging session:
#
#   CUDA_CHECK_KERNEL() is expanded inside kernels/*.cu, which are compiled into the
#   `engine` library. Whether it synchronises is therefore decided when THE LIBRARY is
#   compiled. Defining ENGINE_ALWAYS_SYNC_CHECK on the test or benchmark executable has
#   no effect whatsoever on the launchers those executables call -- the macro has
#   already been expanded. So it has to be PUBLIC on `engine`, which is what this
#   option does.
#
# Default OFF, because a synchronise after every launch makes every benchmark number
# in the project meaningless, and a plausible-looking wrong number is the exact failure
# mode bench/bench_harness.hpp exists to prevent.
#
# Turn it ON when a kernel test fails in a way you cannot localise:
#
#     cmake -B build-sync -DENGINE_SYNC_CHECK_KERNELS=ON -DCMAKE_BUILD_TYPE=Debug
#
# Without it, an in-kernel fault still gets caught -- the next download() is a
# CUDA_CHECK'd cudaMemcpy -- but it is reported against the memcpy rather than the
# launch that actually caused it. With it, the error names the launch.
option(ENGINE_SYNC_CHECK_KERNELS
       "Synchronise and check after every kernel launch, even in release builds" OFF)

# 89 = GeForce RTX 4090 / RTX 4070 SUPER (Ada Lovelace).
# On CMake >= 3.24 you may instead set this to "native" to auto-detect the GPU actually present.
set(ENGINE_CUDA_ARCH "89" CACHE STRING "CUDA architecture to target (e.g. 89 for Ada Lovelace)")

#-----------------------------------------------------------------------------------
# CUDA detection (never fatal)
#-----------------------------------------------------------------------------------
include(CheckLanguage)
include(cmake/EngineCuda.cmake)
engine_setup_cuda()   # sets ENGINE_CUDA_ENABLED, may enable_language(CUDA)

#-----------------------------------------------------------------------------------
# Git commit hash for provenance reporting
#-----------------------------------------------------------------------------------
execute_process(
  COMMAND git rev-parse --short HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE ENGINE_GIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(NOT ENGINE_GIT_HASH)
  set(ENGINE_GIT_HASH "unknown")
endif()

#-----------------------------------------------------------------------------------
# Generated configuration header
#   Lets C++ code do `#if ENGINE_HAS_CUDA` instead of guessing.
#-----------------------------------------------------------------------------------
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/engine_config.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/generated/engine/config.hpp
  @ONLY)

#-----------------------------------------------------------------------------------
# The engine library
#-----------------------------------------------------------------------------------
# Always compiled, with or without a GPU. cuda_device.cpp is in this list on
# purpose: it is #if-guarded internally and returns "no CUDA device" on CPU-only
# builds, so callers (especially benchmarks) never need their own guards.
set(ENGINE_SOURCES
  src/dtype.cpp
  src/tensor.cpp
  src/allocator.cpp
  src/cuda_device.cpp
  src/cpu_ref/vector_add_cpu.cpp
  src/cpu_ref/reduce_cpu.cpp
  src/cpu_ref/softmax_cpu.cpp
  src/cpu_ref/rmsnorm_cpu.cpp
  src/cpu_ref/matmul_cpu.cpp
  src/cpu_ref/residual_rmsnorm_cpu.cpp
  src/cpu_ref/rmsnorm_linear_cpu.cpp
  src/cpu_ref/gemv_cpu.cpp
  src/cpu_ref/embedding_cpu.cpp
  src/cpu_ref/argmax_cpu.cpp
  src/cpu_ref/swiglu_cpu.cpp
  src/pool_allocator.cpp
  src/cuda_stream.cpp
)

set(ENGINE_CUDA_SOURCES
  kernels/vector_add.cu
  kernels/reduce_sum.cu
  kernels/softmax.cu
  kernels/rmsnorm.cu
  kernels/matmul_naive.cu
  kernels/matmul_tiled.cu
  kernels/matmul_register_tiled.cu
  kernels/gemv.cu
  kernels/gemv_fp16.cu
  kernels/embedding.cu
  kernels/argmax.cu
  kernels/swiglu.cu
  kernels/residual_rmsnorm.cu
  kernels/rmsnorm_linear.cu
)

if(ENGINE_CUDA_ENABLED)
  list(APPEND ENGINE_SOURCES ${ENGINE_CUDA_SOURCES})
endif()

add_library(engine STATIC ${ENGINE_SOURCES})
add_library(bare_metal_llm::engine ALIAS engine)

include(GNUInstallDirs)

target_include_directories(engine
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if(ENGINE_CUDA_ENABLED)
  # cudart is needed by the *C++* translation units too (allocator.cpp calls
  # cudaMalloc), not just the .cu files, so link it on the whole target.
  target_link_libraries(engine PUBLIC CUDA::cudart)
  set_target_properties(engine PROPERTIES
    CUDA_ARCHITECTURES "${ENGINE_CUDA_ARCH}"
    CUDA_SEPARABLE_COMPILATION OFF)

  # PUBLIC, not PRIVATE: see the option's comment above. It must reach both the
  # launchers in kernels/*.cu and any caller that expands the macro itself, and the
  # two must agree or the behaviour depends on which header saw which definition.
  if(ENGINE_SYNC_CHECK_KERNELS)
    target_compile_definitions(engine PUBLIC ENGINE_ALWAYS_SYNC_CHECK)
  endif()
endif()

# Warnings. Kept as a helper because tests/benches want the same set.
include(cmake/EngineWarnings.cmake)
engine_apply_warnings(engine)

#-----------------------------------------------------------------------------------
# Installation and CMake package export
#
# Allows downstream consumers to link with:
#   find_package(bare_metal_llm REQUIRED)
#   target_link_libraries(my_app PRIVATE bare_metal_llm::engine)
#-----------------------------------------------------------------------------------
include(CMakePackageConfigHelpers)

install(TARGETS engine
  EXPORT bare_metal_llm_targets
  ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT bare_metal_llm_targets
  FILE bare_metal_llm_targets.cmake
  NAMESPACE bare_metal_llm::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/bare_metal_llm
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/bare_metal_llm-config-version.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bare_metal_llm-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/bare_metal_llm-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/bare_metal_llm
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/bare_metal_llm-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/bare_metal_llm-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/bare_metal_llm
)

#-----------------------------------------------------------------------------------
# Tests and benchmarks
#-----------------------------------------------------------------------------------
if(ENGINE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(ENGINE_BUILD_BENCH)
  add_subdirectory(bench)
endif()

if(ENGINE_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

#-----------------------------------------------------------------------------------
# Configuration summary — printed on every configure so you always know what you got
#-----------------------------------------------------------------------------------
message(STATUS "")
message(STATUS "==================== bare-metal-llm-engine ====================")
message(STATUS "  Version          : ${PROJECT_VERSION}")
message(STATUS "  Build type       : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ compiler     : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
if(ENGINE_CUDA_ENABLED)
  message(STATUS "  CUDA             : ENABLED (nvcc ${CMAKE_CUDA_COMPILER_VERSION})")
  message(STATUS "  CUDA arch        : ${ENGINE_CUDA_ARCH}")
  message(STATUS "  Sync-check       : ${ENGINE_SYNC_CHECK_KERNELS}  (ON = debuggable, benchmarks invalid)")
  message(STATUS "  GPU kernels      : ${ENGINE_CUDA_SOURCES}")
else()
  message(STATUS "  CUDA             : DISABLED  <-- CPU reference path only")
  message(STATUS "                     (expected on the laptop; push and build on")
  message(STATUS "                      the 4090 box to compile the kernels)")
endif()
message(STATUS "  Tests            : ${ENGINE_BUILD_TESTS}")
message(STATUS "  Benchmarks       : ${ENGINE_BUILD_BENCH}")
message(STATUS "===============================================================")
message(STATUS "")
