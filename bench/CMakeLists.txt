#-----------------------------------------------------------------------------------
# bench/CMakeLists.txt
#
# Two executables, and the split is deliberate:
#
#   bench_cpu_ref   always built. It measures the CPU reference implementations, which
#                   are the DENOMINATOR of every "Nx faster" claim in the report. It
#                   needs no GPU, so it runs on the laptop, and the numbers it produces
#                   do not change when you move machines.
#
#   bench_kernels   built only when nvcc was found. Everything it measures lives on the
#                   GPU, so there is nothing for it to do on Machine A.
#
# WHY THESE ARE NOT ctest ENTRIES
#
# A timing measurement is not a pass/fail test. Wiring benchmarks into ctest sounds
# thorough and produces exactly one outcome: a suite that fails intermittently because
# a shared machine was busy, which then gets ignored, which then hides real failures.
# Correctness belongs in tests/; performance belongs in a table you read. The one
# performance property worth asserting -- "this kernel did not regress by 2x" -- needs a
# recorded baseline per machine, and that is what docs/lab-notebook.md is for.
#
# There is a `bench` convenience target at the bottom for running both.
#-----------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------
# cuBLAS: baseline only.
#
# Objective 2 of this project is to hand-write GEMM. cuBLAS exists in this build for
# exactly one purpose -- to answer "how close did my kernel get?" -- and it is linked
# ONLY into bench_kernels, never into the engine library. If cuBLAS ever appears in a
# target other than this one, something has gone wrong with the point of the project.
# See docs/adr/0004-cublas-baseline-only.md.
#
# Guarded on TARGET rather than assumed: cuBLAS ships as a separate component of the
# CUDA toolkit and a minimal or runtime-only install can have cudart without it. That
# is a missing baseline, not a broken build, so it degrades to a warning -- consistent
# with the rest of the build system, where an absent optional dependency never stops
# you from working.
#-----------------------------------------------------------------------------------
option(ENGINE_BENCH_CUBLAS "Benchmark cuBLAS as a matmul baseline" ON)

set(ENGINE_HAVE_CUBLAS OFF)
if(ENGINE_CUDA_ENABLED AND ENGINE_BENCH_CUBLAS)
  if(TARGET CUDA::cublas)
    set(ENGINE_HAVE_CUBLAS ON)
  else()
    message(WARNING
      "ENGINE_BENCH_CUBLAS is ON but the CUDA::cublas target does not exist.\n"
      "  Your CUDA toolkit install has cudart but not cuBLAS -- install the\n"
      "  cuda-libraries-dev package (Linux) or repair the toolkit (Windows).\n"
      "  Building without the cuBLAS baseline: matmul_tiled will have nothing\n"
      "  to be compared against, so you will not know whether it is any good.")
  endif()
endif()

#-----------------------------------------------------------------------------------
# bench_cpu_ref -- the baseline, buildable everywhere
#-----------------------------------------------------------------------------------
add_executable(bench_cpu_ref bench_cpu_ref.cpp)
target_link_libraries(bench_cpu_ref PRIVATE engine)
# So `#include "bench_harness.hpp"` resolves regardless of the working directory the
# compiler was invoked from.
target_include_directories(bench_cpu_ref PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
engine_apply_warnings(bench_cpu_ref)

#-----------------------------------------------------------------------------------
# bench_kernels -- the GPU numbers
#-----------------------------------------------------------------------------------
if(ENGINE_CUDA_ENABLED)
  add_executable(bench_kernels bench_kernels.cu)
  target_link_libraries(bench_kernels PRIVATE engine)
  target_include_directories(bench_kernels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  engine_apply_warnings(bench_kernels)

  set_target_properties(bench_kernels PROPERTIES
    CUDA_ARCHITECTURES "${ENGINE_CUDA_ARCH}")

  # ENGINE_BENCH_CUBLAS is read with #if, so it must always be defined to something --
  # otherwise `#if ENGINE_BENCH_CUBLAS` silently evaluates an undefined identifier as 0
  # and the cuBLAS rows vanish with no indication why. Defining it to 0 explicitly
  # makes the intent visible in compile_commands.json.
  if(ENGINE_HAVE_CUBLAS)
    target_link_libraries(bench_kernels PRIVATE CUDA::cublas)
    target_compile_definitions(bench_kernels PRIVATE ENGINE_BENCH_CUBLAS=1)
  else()
    target_compile_definitions(bench_kernels PRIVATE ENGINE_BENCH_CUBLAS=0)
  endif()

  # A benchmark built against a synchronise-after-every-launch library measures the
  # synchronisation, not the kernel. On short kernels that is an order of magnitude,
  # and the resulting numbers look plausible -- which is precisely why this warns
  # rather than trusting you to remember. See the option in the root CMakeLists.txt.
  if(ENGINE_SYNC_CHECK_KERNELS)
    message(WARNING
      "ENGINE_SYNC_CHECK_KERNELS=ON: every kernel launch in `engine` will call\n"
      "  cudaDeviceSynchronize(). bench_kernels will still build and run, but its\n"
      "  numbers are NOT valid -- do not record them. Configure a separate build\n"
      "  directory without the option for benchmarking.")
  endif()

  # Same argument, one step removed: a Debug build has NDEBUG unset, which turns on the
  # same synchronisation via check.hpp without anyone asking for it. Caught at configure
  # time here as well as at runtime by the binary itself, because a warning you see
  # before waiting five minutes for a benchmark is worth more than one you see after.
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(WARNING
      "CMAKE_BUILD_TYPE=Debug: kernel timings from this build are meaningless\n"
      "  (unoptimised host code, and CUDA_CHECK_KERNEL() synchronises because\n"
      "  NDEBUG is unset). Use -DCMAKE_BUILD_TYPE=RelWithDebInfo, which keeps the\n"
      "  symbols and -lineinfo that Nsight Compute needs.")
  endif()
endif()

#-----------------------------------------------------------------------------------
# Convenience target:  cmake --build build --target bench
#
# USES_TERMINAL so the tables stream out as they are produced instead of being buffered
# until the whole run finishes -- a full bench_kernels run takes minutes, and watching
# it is how you notice a kernel has hung rather than merely being slow.
#-----------------------------------------------------------------------------------
if(ENGINE_CUDA_ENABLED)
  add_custom_target(bench
    COMMAND bench_cpu_ref
    COMMAND bench_kernels
    DEPENDS bench_cpu_ref bench_kernels
    COMMENT "Running CPU baselines, then GPU kernel benchmarks"
    USES_TERMINAL
    VERBATIM)
else()
  add_custom_target(bench
    COMMAND bench_cpu_ref
    DEPENDS bench_cpu_ref
    COMMENT "Running CPU baselines only (no CUDA in this build)"
    USES_TERMINAL
    VERBATIM)
endif()
