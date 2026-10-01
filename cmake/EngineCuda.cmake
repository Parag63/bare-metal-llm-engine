# EngineCuda.cmake
#
# Detects CUDA without ever failing the configure step. This is the single most
# important piece of build logic in the project: the laptop where most of the code is
# written has no NVIDIA toolchain, so a hard `find_package(CUDA REQUIRED)` would make
# the project unbuildable exactly where most of the work happens. The GPU lives on
# Machine B (RTX 4070 SUPER) that this same tree is cloned onto.

macro(engine_setup_cuda)
  set(ENGINE_CUDA_ENABLED OFF)

  if(NOT ENGINE_WITH_CUDA)
    message(STATUS "CUDA: explicitly disabled via -DENGINE_WITH_CUDA=OFF")
  else()
    # check_language() probes for a working nvcc without aborting if it is missing.
    check_language(CUDA)

    if(NOT CMAKE_CUDA_COMPILER)
      message(STATUS "CUDA: no nvcc found -> building CPU reference path only.")
      message(STATUS "      This is normal on the laptop: the tensor library, the CPU")
      message(STATUS "      reference implementations and their tests all build here.")
      message(STATUS "      To compile and verify the kernels: commit, push, and run")
      message(STATUS "      scripts/gpu-run.sh on the GPU machine (RTX 4070 SUPER).")
    else()
      enable_language(CUDA)

      set(CMAKE_CUDA_STANDARD 17)
      set(CMAKE_CUDA_STANDARD_REQUIRED ON)
      set(CMAKE_CUDA_ARCHITECTURES "${ENGINE_CUDA_ARCH}")

      # CUDAToolkit gives us the imported targets CUDA::cudart, CUDA::cublas, etc.
      # NOTE: we deliberately do NOT link cuBLAS into the engine. Objective 2 of the
      # project is to hand-write GEMM; cuBLAS is only ever used as a *benchmark
      # baseline* in bench/, never as an implementation. See ADR 0004.
      find_package(CUDAToolkit REQUIRED)

      # -lineinfo maps PTX/SASS back to source lines, which is what makes Nsight Compute
      # reports readable. Cheap, and does not affect optimisation.
      add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

      # Forward host warnings through nvcc to the host compiler.
      if(MSVC)
        add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)
      else()
        add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>)
      endif()

      message(STATUS "CUDA: found nvcc ${CMAKE_CUDA_COMPILER_VERSION} at ${CMAKE_CUDA_COMPILER}")
      message(STATUS "CUDA: targeting architecture(s) ${ENGINE_CUDA_ARCH}")

      # Warn loudly if the requested arch is not one this nvcc can emit.
      execute_process(COMMAND ${CMAKE_CUDA_COMPILER} --list-gpu-arch
                      OUTPUT_VARIABLE _archs ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
      if(_archs)
        foreach(_a IN LISTS ENGINE_CUDA_ARCH)
          if(NOT _archs MATCHES "compute_${_a}")
            message(WARNING
              "Requested CUDA arch ${_a} is NOT supported by this nvcc.\n"
              "  nvcc can emit: ${_archs}\n"
              "  sm_89 (RTX 4070 SUPER) requires CUDA 11.8 or newer -- check `nvcc --version`.\n"
              "  Either upgrade the toolkit or pass -DENGINE_CUDA_ARCH=<supported arch>.")
          endif()
        endforeach()
      endif()

      set(ENGINE_CUDA_ENABLED ON)
    endif()
  endif()
endmacro()
