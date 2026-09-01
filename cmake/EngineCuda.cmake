# EngineCuda.cmake
#
# Detects CUDA without ever failing the configure step. This is the single most
# important piece of build logic in the project: the laptop where most of the code is
# written has no NVIDIA toolchain, so a hard `find_package(CUDA REQUIRED)` would make
# the project unbuildable exactly where most of the work happens. The GPU lives on a
# second machine (RTX 4090) that this same tree is cloned onto.

function(engine_setup_cuda)
  set(ENGINE_CUDA_ENABLED OFF PARENT_SCOPE)

  if(NOT ENGINE_WITH_CUDA)
    message(STATUS "CUDA: explicitly disabled via -DENGINE_WITH_CUDA=OFF")
    return()
  endif()

  # check_language() probes for a working nvcc without aborting if it is missing.
  check_language(CUDA)

  if(NOT CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA: no nvcc found -> building CPU reference path only.")
    message(STATUS "      This is normal on the laptop: the tensor library, the CPU")
    message(STATUS "      reference implementations and their tests all build here.")
    message(STATUS "      To compile and verify the kernels: commit, push, and run")
    message(STATUS "      scripts/gpu-run.sh on the RTX 4090 machine.")
    return()
  endif()

  enable_language(CUDA)

  set(CMAKE_CUDA_STANDARD 17          PARENT_SCOPE)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON PARENT_SCOPE)
  set(CMAKE_CUDA_ARCHITECTURES "${ENGINE_CUDA_ARCH}" PARENT_SCOPE)

  # CUDAToolkit gives us the imported targets CUDA::cudart, CUDA::cublas, etc.
  # NOTE: we deliberately do NOT link cuBLAS into the engine. Objective 2 of the
  # project is to hand-write GEMM; cuBLAS is only ever used as a *benchmark
  # baseline* in bench/, never as an implementation. See ADR 0004.
  find_package(CUDAToolkit REQUIRED)

  # -lineinfo maps PTX/SASS back to source lines, which is what makes Nsight Compute
  # reports readable. Cheap, and does not affect optimisation.
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

  # Forward host warnings through nvcc to the host compiler.
  #
  # This MUST branch on the host compiler, and the reason is a trap. nvcc does not
  # compile host code itself -- it hands it to cl.exe on Windows and g++/clang++
  # elsewhere. `-Xcompiler=-Wall` therefore means two completely different things:
  #
  #   g++/clang++ : -Wall = the usual sensible warning set.
  #   MSVC        : cl.exe accepts -Wall as a synonym for /Wall, which is NOT the
  #                 equivalent of gcc's -Wall. It enables *every* warning MSVC has,
  #                 including thousands from the Windows SDK and CRT headers, and the
  #                 real warnings about your kernels drown in the noise. /W4 is the
  #                 MSVC flag that corresponds to what -Wall means everywhere else.
  #
  # Since the RTX 4090 machine is a Windows box, this branch is the one that runs when
  # the kernels are actually compiled -- getting it wrong would make nvcc warnings
  # useless exactly where they matter most. (engine_apply_warnings() in
  # EngineWarnings.cmake makes the same distinction for the C++ targets; this is the
  # CUDA-language counterpart, which that function deliberately does not touch.)
  if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)
  else()
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>)
  endif()

  message(STATUS "CUDA: found nvcc ${CMAKE_CUDA_COMPILER_VERSION} at ${CMAKE_CUDA_COMPILER}")
  message(STATUS "CUDA: targeting architecture(s) ${ENGINE_CUDA_ARCH}")

  # Warn loudly if the requested arch is not one this nvcc can emit. Two ways this
  # bites: an nvcc too OLD to know about Ada (sm_89 needs CUDA >= 11.8), or a future
  # toolkit that has retired an older arch you fell back to.
  execute_process(COMMAND ${CMAKE_CUDA_COMPILER} --list-gpu-arch
                  OUTPUT_VARIABLE _archs ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(_archs)
    foreach(_a IN LISTS ENGINE_CUDA_ARCH)
      if(NOT _archs MATCHES "compute_${_a}")
        message(WARNING
          "Requested CUDA arch ${_a} is NOT supported by this nvcc.\n"
          "  nvcc can emit: ${_archs}\n"
          "  sm_89 (RTX 4090) requires CUDA 11.8 or newer -- check `nvcc --version`.\n"
          "  Either upgrade the toolkit or pass -DENGINE_CUDA_ARCH=<supported arch>.")
      endif()
    endforeach()
  endif()

  set(ENGINE_CUDA_ENABLED ON PARENT_SCOPE)
endfunction()
