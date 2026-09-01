#-----------------------------------------------------------------------------------
# tests/CMakeLists.txt
#
# ONE executable, not one per file. The harness in test_framework.hpp registers every
# TEST() into a single global registry, so `engine_tests --filter=X` selects a subset
# at runtime. That keeps link times down and, more usefully, means `engine_tests` with
# no arguments is always the complete picture of where the project stands.
#
# ctest then gets one entry PER SUITE (see the loop at the bottom), so a CI log shows
# which area broke without you having to read the output. Both views, one binary.
#-----------------------------------------------------------------------------------

set(ENGINE_TEST_SOURCES
  test_framework.cpp     # the harness itself, plus main()
  golden.cpp             # ENGREF01 reader
  test_dtype.cpp         # dtype table, sizes, INT4 packing arithmetic
  test_cpu_ref.cpp       # the CPU oracle, vs tools/gen_reference.py
  test_tensor.cpp        # Module 1 specification (mostly TEST_PENDING until Oct 2026)
)

# test_kernels.cu is added ONLY when there is an nvcc to compile it. On the laptop
# that leaves 5 files and a suite that still runs and still means something; the GPU
# tests are then simply absent rather than failing. Note this is the *only* place in
# the test build that is conditional -- everything above compiles everywhere, which is
# what makes the CPU-only machine a real development environment rather than a
# degraded one.
if(ENGINE_CUDA_ENABLED)
  list(APPEND ENGINE_TEST_SOURCES test_kernels.cu)
endif()

add_executable(engine_tests ${ENGINE_TEST_SOURCES})

target_link_libraries(engine_tests PRIVATE engine)

# Needed so `#include "golden.hpp"` works from any test file regardless of where the
# compiler was invoked from.
target_include_directories(engine_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

#-----------------------------------------------------------------------------------
# Where the reference data lives.
#
# Baked in as an ABSOLUTE path from the source tree, for one reason: `ctest` runs
# tests with the working directory set to the build tree, so the fallback
# "tests/golden" relative path in golden.cpp would resolve to build/tests/golden and
# find nothing. Rather than have the tests skip mysteriously under ctest but pass when
# run by hand, the path is fixed at configure time.
#
# The ENGINE_GOLDEN_DIR environment variable still overrides it at runtime, which is
# how you point a build at regenerated data without reconfiguring.
#
# The directory does NOT need to exist yet. Reference data is generated output (28 MiB
# of it) and is gitignored, so a fresh clone has no golden/ at all -- the tests skip
# with the generator command until you run:
#
#     python3 tools/gen_reference.py
#-----------------------------------------------------------------------------------
target_compile_definitions(engine_tests PRIVATE
  ENGINE_GOLDEN_DIR_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/golden")

engine_apply_warnings(engine_tests)

if(ENGINE_CUDA_ENABLED)
  set_target_properties(engine_tests PROPERTIES
    CUDA_ARCHITECTURES "${ENGINE_CUDA_ARCH}")

  # NOTE -- and this is a trap worth understanding, because the obvious thing here is
  # wrong. The tempting line is:
  #
  #     target_compile_definitions(engine_tests PRIVATE ENGINE_ALWAYS_SYNC_CHECK)
  #
  # reasoning that a test binary should synchronise after every launch so an in-kernel
  # fault is attributed to the launch that caused it. The intent is right; the mechanism
  # does nothing. CUDA_CHECK_KERNEL() is expanded inside kernels/*.cu, which are
  # compiled into the `engine` library -- by the time engine_tests is compiled, those
  # macros have long since been expanded with whatever definition the library saw. A
  # definition on this target only affects test_kernels.cu itself, which never calls
  # CUDA_CHECK_KERNEL() at all (it goes through the host launchers, by design).
  #
  # The real control is the library-wide ENGINE_SYNC_CHECK_KERNELS option in the root
  # CMakeLists.txt. It is off by default; turn it on in a separate build directory when
  # a kernel test fails somewhere you cannot localise:
  #
  #     cmake -B build-sync -DENGINE_SYNC_CHECK_KERNELS=ON -DCMAKE_BUILD_TYPE=Debug
  #     cmake --build build-sync -j && ctest --test-dir build-sync -R kernels
  #
  # Correctness is not at stake either way: every test downloads its result, download()
  # is a CUDA_CHECK'd cudaMemcpy, and a synchronous memcpy after a faulting kernel
  # reports the fault. Only the reported LINE NUMBER differs -- and only when something
  # is already broken.
  if(NOT ENGINE_SYNC_CHECK_KERNELS)
    message(STATUS "Tests: kernel launches are async-checked only "
                   "(-DENGINE_SYNC_CHECK_KERNELS=ON to localise in-kernel faults)")
  endif()
endif()

#-----------------------------------------------------------------------------------
# ctest registration: one entry per suite.
#
# Keep this list in sync with the suite names used in TEST(suite, name). A typo is
# caught immediately: engine_tests exits 2 when a --filter matches nothing, precisely
# so that a renamed suite cannot leave behind a ctest entry that runs zero tests and
# reports PASS forever.
#
# It deliberately does NOT check for "at least one test passed" -- a suite that is
# entirely TEST_PENDING (tensor.*, until Module 1 is written) passes nothing and that
# is the correct state, not a failure.
#-----------------------------------------------------------------------------------
set(ENGINE_TEST_SUITES
  dtype        # DType metadata and quantisation arithmetic
  golden       # is the reference data itself trustworthy?
  cpu_ref      # the CPU oracle, vs PyTorch/NumPy
  storage      # Storage allocation and device rejection
  tensor       # Module 1: shapes, strides, views  (pending until implemented)
)

if(ENGINE_CUDA_ENABLED)
  list(APPEND ENGINE_TEST_SUITES kernels)
endif()

foreach(_suite IN LISTS ENGINE_TEST_SUITES)
  add_test(NAME ${_suite}
           COMMAND engine_tests --filter=${_suite}.)
  # Generous timeout: the 512x512x512 matmul and the 8x50257 softmax are real work,
  # and a first-attempt kernel with a bad launch configuration can be very slow
  # without being wrong. Long enough to avoid false failures, short enough to catch a
  # kernel that has genuinely hung.
  set_tests_properties(${_suite} PROPERTIES TIMEOUT 300)
endforeach()

# And one entry that runs EVERYTHING, including any suite missing from the list above.
# This is the one CI should look at; the per-suite entries are for readable output.
add_test(NAME all COMMAND engine_tests)
set_tests_properties(all PROPERTIES TIMEOUT 900)

#-----------------------------------------------------------------------------------
# Convenience target: regenerate the golden files.
#
#     cmake --build build --target reference_data
#
# Not a dependency of engine_tests, deliberately. Generating 28 MiB of reference data
# is not something that should happen implicitly as part of a build, and the tests
# already skip cleanly when it is absent. It is also the one step that needs Python
# and PyTorch, which the C++ build has no business requiring.
#-----------------------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_FOUND)
  add_custom_target(reference_data
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/tools/gen_reference.py
            --out-dir ${CMAKE_CURRENT_SOURCE_DIR}/golden
    COMMENT "Generating reference data with tools/gen_reference.py"
    VERBATIM)
else()
  message(STATUS "Python3 not found; the `reference_data` target is unavailable.")
  message(STATUS "  Run tools/gen_reference.py by hand before expecting the")
  message(STATUS "  cpu_ref and kernels suites to do anything but skip.")
endif()
