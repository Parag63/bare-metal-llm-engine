cmake_minimum_required(VERSION 3.20)
project(minimal_downstream LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Pure downstream consumption when built standalone: finds installed package.
# When added in-tree via add_subdirectory(examples), links against the alias target.
if(NOT TARGET bare_metal_llm::engine)
  find_package(bare_metal_llm REQUIRED)
endif()

add_executable(minimal minimal.cpp)
target_link_libraries(minimal PRIVATE bare_metal_llm::engine)
