cmake_minimum_required(VERSION 3.10)
project(MyProject)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add include directories
include_directories(include)

# Add the FastFem submodule
set(BUILD_TESTS OFF CACHE BOOL "Disable tests for FastFem submodule")
add_subdirectory(lib/FastFem)

# Add the deal.II implementation
add_subdirectory(src/dealII)

# Add the fastFEM implementation
add_subdirectory(src/fastFEM)

# Add the main executable
add_subdirectory(src/main)