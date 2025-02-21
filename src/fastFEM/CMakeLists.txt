cmake_minimum_required(VERSION 3.10)

# Make OpenMP optional
option(USE_OPENMP "Enable OpenMP support" ON)

# Set compiler flags for different build types
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Allow narrowing conversions
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing")

# Enable optional fast math optimizations.
option(USE_FAST_MATH "Enable fast math optimizations." OFF)

if(USE_FAST_MATH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -funsafe-math-optimizations")
    message(STATUS "Enabled fast math optimizations (-Ofast -funsafe-math-optimizations)")
endif()

# Add include directories
include_directories(../../lib/FastFem/include/)

# Collect all source files
file(GLOB_RECURSE LIB_SOURCES "../../lib/FastFem/src/*.cpp")

# Define the shared library
add_library(FastFem SHARED ${LIB_SOURCES})
target_include_directories(FastFem PUBLIC ../../lib/FastFem/include/)

if(USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(FastFem PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(FastFem PUBLIC HAVE_OPENMP)
    else()
        message(WARNING "OpenMP requested but not found. Building without OpenMP support.")
    endif()
endif()

add_executable(MinSurFF MinSurFF.cpp)

target_link_libraries(MinSurFF FastFem)

target_include_directories(MinSurFF PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Enable testing support
include(CTest)
add_test(NAME RunAllTests COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure)