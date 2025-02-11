cmake_minimum_required(VERSION 3.12.0)
project(MINSURFEM LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED "ON")

# Set default build type to Release.
if(NOT CMAKE_BUILD_TYPE OR "${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
message(STATUS)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS)
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  add_definitions(-DBUILD_TYPE_DEBUG)
endif()

##################
# COMPILER FLAGS #
##################
# Add useful compiler flags.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfloat-conversion -Wmissing-braces -Wnon-virtual-dtor")

# Set compiler flags for different build types
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Enable optional fast math optimizations.
option(USE_FAST_MATH "Enable fast math optimizations." OFF)

if(USE_FAST_MATH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -funsafe-math-optimizations")
    message(STATUS "Enabled fast math optimizations (-Ofast -funsafe-math-optimizations)")
endif()


# Locate MPI compiler.
find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILER "${MPI_CXX_COMPILER}")

##########################
# Include subdirectories #
##########################
#add_subdirectory(src/fastFEM)
add_subdirectory(src/dealII)
add_subdirectory(src/main)