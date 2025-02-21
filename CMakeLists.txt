cmake_minimum_required(VERSION 3.10)
project(MinSurProject)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE OR "${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  add_definitions(-DBUILD_TYPE_DEBUG)
endif()

# Add FastFem library
set(BUILD_FASTFEM_TESTS OFF CACHE BOOL "Disable tests for FastFem" FORCE)
add_subdirectory(lib/FastFem)

# Add fastfem minsur 
add_subdirectory(src/fastFEM)

# Check if deal.II is available and conditionally add the dealII minsur subdirectory
find_package(deal.II 9.3.1 
  HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR} $ENV{mkDealiiPrefix})
if(deal.II_FOUND)
    add_subdirectory(src/dealII)
else()
    message(WARNING "deal.II not found, skipping MinSurDealII executable.")
endif()