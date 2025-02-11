# Add sources
file(GLOB_RECURSE SOURCES "*.cpp")

###############
# find OpenMP #
###############
#cmake_policy(SET CMP0074 NEW) # check if this is needed
find_package(OpenMP)

# add sources to the library
add_library(MinSurFastFEM ${SOURCES})
target_include_directories(MinSurFastFEM PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Link OpenMP libraries
if(OpenMP_CXX_FOUND)
    target_link_libraries(MinSurFastFEM PUBLIC OpenMP::OpenMP_CXX)
endif()