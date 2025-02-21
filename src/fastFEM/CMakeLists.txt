# Add sources
file(GLOB_RECURSE SOURCES "*.cpp")

# Create the fastFEM library
add_library(MinSurFastFEM ${SOURCES})
target_include_directories(MinSurFastFEM PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Link against FastFem
target_link_libraries(MinSurFastFEM PUBLIC FastFem)