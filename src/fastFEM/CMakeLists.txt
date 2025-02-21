# Add sources
file(GLOB_RECURSE SOURCES "*.cpp")

# Create the fastFEM executable
add_executable(MinSurFastFEM ${SOURCES})
target_include_directories(MinSurFastFEM PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Link against FastFem (keyword-based signature)
target_link_libraries(MinSurFastFEM PUBLIC FastFem)

# Add compiler flags
target_compile_options(MinSurFastFEM PRIVATE
    -Wfloat-conversion
    -Wmissing-braces
    -Wnon-virtual-dtor
)