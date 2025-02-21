# Add sources
file(GLOB_RECURSE SOURCES "*.cpp")

# Create the main executable
add_executable(main ${SOURCES})

# Link against the libraries
target_link_libraries(main PRIVATE MinSurDealII MinSurFastFEM)