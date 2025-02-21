# Add sources
file(GLOB_RECURSE SOURCES "*.cpp")

# Locate Boost
find_package(Boost 1.72.0 REQUIRED
  COMPONENTS filesystem iostreams serialization
  HINTS ${BOOST_DIR} $ENV{BOOST_DIR} $ENV{mkBoostPrefix})
message(STATUS "Using the Boost-${Boost_VERSION} configuration found at ${Boost_DIR}")
include_directories(${Boost_INCLUDE_DIRS})

deal_ii_initialize_cached_variables()

# Create the executable
add_executable(MinSurDealII ${SOURCES})
target_include_directories(MinSurDealII PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Link the executable with deal.II (plain signature)
deal_ii_setup_target(MinSurDealII)

# Add compiler flags
target_compile_options(MinSurDealII PRIVATE
    -Wfloat-conversion
    -Wmissing-braces
    -Wnon-virtual-dtor
)