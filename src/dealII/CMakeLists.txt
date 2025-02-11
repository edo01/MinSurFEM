# Add sources
file(GLOB_RECURSE SOURCES "*.cpp")

# Locate Boost.
find_package(Boost 1.72.0 REQUIRED
  COMPONENTS filesystem iostreams serialization
  HINTS ${BOOST_DIR} $ENV{BOOST_DIR} $ENV{mkBoostPrefix})
message(STATUS "Using the Boost-${Boost_VERSION} configuration found at ${Boost_DIR}")
message(STATUS)
include_directories(${Boost_INCLUDE_DIRS})

# Locate deal.II and initialize its variables.
find_package(deal.II 9.3.1 REQUIRED
  HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR} $ENV{mkDealiiPrefix})
deal_ii_initialize_cached_variables()


# add sources to the library
add_library(MinSurDealII ${SOURCES})
target_include_directories(MinSurDealII PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Link the library with deal.II
deal_ii_setup_target(MinSurDealII)