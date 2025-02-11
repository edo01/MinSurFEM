add_executable(minsur main.cpp)
target_link_libraries(minsurf_main PRIVATE MinSurFastFEM MinSurDealii)
target_include_directories(minsurf_main PRIVATE ${CMAKE_SOURCE_DIR}/include)