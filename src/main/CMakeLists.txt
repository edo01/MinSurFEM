add_executable(minsur main.cpp)
target_link_libraries(minsur PRIVATE MinSurDealII) #MinSurFastFEM
target_include_directories(minsur PRIVATE ${CMAKE_SOURCE_DIR}/include)