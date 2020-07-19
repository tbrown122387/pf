cmake_minimum_required(VERSION 3.12)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -g")

file(GLOB SOURCES ${CMAKE_SOURCE_DIR}/test/*.cpp)
add_executable(${PROJECT_NAME}_test ${SOURCES})


target_include_directories(${PROJECT_NAME}_test 
    PUBLIC 
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
target_compile_features(${PROJECT_NAME}_test PUBLIC cxx_std_17)
target_link_libraries(${PROJECT_NAME}_test Eigen3::Eigen Catch2::Catch2 ${Boost_LIBRARIES})


#add_test(NAME PF_resampler_tests COMMAND SI_detail_tests)
#add_test(NAME SI_base_unit_tests COMMAND SI_base_unit_tests)
#add_test(NAME SI_derived_unit_tests COMMAND SI_derived_unit_tests)


#target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_NAME}/../include/pf)
#target_link_libraries(${PROJECT_NAME} pf::pf Eigen3::Eigen ${Boost_LIBRARIES} Catch2::Catch2)

#include(CTest) # not sure if this is 100% necessary
#include(Catch)
#catch_discover_tests(tests)
#enable_testing()
