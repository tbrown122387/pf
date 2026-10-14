cmake_minimum_required(VERSION 3.12)

file(GLOB SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/test/*.cpp)
add_executable(${PROJECT_NAME}_test ${SOURCES})


target_include_directories(${PROJECT_NAME}_test
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
target_compile_features(${PROJECT_NAME}_test PUBLIC cxx_std_17)
target_compile_options(${PROJECT_NAME}_test PRIVATE -Wall -Wextra -Werror)
target_link_libraries(${PROJECT_NAME}_test Eigen3::Eigen Catch2::Catch2WithMain ${Boost_LIBRARIES})


include(Catch)
catch_discover_tests(${PROJECT_NAME}_test)
