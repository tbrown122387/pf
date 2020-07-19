project(pf_example)

cmake_minimum_required (VERSION 3.12)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 17)

# "install" pf
find_package(pf CONFIG REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Catch2 2.9.2)

file(GLOB SOURCES ${CMAKE_SOURCE_DIR}/*.h ${CMAKE_SOURCE_DIR}/*.cpp)
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} pf::pf) 
