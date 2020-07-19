cmake_minimum_required(VERSION 3.12)

project("pf" VERSION 1.0.1
             DESCRIPTION "A header-only c++ template library for fast particle filtering."
             HOMEPAGE_URL "https://github.com/tbrown122387/pf")

include(GNUInstallDirs)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Boost 1.65.1 REQUIRED)
find_package(Catch2 2.9.2 REQUIRED)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(
  ${PROJECT_NAME}
  INTERFACE $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(${PROJECT_NAME} 
    INTERFACE 
        Eigen3::Eigen 
        ${Boost_LIBRARIES}
        Catch2::Catch2)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

## INSTALL 
install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}_Targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

include(CMakePackageConfigHelpers)
write_basic_package_version_file("${PROJECT_NAME}ConfigVersion.cmake"
                                 VERSION ${PROJECT_VERSION}
                                 COMPATIBILITY SameMajorVersion)


configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION
  ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)

install(EXPORT ${PROJECT_NAME}_Targets
        FILE ${PROJECT_NAME}Targets.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)
install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
              "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/pf DESTINATION include)

## Tests
include(CTest)
enable_testing()
add_subdirectory(test)


