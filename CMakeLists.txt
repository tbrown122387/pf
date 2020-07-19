# File              : CMakeLists.txt
# Author            : chenxu <chenxu@mail.ustc.edu.cn>
# Date              : 6 2020-01-18 01:36:45
# Last Modified Date: 6 2020-01-18 01:36:45

cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(pf)

file(GLOB headers 
  ${PROJECT_SOURCE_DIR}/include/pf/*.h)

install(FILES ${headers} DESTINATION include/pf)
