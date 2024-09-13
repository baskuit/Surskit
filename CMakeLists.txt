cmake_minimum_required(VERSION 3.8.0)
set (CMAKE_CXX_STANDARD 23)

project(pinyon VERSION 0.2.0)

add_library(pinyon INTERFACE)
target_include_directories(pinyon INTERFACE include)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  add_executable(main main.cc)
  target_link_libraries(main PRIVATE pinyon)
endif()