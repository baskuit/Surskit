cmake_minimum_required(VERSION 3.8.0)
set (CMAKE_CXX_STANDARD 23)

project(pinyon VERSION 0.1.0)

add_library(pinyon INTERFACE)
target_include_directories(pinyon INTERFACE include)

add_executable(main main.cc)
target_link_libraries(main PRIVATE pinyon)