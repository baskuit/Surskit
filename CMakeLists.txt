cmake_minimum_required(VERSION 3.8.0)
set (CMAKE_CXX_STANDARD 23)

project(pinyon VERSION 0.1.0)

add_library(pinyon INTERFACE)

target_include_directories(pinyon INTERFACE src)

add_subdirectory(extern/lrslib)
target_link_libraries(pinyon INTERFACE lrsnashlib)

# Testing
file(GLOB_RECURSE TEST_FILES tests/*.cc)

foreach(TEST_FILE ${TEST_FILES})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    string(REPLACE "/" "_" TEST_TARGET_NAME ${TEST_NAME})
    add_executable(${TEST_TARGET_NAME} ${TEST_FILE})
    target_link_libraries(${TEST_TARGET_NAME} pinyon)
    add_test(NAME ${TEST_TARGET_NAME} COMMAND ${TEST_TARGET_NAME})
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES
                          RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
endforeach()
enable_testing()

# Benchmarking
file(GLOB BENCHMARK_FILES benchmark/*.cc)

foreach(FILE ${BENCHMARK_FILES})
    get_filename_component(B_NAME ${FILE} NAME_WE)
    add_executable(${B_NAME} ${FILE})
    target_link_libraries(${B_NAME} pinyon)
    set_target_properties(${B_NAME} PROPERTIES
                      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmark)
endforeach()
