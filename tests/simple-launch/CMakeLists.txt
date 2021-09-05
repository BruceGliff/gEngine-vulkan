# CMake for compilation test.
# The program "./simply-launch" should return with 0 code

cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)

set(PROJECT_NAME simply-launch)
project(${PROJECT_NAME})

add_executable(${PROJECT_NAME} src/main.cc)

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -O2 -g)


set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${PROJECT_NAME})

add_custom_command(
  TARGET ${PROJECT_NAME} POST_BUILD
  COMMAND ${PROJECT_NAME}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin/
  COMMENT "Runs test program to CMake. ${CMAKE_BINARY_DIR}/bin/${PROJECT_NAME}"
)