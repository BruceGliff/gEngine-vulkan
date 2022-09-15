# CMake for external dependencies test (option parser and boost library check).
# The program "./test_opt_parser" should compile and launch.

cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)

set(PROJECT_NAME test_opt_parser)
project(${PROJECT_NAME})

add_executable(${PROJECT_NAME} src/main.cc)

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -O2 -g)

target_include_directories(${PROJECT_NAME} PUBLIC ${LIBRARY_INCLUDE_PATH})
target_link_libraries(${PROJECT_NAME} Boost::program_options)

set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${PROJECT_NAME})

add_custom_command(
  TARGET ${PROJECT_NAME} POST_BUILD
  COMMAND ${PROJECT_NAME} --random_number 999 | grep 999
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin/
  COMMENT "Runs test program to check external dependencies (option parser loader). ${CMAKE_BINARY_DIR}/bin/${PROJECT_NAME}"
)
