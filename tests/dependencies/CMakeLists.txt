# CMake for external dependencies test.
# The program "./test_dependencies" should launch with empty window and no errors

cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)

set(PROJECT_NAME test_dependencies)
project(${PROJECT_NAME})

add_executable(${PROJECT_NAME} src/main.cc)

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -O2 -g)

target_include_directories(${PROJECT_NAME} PUBLIC ${LIBRARY_INCLUDE_PATH})
target_link_libraries(${PROJECT_NAME} glfw Vulkan::Vulkan)

set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${PROJECT_NAME})

add_custom_command(
  TARGET ${PROJECT_NAME} POST_BUILD
  COMMAND ${PROJECT_NAME}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin/
  COMMENT "Runs test program to check external dependencies. ${CMAKE_BINARY_DIR}/bin/${PROJECT_NAME}"
)
