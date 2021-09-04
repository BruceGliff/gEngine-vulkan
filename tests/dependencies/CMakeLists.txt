# CMake for external dependencies test.
# The program "./test_dependencies" should launch with empty window and no errors

cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)

set(PROJECT_NAME test_dependencies)
project(${PROJECT_NAME})

set(CMAKE_CXX_COMPILER clang++)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES_CC src/main.cc)
add_executable(${PROJECT_NAME} ${SOURCES_CC})

set_source_files_properties(${SOURCES_CC} COMPILE_FLAGS "-Wall -O2 -g")


# template from GLFW Documentation
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

target_link_libraries(${PROJECT_NAME} glfw)
target_link_libraries(${PROJECT_NAME} vulkan)


set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)

add_custom_command(
  TARGET ${PROJECT_NAME} POST_BUILD
  COMMAND ${PROJECT_NAME}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin/
  COMMENT "Runs test program to check external dependencies. ${CMAKE_BINARY_DIR}/bin/${PROJECT_NAME}"
)
