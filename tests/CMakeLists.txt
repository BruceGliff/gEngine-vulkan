# CMake for testing
# Launches test builds

cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)

project(test)

add_subdirectory(simple-launch)
add_subdirectory(dependencies)
add_subdirectory(objloader)
add_subdirectory(opt-parser)
