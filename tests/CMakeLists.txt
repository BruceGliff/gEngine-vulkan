# CMake for testing
# Launchs test builds

cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)

project(test)

add_subdirectory(dependencies)
add_subdirectory(simple-launch)

set(CMAKE_CXX_COMPILER /usr/bin/clang++)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
