# Launches testing and project building

cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)

project(Initial)

add_subdirectory(tests)
add_subdirectory(src)

set(CMAKE_CXX_COMPILER /usr/bin/clang++)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)