# Launches testing and project building

cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)

set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_C_COMPILER clang)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

project(Initial)

add_subdirectory(external/glm)
add_subdirectory(external/glfw)
add_subdirectory(external/tiniobjloader)

find_package(Vulkan REQUIRED)

set(LIBRARY_INCLUDE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/external/glm/
  ${CMAKE_CURRENT_SOURCE_DIR}/external/glfw/include
  ${CMAKE_CURRENT_SOURCE_DIR}/external/stb
  ${CMAKE_CURRENT_SOURCE_DIR}/external/tiniobjloader/
  ${Vulkan_INCLUDE_DIRS}
   )

add_subdirectory(tests)
add_subdirectory(src)
