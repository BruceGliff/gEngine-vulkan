#Project building

set(PROJECT_NAME gEngine)
project(${PROJECT_NAME})

file(GLOB_RECURSE SRC_FILES lib/*.cc)
add_executable(${PROJECT_NAME} main.cc ${SRC_FILES})

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -O0 -g )
target_include_directories(${PROJECT_NAME} PUBLIC ${LIBRARY_INCLUDE_PATH})
target_link_libraries(${PROJECT_NAME} glfw Vulkan::Vulkan)

set(OUT_DIR ${CMAKE_BINARY_DIR}/bin/${PROJECT_NAME})
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${OUT_DIR})

#Shader compiling.
function(compile_shader)
  set(SHADER_OUTPUT_DIR ${OUT_DIR}/assets/shaders)
  set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
  file(GLOB SHADERS ${SHADER_DIR}/*.vert ${SHADER_DIR}/*.frag)

  add_custom_command(
    OUTPUT ${SHADER_OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUTPUT_DIR}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Creating assets/shaders directory."
  )
  foreach(SHADER IN LISTS SHADERS)
    get_filename_component(FILENAME ${SHADER} NAME)
    add_custom_command(
      OUTPUT ${SHADER_OUTPUT_DIR}/${FILENAME}.spv
      COMMAND glslc ${SHADER} -o ${SHADER_OUTPUT_DIR}/${FILENAME}.spv
      DEPENDS ${SHADER} ${SHADER_OUTPUT_DIR}
      COMMENT "Compiling ${FILENAME}"
    )
    list(APPEND SPV_SHADERS ${SHADER_OUTPUT_DIR}/${FILENAME}.spv)
  endForeach()

  set(SHADER_TARGET compile-shaders)
  add_custom_target(${SHADER_TARGET} ALL DEPENDS ${SPV_SHADERS})
  add_dependencies(${PROJECT_NAME} ${SHADER_TARGET})
endfunction()

compile_shader()
