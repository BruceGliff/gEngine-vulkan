#version 450

layout(binding = 0) uniform UniformBufferOBject {
  mat4 model;
  mat4 view;
  mat4 proj;
} ubo;

layout(location = 0) in vec2 inPosiiton;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 outFragColor;

void main() {
  gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosiiton, 0.0, 1.0);
  outFragColor = inColor;
}
