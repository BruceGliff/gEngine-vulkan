#pragma once

#include <glm/glm.hpp>

#include <array>

// Forward declaration.
struct VkVertexInputBindingDescription;
struct VkVertexInputAttributeDescription;

struct Vertex {
  glm::vec3 Position;
  glm::vec3 Color;
  glm::vec2 TexCoord;

  template <int No>
  using AttrDescr = std::array<VkVertexInputAttributeDescription, No>;
  static VkVertexInputBindingDescription getBindDescription();
  static AttrDescr<3> getAttrDescription();
};
