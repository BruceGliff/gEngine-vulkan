#pragma once

#include <glm/glm.hpp>

#include <array>

// Forward declaration.
struct VkVertexInputBindingDescription;
struct VkVertexInputAttributeDescription;

struct Vertex {
  glm::vec2 Position;
  glm::vec3 Color;

  template <int No>
  using AttrDescr = std::array<VkVertexInputAttributeDescription, No>;
  static VkVertexInputBindingDescription getBindDescription();
  static AttrDescr<2> getAttrDescription();
};
