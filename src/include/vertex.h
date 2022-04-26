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

  // For using as key in unordered_map.
  bool operator==(const Vertex &other) const;
};

// For using as key in unordered_map.
namespace std {
template <> struct hash<Vertex> {
  size_t operator()(Vertex const &vertex) const;
};
} // namespace std