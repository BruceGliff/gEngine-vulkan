#pragma once

#include <glm/glm.hpp>

// For hashing.
#include <array>
#include <functional>

// Forward declaration.
namespace vk {
struct VertexInputBindingDescription;
struct VertexInputAttributeDescription;
} // namespace vk

struct Vertex {
  glm::vec3 Position;
  glm::vec3 Color;
  glm::vec2 TexCoord;

  template <int No>
  using AttrDescr = std::array<vk::VertexInputAttributeDescription, No>;
  static vk::VertexInputBindingDescription getBindDescription();
  static AttrDescr<3> getAttrDescription();

  static constexpr auto size() { return sizeof(Vertex); }

  // For using as key in unordered_map.
  bool operator==(const Vertex &other) const;
};

// For using as key in unordered_map.
namespace std {
template <> struct hash<Vertex> {
  size_t operator()(Vertex const &vertex) const;
};
} // namespace std
