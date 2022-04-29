#include "vertex.h"

// Experimental for hashing.
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

VkVertexInputBindingDescription Vertex::getBindDescription() {
  VkVertexInputBindingDescription bindingDescription{
      .binding = 0,
      .stride = sizeof(Vertex),
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX};
  return bindingDescription;
}

Vertex::AttrDescr<3> Vertex::getAttrDescription() {
  AttrDescr<3> Descriptions;
  Descriptions[0].binding = 0;
  Descriptions[0].location = 0;
  Descriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
  Descriptions[0].offset = offsetof(Vertex, Position);

  Descriptions[1].binding = 0;
  Descriptions[1].location = 1;
  Descriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
  Descriptions[1].offset = offsetof(Vertex, Color);

  Descriptions[2].binding = 0;
  Descriptions[2].location = 2;
  Descriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
  Descriptions[2].offset = offsetof(Vertex, TexCoord);

  return Descriptions;
}

bool Vertex::operator==(const Vertex &other) const {
  return Position == other.Position && Color == other.Color &&
         TexCoord == other.TexCoord;
}

// For using as key in unordered_map.
size_t std::hash<Vertex>::operator()(Vertex const &vertex) const {
  return ((hash<glm::vec3>()(vertex.Position) ^
           (hash<glm::vec3>()(vertex.Color) << 1)) >>
          1) ^
         (hash<glm::vec2>()(vertex.TexCoord) << 1);
}
