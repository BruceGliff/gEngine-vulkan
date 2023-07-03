#pragma once

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

#include <vulkan/vulkan.hpp>

#include "../environment/platform_handler.h"
// FIXME create separate folder.
#include "../image/BufferBuilder.hpp"

namespace gEng {

struct UniformBufferObject final {
  glm::mat4 Model;
  glm::mat4 View;
  glm::mat4 Proj;
};

struct UniformBuffer {
  // FIXME should be private
  static constexpr auto Size = sizeof(UniformBufferObject);

  vk::Buffer Buf;
  vk::DeviceMemory Mem;

  BufferBuilder BB;

public:
  UniformBuffer(const PlatformHandler &PltMgr)
      : BB{PltMgr.get<vk::Device>(), PltMgr.get<vk::PhysicalDevice>()} {
    std::tie(Buf, Mem) = BB.create<BufferBuilder::Type>(
        Size, vk::BufferUsageFlagBits::eUniformBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent);
  }
};

} // namespace gEng
