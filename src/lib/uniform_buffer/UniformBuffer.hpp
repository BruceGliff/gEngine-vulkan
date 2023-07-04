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

  vk::Device Dev;
  vk::Buffer Buf;
  vk::DeviceMemory Mem;

public:
  UniformBuffer() = delete;
  UniformBuffer(UniformBuffer const &) = delete;
  UniformBuffer(UniformBuffer &&) = delete;
  UniformBuffer(const PlatformHandler &PltMgr) : Dev{PltMgr.get<vk::Device>()} {
    BufferBuilder BB{Dev, PltMgr.get<vk::PhysicalDevice>()};
    std::tie(Buf, Mem) = BB.create<BufferBuilder::Type>(
        Size, vk::BufferUsageFlagBits::eUniformBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent);
  }

  auto getDescriptorBuffInfo() const {
    return vk::DescriptorBufferInfo{Buf, 0, Size};
  }

  void load(const UniformBufferObject &Obj) {
    void *Data = Dev.mapMemory(Mem, 0, Size);
    memcpy(Data, &Obj, Size);
    Dev.unmapMemory(Mem);
  }

  ~UniformBuffer() {
    Dev.destroyBuffer(Buf);
    Dev.freeMemory(Mem);
  }
};

} // namespace gEng
