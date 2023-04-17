#pragma once

#include "../utils/BuilderInterface.hpp"

#include <vulkan/vulkan.hpp>

namespace gEng {

struct BufferBuilder : BuilderInterface<BufferBuilder> {
  friend BuilderInterface<BufferBuilder>;

  using Type = std::pair<vk::Buffer, vk::DeviceMemory>;

  vk::Device Dev;
  vk::PhysicalDevice PhysDev;
  BufferBuilder(vk::Device Dev, vk::PhysicalDevice PhysDev)
      : Dev{Dev}, PhysDev{PhysDev} {}

  // template <typename T, typename... Args> T create(Args &&...args);
  // TODO Is this ok? NO: AMBIGUOUS
  template <typename T, typename... Args> T create(Args... args);
};

} // namespace gEng
