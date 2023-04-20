#pragma once

#include "../utils/BuilderInterface.hpp"

#include <vulkan/vulkan.hpp>

namespace gEng {

struct ImageBuilder : BuilderInterface<ImageBuilder> {
  friend BuilderInterface<ImageBuilder>;

  using Type = std::pair<vk::Image, vk::DeviceMemory>;

  vk::Device Dev;
  vk::PhysicalDevice PhysDev;
  ImageBuilder(vk::Device Dev, vk::PhysicalDevice PhysDev)
      : Dev{Dev}, PhysDev{PhysDev} {}

  template <typename T, typename... Args> T create(Args... args);
};

} // namespace gEng
