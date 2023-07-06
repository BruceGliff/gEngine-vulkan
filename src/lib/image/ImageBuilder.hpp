#pragma once

#include "../utils/BuilderInterface.hpp"

#include <vulkan/vulkan.hpp>

namespace gEng {
class PlatformHandler;

struct ImageBuilder : BuilderInterface<ImageBuilder> {
  friend BuilderInterface<ImageBuilder>;

  using Type = std::pair<vk::Image, vk::DeviceMemory>;

  vk::Device Dev;
  vk::PhysicalDevice PhysDev;
  ImageBuilder(vk::Device Dev, vk::PhysicalDevice PhysDev)
      : Dev{Dev}, PhysDev{PhysDev} {}

  void transitionImageLayout(PlatformHandler const &, vk::Image, vk::Format,
                             vk::ImageLayout, vk::ImageLayout, uint32_t);

  template <typename T, typename... Args> T create(Args... args);
};

} // namespace gEng
