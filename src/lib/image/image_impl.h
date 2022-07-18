#pragma once

#include <cstdint>

#include <vulkan/vulkan.hpp>

namespace gEng {

class ImageImpl {

  uint32_t mipLvl;

  vk::Image Img;
  vk::DeviceMemory Mem;

public:
  ImageImpl(void *RawData, uint32_t Width, uint32_t Height, uint32_t Size);
};

} // namespace gEng
