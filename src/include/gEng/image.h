#pragma once

#include <cstdint>
#include <string_view>

namespace vk {
class Image;
class DeviceMemory;
} // namespace vk

namespace gEng {

class ImageImpl;

// This class is a wrap of implementation around VkImage.
// FIXME struct is temporary
struct Image {
  ImageImpl *ImageVk;

public:
  Image(std::string_view Path);
  ~Image();
};

void setImg(vk::Image &, vk::DeviceMemory &, uint32_t &, Image &);

} // namespace gEng
