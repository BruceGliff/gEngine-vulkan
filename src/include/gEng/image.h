#pragma once

#include <cstdint>
#include <string_view>

namespace vk {
class Image;
class DeviceMemory;
class ImageView;
class Sampler;
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

void setImg(vk::Image &I, vk::DeviceMemory &M, uint32_t &L, vk::ImageView &IW,
            vk::Sampler &S, Image &Img);

} // namespace gEng
