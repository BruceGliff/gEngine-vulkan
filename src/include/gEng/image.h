#pragma once

#include <cstdint>
#include <string_view>

namespace gEng {

class ImageImpl;

// This class is a wrap of implementation around VkImage.
class Image {
  ImageImpl *ImageVk;

public:
  Image(std::string_view Path);
  ~Image();
};

} // namespace gEng
