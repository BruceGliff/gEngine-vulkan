#include "gEng/image.h"

#include "image_impl.h"

#include <stdexcept>
#include <tuple>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

using namespace gEng;

static std::tuple<void *, uint32_t, uint32_t, uint32_t>
loadRawImg(std::string_view Path) {
  int Width{0}, Height{0}, Channels{0};
  void *RawData = static_cast<void *>(
      stbi_load(Path.data(), &Width, &Height, &Channels, STBI_rgb_alpha));
  uint32_t const Size = Width * Height * 4;

  if (!RawData)
    throw std::runtime_error("failed to load texture image!");

  auto Cast = [](int X) { return static_cast<uint32_t>(X); };
  return std::make_tuple(RawData, Cast(Width), Cast(Height), Cast(Size));
}

Image::Image(std::string_view Path) {
  auto [RawData, Width, Height, Size] = loadRawImg(Path);

  ImageVk = new ImageImpl{RawData, Width, Height, Size};

  stbi_image_free(RawData);
}

Image::~Image() { delete ImageVk; }
