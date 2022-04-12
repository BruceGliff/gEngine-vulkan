#include "image/image.h"

#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

image::image(std::string_view Pass) {
  int texWidth, texHeight, texChannels;
  RawData = static_cast<void *>(stbi_load(Pass.data(), &texWidth, &texHeight,
                                          &texChannels, STBI_rgb_alpha));
  Size = texWidth * texHeight * 4;

  if (!RawData)
    throw std::runtime_error("failed to load texture image!");
}

image::~image() { stbi_image_free(RawData); }

void *image::getRawData() const { return RawData; }

uint32_t image::getSize() const { return Size; }
