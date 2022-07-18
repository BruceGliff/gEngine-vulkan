#include "image_impl.h"

#include "../global.h"

#include <cmath>

using namespace gEng;

ImageImpl::ImageImpl(void *RawData, uint32_t Width, uint32_t Height,
                     uint32_t Size) {

  vk::Device Dev = GL::getDevice();

  mipLvl =
      static_cast<uint32_t>(std::floor(std::log2(std::max(Width, Height)))) + 1;

  auto [StagingBuff, StagingBuffMem] =
      GL::call::createBuffer(Size, vk::BufferUsageFlagBits::eTransferSrc,
                             vk::MemoryPropertyFlagBits::eHostVisible |
                                 vk::MemoryPropertyFlagBits::eHostCoherent);

  void *Data = Dev.mapMemory(StagingBuffMem, 0, Size);
  memcpy(Data, RawData, Size);
  Dev.unmapMemory(StagingBuffMem);

  std::tie(Img, Mem) = GL::call::createImage(
      Width, Height, mipLvl, vk::SampleCountFlagBits::e1,
      vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal,
      vk::ImageUsageFlagBits::eTransferSrc |
          vk::ImageUsageFlagBits::eTransferDst |
          vk::ImageUsageFlagBits::eSampled,
      vk::MemoryPropertyFlagBits::eDeviceLocal);

  transitionImageLayout(textureImage, vk::Format::eR8G8B8A8Srgb,
                        vk::ImageLayout::eUndefined,
                        vk::ImageLayout::eTransferDstOptimal, mipLevels);
  copyBufferToImage(StagingBuff, textureImage, Width, Height);
  // Transitioning to SHADER_READ_ONLY while generating mipmaps.
  generateMipmaps(textureImage, vk::Format::eR8G8B8A8Srgb, Width, Height,
                  mipLevels);

  Dev.destroyBuffer(StagingBuff);
  Dev.freeMemory(StagingBuffMem);
}