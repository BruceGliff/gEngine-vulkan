#include "ImageBuilder.hpp"

using namespace gEng;

// Finds right type of memory to use.
// FIXME copy-paste from BufferBuilder.cc
static uint32_t findMemoryType(vk::PhysicalDevice PhysDev, uint32_t TypeFilter,
                               vk::MemoryPropertyFlags Properties) {
  vk::PhysicalDeviceMemoryProperties MemProps = PhysDev.getMemoryProperties();

  for (uint32_t i = 0; i != MemProps.memoryTypeCount; ++i)
    if ((TypeFilter & (1 << i)) &&
        (MemProps.memoryTypes[i].propertyFlags & Properties) == Properties)
      return i;

  throw std::runtime_error("failed to find suitable memory type!");
}

// FIXME use commit history to find previous (non-working) impl.
// and try to fix it.
template <>
ImageBuilder::Type ImageBuilder::create<ImageBuilder::Type>(
    uint32_t Width, uint32_t Height, uint32_t MipLevls,
    vk::SampleCountFlagBits NumSample, vk::Format Fmt, vk::ImageTiling Tiling,
    vk::Flags<vk::ImageUsageFlagBits> Usage, vk::MemoryPropertyFlagBits Props) {
  vk::ImageCreateInfo ImageInfo{{},        vk::ImageType::e2D,
                                Fmt,       {Width, Height, 1},
                                MipLevls,  1,
                                NumSample, Tiling,
                                Usage,     vk::SharingMode::eExclusive};
  vk::Image Image = Dev.createImage(ImageInfo);
  vk::MemoryRequirements MemReq = Dev.getImageMemoryRequirements(Image);
  // FIXME
  // Allocate Memory will move. Think:
  // ImageBuilder : MemoryBuilder
  // BufferBuilder : MemoryBuilder
  vk::DeviceMemory ImageMem = Dev.allocateMemory(
      {MemReq.size, findMemoryType(PhysDev, MemReq.memoryTypeBits, Props)});

  Dev.bindImageMemory(Image, ImageMem, 0);

  return {Image, ImageMem};
}

template <>
vk::ImageView
ImageBuilder::create<vk::ImageView>(vk::Image Image, vk::Format Format,
                                    vk::ImageAspectFlagBits AspectFlags,
                                    uint32_t MipLevels) {
  return Dev.createImageView({{},
                              Image,
                              vk::ImageViewType::e2D,
                              Format,
                              {},
                              {AspectFlags, 0, MipLevels, 0, 1}});
}
