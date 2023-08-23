#include "ImageBuilder.hpp"

#include "../environment/platform_handler.h"

using namespace gEng;

static bool hasStencilComponent(vk::Format Fmt) {
  return Fmt == vk::Format::eD32SfloatS8Uint ||
         Fmt == vk::Format::eD24UnormS8Uint;
}

void ImageBuilder::transitionImageLayout(PlatformHandler const &PltMgn,
                                         vk::Image Image, vk::Format Fmt,
                                         vk::ImageLayout OldLayout,
                                         vk::ImageLayout NewLayout,
                                         uint32_t MipLvls) {
  auto CmdBuffer = PltMgn.getSSTC();

  vk::ImageMemoryBarrier Barrier{
      {}, {}, OldLayout, NewLayout,
      {}, {}, Image,     {vk::ImageAspectFlagBits::eColor, 0, MipLvls, 0, 1}};
  vk::PipelineStageFlags SrcStage;
  vk::PipelineStageFlags DstStage;

  using IL = vk::ImageLayout;
  if (NewLayout == IL::eDepthStencilAttachmentOptimal) {
    Barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;
    if (hasStencilComponent(Fmt))
      Barrier.subresourceRange.aspectMask |= vk::ImageAspectFlagBits::eStencil;
    // TODO this else causes validation error. But it is actually useless.
    // else
    //  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  }

  using AF = vk::AccessFlagBits;
  using PS = vk::PipelineStageFlagBits;
  if (OldLayout == IL::eUndefined && NewLayout == IL::eTransferDstOptimal) {
    Barrier.srcAccessMask = {};
    Barrier.dstAccessMask = AF::eTransferWrite;
    SrcStage = PS::eTopOfPipe;
    DstStage = PS::eTransfer;
  } else if (OldLayout == IL::eTransferDstOptimal &&
             NewLayout == IL::eShaderReadOnlyOptimal) {
    Barrier.srcAccessMask = AF::eTransferWrite;
    Barrier.dstAccessMask = AF::eShaderRead;
    SrcStage = PS::eTransfer;
    DstStage = PS::eFragmentShader;
  } else if (OldLayout == IL::eUndefined &&
             NewLayout == IL::eDepthStencilAttachmentOptimal) {
    Barrier.srcAccessMask = {};
    Barrier.dstAccessMask =
        AF::eDepthStencilAttachmentRead | AF::eDepthStencilAttachmentWrite;
    SrcStage = PS::eTopOfPipe;
    DstStage = PS::eEarlyFragmentTests;
  } else
    throw std::invalid_argument("unsupported layout transition!");

  CmdBuffer.pipelineBarrier(SrcStage, DstStage, {}, nullptr, nullptr, Barrier);
}

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
