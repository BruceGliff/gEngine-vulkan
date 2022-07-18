#include "global.h"

static uint32_t findMemoryType(uint32_t TypeFilter,
                               vk::MemoryPropertyFlags Properties);

namespace GL {
// RIGHT NOW THIS IS HARSH APPROACH JUST FOR TEST.

// TODO later this has to be class with distructor/constructor/singleton
// handler, chekers of invalid accesses and so one.
vk::Device Dev;
void setDevice(vk::Device Device) { Dev = Device; }
vk::Device getDevice() { return Dev; }

vk::PhysicalDevice PhysDev;
void setDevice(vk::PhysicalDevice Device) { PhysDev = Device; }
vk::PhysicalDevice getPhysDevice() { return PhysDev; }

namespace call {
std::pair<vk::Buffer, vk::DeviceMemory>
createBuffer(vk::DeviceSize Size, vk::BufferUsageFlags Usage,
             vk::MemoryPropertyFlags Properties) {
  // TODO for transfering VK_QUEUE_TRANSFER_BIT is needed, but it included in
  // VK_QUEUE_GRAPHICS_BIT or COMPUTE_BIT. But it would be nice to create
  // queue family specially with TRANSFER_BIT.
  vk::Buffer Buffer =
      Dev.createBuffer({{}, Size, Usage, vk::SharingMode::eExclusive});
  vk::MemoryRequirements MemReq = Dev.getBufferMemoryRequirements(Buffer);
  vk::DeviceMemory Memory = Dev.allocateMemory(
      {MemReq.size, findMemoryType(MemReq.memoryTypeBits, Properties)});
  Dev.bindBufferMemory(Buffer, Memory, 0);

  return std::make_pair(Buffer, Memory);
}

std::pair<vk::Image, vk::DeviceMemory>
createImage(uint32_t Width, uint32_t Height, uint32_t MipLevls,
            vk::SampleCountFlagBits NumSample, vk::Format Fmt,
            vk::ImageTiling Tiling, vk::ImageUsageFlags Usage,
            vk::MemoryPropertyFlags Props) {
  vk::ImageCreateInfo ImageInfo{{},        vk::ImageType::e2D,
                                Fmt,       {Width, Height, 1},
                                MipLevls,  1,
                                NumSample, Tiling,
                                Usage,     vk::SharingMode::eExclusive};
  vk::Image Image = Dev.createImage(ImageInfo);
  vk::MemoryRequirements MemReq = Dev.getImageMemoryRequirements(Image);
  vk::DeviceMemory ImageMem = Dev.allocateMemory(
      {MemReq.size, findMemoryType(MemReq.memoryTypeBits, Props)});

  Dev.bindImageMemory(Image, ImageMem, 0);

  return {Image, ImageMem};
}

void transitionImageLayout(vk::Image Image, vk::Format Fmt,
                           vk::ImageLayout OldLayout, vk::ImageLayout NewLayout,
                           uint32_t MipLvls) {
  vk::CommandBuffer CmdBuffer = beginSingleTimeCommands();

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

  endSingleTimeCommands(CmdBuffer);
}

} // namespace call

} // namespace GL

static uint32_t findMemoryType(uint32_t TypeFilter,
                               vk::MemoryPropertyFlags Properties) {
  vk::PhysicalDeviceMemoryProperties MemProps =
      GL::PhysDev.getMemoryProperties();

  for (uint32_t i = 0; i != MemProps.memoryTypeCount; ++i)
    if ((TypeFilter & (1 << i)) &&
        (MemProps.memoryTypes[i].propertyFlags & Properties) == Properties)
      return i;

  throw std::runtime_error("failed to find suitable memory type!");
}