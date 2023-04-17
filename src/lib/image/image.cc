#include "gEng/image.h"

#include "BufferBuilder.hpp"

#include "../environment/platform_handler.h"

#include <stdexcept>
#include <tuple>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace gEng {

static bool hasStencilComponent(vk::Format Fmt) {
  return Fmt == vk::Format::eD32SfloatS8Uint ||
         Fmt == vk::Format::eD24UnormS8Uint;
}
// Finds right type of memory to use.
// FIXME copy-paste from BufferBuilder.cc
uint32_t findMemoryType(vk::PhysicalDevice PhysDev, uint32_t TypeFilter,
                        vk::MemoryPropertyFlags Properties) {
  vk::PhysicalDeviceMemoryProperties MemProps = PhysDev.getMemoryProperties();

  for (uint32_t i = 0; i != MemProps.memoryTypeCount; ++i)
    if ((TypeFilter & (1 << i)) &&
        (MemProps.memoryTypes[i].propertyFlags & Properties) == Properties)
      return i;

  throw std::runtime_error("failed to find suitable memory type!");
}

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

static void transitionImageLayout(PlatformHandler const &PltMgn,
                                  vk::Image Image, vk::Format Fmt,
                                  vk::ImageLayout OldLayout,
                                  vk::ImageLayout NewLayout, uint32_t MipLvls) {
  auto [SSTC, CmdBuffer] = PltMgn.getSSTC();

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
static void copyBufferToImage(PlatformHandler const &PltMgn, vk::Buffer Buffer,
                              vk::Image Image, uint32_t Width,
                              uint32_t Height) {
  auto [SSTC, CmdBuff] = PltMgn.getSSTC();

  vk::BufferImageCopy Reg{0,         0,
                          0,         {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                          {0, 0, 0}, {Width, Height, 1}};
  CmdBuff.copyBufferToImage(Buffer, Image, vk::ImageLayout::eTransferDstOptimal,
                            Reg);
}
static void generateMipmaps(PlatformHandler const &PltMgr, vk::Image Img,
                            vk::Format Fmt, uint32_t Width, uint32_t Height,
                            uint32_t MipLvls) {
  // Check if image format supports linear blitting.
  auto PhysDev = PltMgr.get<vk::PhysicalDevice>();
  vk::FormatProperties FmtProps = PhysDev.getFormatProperties(Fmt);

  if (!(FmtProps.optimalTilingFeatures &
        vk::FormatFeatureFlagBits::eSampledImageFilterLinear))
    throw std::runtime_error(
        "texture image format does not support linear blitting!");

  auto [SSTC, CmdBuff] = PltMgr.getSSTC();

  vk::ImageMemoryBarrier Barrier{
      {}, {}, {},  {},
      {}, {}, Img, {vk::ImageAspectFlagBits::eColor, {/*miplevel*/}, 1, 0, 1}};

  int32_t MipWidth = static_cast<int32_t>(Width);
  int32_t MipHeight = static_cast<int32_t>(Height);

  using IL = vk::ImageLayout;
  using AF = vk::AccessFlagBits;
  using Off = vk::Offset3D;
  for (uint32_t i = 1; i != MipLvls; ++i) {
    Barrier.subresourceRange.baseMipLevel = i - 1;
    Barrier.oldLayout = IL::eTransferDstOptimal;
    Barrier.newLayout = IL::eTransferSrcOptimal;
    Barrier.srcAccessMask = AF::eTransferWrite;
    Barrier.dstAccessMask = AF::eTransferRead;
    CmdBuff.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eTransfer, {}, nullptr,
                            nullptr, Barrier);

    auto GetHalf = [](int32_t MipDim) { return MipDim > 1 ? MipDim / 2 : 1; };
    vk::ImageBlit Blit{
        /*Src*/ {vk::ImageAspectFlagBits::eColor, i - 1, 0, 1},
        {Off{0, 0, 0}, Off{MipWidth, MipHeight, 1}},
        /*Dst*/ {vk::ImageAspectFlagBits::eColor, i, 0, 1},
        {Off{0, 0, 0}, Off{GetHalf(MipWidth), GetHalf(MipHeight), 1}}};
    // must be submitted to a queue with graphics capability.
    CmdBuff.blitImage(Img, IL::eTransferSrcOptimal, Img,
                      IL::eTransferDstOptimal, Blit, vk::Filter::eLinear);

    Barrier.oldLayout = IL::eTransferSrcOptimal;
    Barrier.newLayout = IL::eShaderReadOnlyOptimal;
    Barrier.srcAccessMask = AF::eTransferRead;
    Barrier.dstAccessMask = AF::eShaderRead;
    CmdBuff.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eFragmentShader, {},
                            nullptr, nullptr, Barrier);

    if (MipWidth > 1)
      MipWidth /= 2;
    if (MipHeight > 1)
      MipHeight /= 2;
  }

  Barrier.subresourceRange.baseMipLevel = MipLvls - 1;
  Barrier.oldLayout = IL::eTransferDstOptimal;
  Barrier.newLayout = IL::eShaderReadOnlyOptimal;
  Barrier.srcAccessMask = AF::eTransferWrite;
  Barrier.dstAccessMask = AF::eShaderRead;
  CmdBuff.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                          vk::PipelineStageFlagBits::eFragmentShader, {},
                          nullptr, nullptr, Barrier);
}

struct ImageBuilder : BuilderInterface<ImageBuilder> {
  friend BuilderInterface<ImageBuilder>;

  using Type = std::pair<vk::Image, vk::DeviceMemory>;

  vk::Device Dev;
  vk::PhysicalDevice PhysDev;
  ImageBuilder(vk::Device Dev, vk::PhysicalDevice PhysDev)
      : Dev{Dev}, PhysDev{PhysDev} {}

  // template <typename T, typename... Args> T create(Args &&...args);
  // TODO is this legal?
  template <typename T, typename... Args> T create(Args... args);
};

template <>
ImageBuilder::Type ImageBuilder::create<ImageBuilder::Type>(
    uint32_t Width, uint32_t Height, uint32_t MipLevls,
    vk::SampleCountFlagBits NumSample, vk::Format Fmt, vk::ImageTiling Tiling,
    vk::ImageUsageFlags Usage, vk::MemoryPropertyFlags Props) {
  vk::ImageCreateInfo ImageInfo{{},        vk::ImageType::e2D,
                                Fmt,       {Width, Height, 1},
                                MipLevls,  1,
                                NumSample, Tiling,
                                Usage,     vk::SharingMode::eExclusive};
  vk::Image Image = Dev.createImage(ImageInfo);
  vk::MemoryRequirements MemReq = Dev.getImageMemoryRequirements(Image);
  vk::DeviceMemory ImageMem = Dev.allocateMemory(
      {MemReq.size, findMemoryType(PhysDev, MemReq.memoryTypeBits, Props)});

  Dev.bindImageMemory(Image, ImageMem, 0);

  return {Image, ImageMem};
}

class ImageImpl final {
  // FIXME make it private
public:
  vk::Image Img;
  vk::DeviceMemory Mem;
  uint32_t mipLevels;

public:
  ImageImpl(void *RawData, uint32_t Width, uint32_t Height, uint32_t Size) {
    auto &PltMgn = PlatformHandler::getInstance();
    auto Dev = PltMgn.get<vk::Device>();
    auto PhysDev = PltMgn.get<vk::PhysicalDevice>();

    BufferBuilder BB{Dev, PhysDev};
    ImageBuilder IB{Dev, PhysDev};

    using MP = vk::MemoryPropertyFlagBits;
    using BU = vk::BufferUsageFlagBits;
    using IU = vk::ImageUsageFlagBits;
    auto [StagingBuff, StagingBuffMem] = BB.create<BufferBuilder::Type>(
        vk::DeviceSize{Size}, BU::eTransferSrc,
        vk::MemoryPropertyFlags{MP::eHostVisible | MP::eHostCoherent});

    void *Data = Dev.mapMemory(StagingBuffMem, 0, Size);
    memcpy(Data, RawData, Size);
    Dev.unmapMemory(StagingBuffMem);

    auto const MIPlvl =
        static_cast<uint32_t>(std::floor(std::log2(std::max(Width, Height)))) +
        1;
    // FIXME temprorary
    mipLevels = MIPlvl;

    std::tie(Img, Mem) = IB.create<ImageBuilder::Type>(
        Width, Height, MIPlvl, vk::SampleCountFlagBits::e1,
        vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal,
        IU::eTransferSrc | IU::eTransferDst | IU::eSampled, MP::eDeviceLocal);

    transitionImageLayout(PltMgn, Img, vk::Format::eR8G8B8A8Srgb,
                          vk::ImageLayout::eUndefined,
                          vk::ImageLayout::eTransferDstOptimal, MIPlvl);

    copyBufferToImage(PltMgn, StagingBuff, Img, Width, Height);
    // Transitioning to SHADER_READ_ONLY while generating mipmaps.
    generateMipmaps(PltMgn, Img, vk::Format::eR8G8B8A8Srgb, Width, Height,
                    MIPlvl);

    Dev.destroyBuffer(StagingBuff);
    Dev.freeMemory(StagingBuffMem);
  }
};

void setImg(vk::Image &I, vk::DeviceMemory &M, uint32_t &L, Image &Img) {
  I = Img.ImageVk->Img;
  M = Img.ImageVk->Mem;
  L = Img.ImageVk->mipLevels;
}

Image::Image(std::string_view Path) {
  auto [RawData, Width, Height, Size] = loadRawImg(Path);

  ImageVk = new ImageImpl{RawData, Width, Height, Size};

  // TODO move this free to loadRawImg dependency.
  stbi_image_free(RawData);
}

Image::~Image() { delete ImageVk; }

} // namespace gEng
