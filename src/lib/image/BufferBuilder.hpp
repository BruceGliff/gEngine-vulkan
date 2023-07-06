#pragma once

#include "../environment/platform_handler.h"
#include "../utils/BuilderInterface.hpp"

#include <vulkan/vulkan.hpp>

namespace gEng {

struct BufferBuilder : BuilderInterface<BufferBuilder> {
  friend BuilderInterface<BufferBuilder>;

  using Type = std::pair<vk::Buffer, vk::DeviceMemory>;

  vk::Device Dev;
  vk::PhysicalDevice PhysDev;
  BufferBuilder(vk::Device Dev, vk::PhysicalDevice PhysDev)
      : Dev{Dev}, PhysDev{PhysDev} {}

  Type create(vk::DeviceSize, vk::Flags<vk::MemoryPropertyFlagBits>,
              vk::Flags<vk::MemoryPropertyFlagBits>) const;

  template <typename Data>
  Type
  createViaStaging(Data &&D, vk::Flags<vk::MemoryPropertyFlagBits>,
                   vk::Flags<vk::MemoryPropertyFlagBits> Properties) const {
    auto const Size = D.size();
    auto const *Src = D.data();
    auto [StagingBuff, StagingBuffMem] = create<BufferBuilder::Type>(
        Size, vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent);

    void *Mmr = Dev.mapMemory(StagingBuffMem, 0, Size);
    std::memcpy(Mmr, Src, Size);
    Dev.unmapMemory(StagingBuffMem);

    auto BufMem =
        create<BufferBuilder::Type>(Size,
                                    vk::BufferUsageFlagBits::eTransferDst |
                                        vk::BufferUsageFlagBits::eVertexBuffer,
                                    vk::MemoryPropertyFlagBits::eDeviceLocal);

    auto &PltMgn = PlatformHandler::getInstance();
    auto [SSRC, CmdBuff] = PltMgn.getSSTC();
    CmdBuff.copyBuffer(StagingBuff, std::get<vk::Buffer>(BufMem),
                       vk::BufferCopy{{}, {}, Size});

    Dev.destroyBuffer(StagingBuff);
    Dev.freeMemory(StagingBuffMem);
    return BufMem;
  }
};

} // namespace gEng
