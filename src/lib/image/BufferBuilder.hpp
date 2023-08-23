#pragma once

#include "../environment/platform_handler.h"
#include "../utils/BuilderInterface.hpp"
#include "BufferWithMemory.h"

#include <vulkan/vulkan.hpp>

namespace gEng {

struct BufferBuilder : BuilderInterface<BufferBuilder> {
  friend BuilderInterface<BufferBuilder>;

  using Type = BufferWithMemory;

  vk::Device Dev;
  vk::PhysicalDevice PhysDev;
  BufferBuilder(vk::Device Dev, vk::PhysicalDevice PhysDev)
      : Dev{Dev}, PhysDev{PhysDev} {}

  Type create(vk::DeviceSize Size, vk::BufferUsageFlags Usage,
              vk::MemoryPropertyFlags Properties) const;

  template <typename Data>
  Type createViaStaging(Data &&D, vk::BufferUsageFlags Usage,
                        vk::MemoryPropertyFlags Properties) const {
    using ValueTy = typename std::remove_reference_t<Data>::value_type;
    auto const Size = D.size() * sizeof(ValueTy);
    auto const *Src = D.data();
    auto Staging = create(Size, vk::BufferUsageFlagBits::eTransferSrc,
                          vk::MemoryPropertyFlagBits::eHostVisible |
                              vk::MemoryPropertyFlagBits::eHostCoherent);

    Staging.store(Src, Size);

    auto BufMem = create(Size, Usage, Properties);

    auto &PltMgn = PlatformHandler::getInstance();
    auto CmdBuff = PltMgn.getSSTC();
    CmdBuff.copyBuffer(Staging.Buffer, BufMem.Buffer,
                       vk::BufferCopy{{}, {}, Size});
    // Here quite tricky as in ~SSRC job emiter calls and at this point
    // all buffers should be consistent. But alse here ~Staging calls,
    // Which frees memory.
    return BufMem;
  }
};

} // namespace gEng
