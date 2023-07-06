#include "BufferBuilder.hpp"

using namespace gEng;

// Finds right type of memory to use.
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
BufferBuilder::Type
BufferBuilder::create(vk::DeviceSize Size,
                      vk::Flags<vk::MemoryPropertyFlagBits> Usage,
                      vk::Flags<vk::MemoryPropertyFlagBits> Properties) const {
  // TODO for transfering VK_QUEUE_TRANSFER_BIT is needed, but it included in
  // VK_QUEUE_GRAPHICS_BIT or COMPUTE_BIT. But it would be nice to create
  // queue family specially with TRANSFER_BIT.
  vk::Buffer Buffer =
      Dev.createBuffer({{}, Size, Usage, vk::SharingMode::eExclusive});
  vk::MemoryRequirements MemReq = Dev.getBufferMemoryRequirements(Buffer);
  vk::DeviceMemory Memory = Dev.allocateMemory(
      {MemReq.size,
       findMemoryType(PhysDev, MemReq.memoryTypeBits, Properties)});
  Dev.bindBufferMemory(Buffer, Memory, 0);

  return std::make_pair(Buffer, Memory);
}
