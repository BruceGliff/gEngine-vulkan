#pragma once

#include <vulkan/vulkan.hpp>

// These classes and calls for vk-dependent objects, which permanent through
// whole program.

// GLOBAL
namespace GL {

void setDevice(vk::Device Dev);
vk::Device getDevice();

namespace call {
std::pair<vk::Buffer, vk::DeviceMemory>
createBuffer(vk::DeviceSize Size, vk::BufferUsageFlags Usage,
             vk::MemoryPropertyFlags Properties);

std::pair<vk::Image, vk::DeviceMemory>
createImage(uint32_t Width, uint32_t Height, uint32_t MipLevls,
            vk::SampleCountFlagBits NumSample, vk::Format Fmt,
            vk::ImageTiling Tiling, vk::ImageUsageFlags Usage,
            vk::MemoryPropertyFlags Props);
} // namespace call

} // namespace GL