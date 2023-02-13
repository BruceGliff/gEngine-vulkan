#pragma once

#include <cstdint>
#include <optional>

#include <vulkan/vulkan.hpp>

namespace gEng {
namespace detail {

// Checks which queue families are supported by the device and which one of
// these supports the commands that we want to use.
struct QueueFamilyIndices final {
  // optional just because we may be want to select GPU with some family, but
  // it is not strictly necessary.
  std::optional<uint32_t> GraphicsFamily{};
  // Not every device can support presentation of the image, so need to
  // check that divece has proper family queue.
  std::optional<uint32_t> PresentFamily{};

  bool isComplete() {
    return GraphicsFamily.has_value() && PresentFamily.has_value();
  }
};

inline QueueFamilyIndices findQueueFamilies(vk::SurfaceKHR Surface,
                                            vk::PhysicalDevice PhysDev) {
  QueueFamilyIndices Indices;
  std::vector<vk::QueueFamilyProperties> QueueFamilies =
      PhysDev.getQueueFamilyProperties();

  // TODO: rething this approach.
  uint32_t i{0};
  for (auto &&queueFamily : QueueFamilies) {
    // For better performance one queue family has to support all requested
    // queues at once, but we also can treat them as different families for
    // unified approach.
    if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)
      Indices.GraphicsFamily = i;

    // Checks for presentation family support.
    if (PhysDev.getSurfaceSupportKHR(i, Surface))
      Indices.PresentFamily = i;

    // Not quite sure why the hell we need this early-break.
    if (Indices.isComplete())
      return Indices;
    ++i;
  }

  return QueueFamilyIndices{};
}

// Swapchain requires more details to be checked.
// - basic surface capabilities.
// - surface format (pixel format, color space).
// - available presentation mode.
struct SwapchainSupportDetails final {
  vk::SurfaceCapabilitiesKHR capabilities;
  std::vector<vk::SurfaceFormatKHR> formats;
  std::vector<vk::PresentModeKHR> presentModes;
};

// TODO maybe everything regarding swapchain move to another module.
// This section covers how to query the structs that include this information.
static SwapchainSupportDetails
querySwapchainSupport(vk::SurfaceKHR Surface, vk::PhysicalDevice Device) {
  return SwapchainSupportDetails{
      .capabilities = Device.getSurfaceCapabilitiesKHR(Surface),
      .formats = Device.getSurfaceFormatsKHR(Surface),
      .presentModes = Device.getSurfacePresentModesKHR(Surface)};
}

} // namespace detail
} // namespace gEng
