#pragma once

#include <optional>
#include <cstdint>

namespace vk {
  class PhysicalDevice;
  class SurfaceKHR;
} // namespace vk

namespace gEng {

// Builder for platform specific handles.
// Used in PlatformManager.
// All function defined in PlatformBuilder.cc
struct PlatformBuilder final {
  template <typename T, typename... Args> static T create(Args &&...args);

#ifndef NDEBUG
  static bool constexpr EnableDebug{true};
#else
  static bool constexpr EnableDebug{false};
#endif
};

// Deleter for platform specific handles.
struct PlatformDeleter final {
  template <typename T> static void destroy(T &&) {}
};

// Checks which queue families are supported by the device and which one of
// these supports the commands that we want to use.
struct QueueFamilyIndices {
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
  QueueFamilyIndices findQueueFamilies(vk::SurfaceKHR, vk::PhysicalDevice const &);

} // namespace gEng
