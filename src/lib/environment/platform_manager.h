#pragma once

#include "gEng/utils/singletone_base.h"
#include "gEng/window.h"

#include "debug_callback.h"

#include <cstdint>
#include <memory>
#include <optional>

#include <vulkan/vulkan.hpp>

namespace gEng {

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

// This class designed to generate and release all platform specific vk handles.
// This is singleton.
class PltManager : public SingletoneBase<PltManager> {
  std::optional<vk::Instance> Instance{};
  std::optional<vk::Device> Device{};
  std::optional<vk::SurfaceKHR> Surface{};
  vk::PhysicalDevice PhysDev;

  // Determines if Device is matched requirements.
  bool isDeviceSuitable(vk::PhysicalDevice const &Device) const;

public:
#ifndef NDEBUG
  static bool constexpr EnableDebug{true};
#else
  static bool constexpr EnableDebug{false};
#endif

  vk::Instance createInstance();
  vk::SurfaceKHR createSurface(gEng::Window const &Window);
  vk::PhysicalDevice createPhysicalDevice();
  vk::Device createDevice();

  // Fills indices of queue families that support Presentation and Graphics
  QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice const &PhysDev) const;

  ~PltManager();

private:
  DebugMessenger<EnableDebug> DbgMsger;
};

} // namespace gEng
