#pragma once

#include "gEng/window.h"

#include "debug_callback.h"

#include <memory>
#include <optional>

#include <vulkan/vulkan.hpp>

namespace gEng {

// This class designed to generate and release all platform specific vk handles.
// This is singleton.
class PltManager {
  static std::unique_ptr<PltManager> Mgr;

  std::optional<vk::Instance> Instance{};
  std::optional<vk::Device> Dev{};
  std::optional<vk::SurfaceKHR> Surface{};
  vk::PhysicalDevice PhysDev;

  // Private constructor as it accessable from getInstance but only from it.
  PltManager() {}

public:
#ifndef NDEBUG
  static bool constexpr EnableDebug{true};
#else
  static bool constexpr EnableDebug{false};
#endif

  // Returns instance of the platform manager.
  static PltManager &getMgrInstance();

  vk::Instance createInstance();
  vk::SurfaceKHR createSurface(gEng::Window const &Window);

  ~PltManager();

private:
  DebugMessenger<EnableDebug> DbgMsger;
};

} // namespace gEng