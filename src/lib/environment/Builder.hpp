#pragma once

#include <vulkan/vulkan.hpp>

#include "../utils/BuilderInterface.hpp"

namespace gEng {

struct PltBuilder : BuilderInterface<PltBuilder> {
  friend BuilderInterface<PltBuilder>;
  template <typename T, typename... Args> T create(Args &&...args);

#ifndef NDEBUG
  static bool constexpr EnableDebug{true};
#else
  static bool constexpr EnableDebug{false};
#endif
};

// Forward declaration.
class PlatformHandler;
struct ChainsBuilder : BuilderInterface<ChainsBuilder> {
  friend BuilderInterface<ChainsBuilder>;
  vk::Format Fmt{};
  vk::Extent2D Ext{};
  // TODO why do not put Plt as member?

  vk::Format findDepthFmt(vk::PhysicalDevice PhysDev) const;

  template <typename T, typename... Args>
  T create(PlatformHandler const &Plt, Args &&...args);
};

} // namespace gEng
