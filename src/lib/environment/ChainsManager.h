#pragma once

#include "Builder.hpp"
#include "detail/Types.hpp"
#include "platform_handler.h"

namespace gEng {

class Window;

class ChainsManager final {
  PlatformHandler const *PltMgr{nullptr};
  ChainsBuilder B;

  vk::SwapchainKHR Swapchain{};
  detail::Swapchains SCs{};

public:
  ChainsManager() = default;

  void init(PlatformHandler const &PltIn, Window const &W) {
    PltMgr = &PltIn;
    Swapchain = B.create<vk::SwapchainKHR>(*PltMgr, W);
    SCs = B.create<detail::Swapchains>(*PltMgr, Swapchain);
  }
  // Believe this getters are temporary.
  vk::SwapchainKHR &getSwapchain() { return Swapchain; }
  vk::Format &getFormat() { return B.Fmt; }
  std::vector<vk::Image> &getImages() { return SCs.Img; }
  std::vector<vk::ImageView> &getImageViews() { return SCs.ImgView; }
  vk::Extent2D &getExtent() { return B.Ext; }
};

} // namespace gEng
