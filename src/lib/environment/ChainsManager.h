#pragma once

#include "Builder.hpp"
#include "platform_handler.h"

namespace gEng {

class ChainsManager final {
  PlatformHandler const *PltMgr{nullptr};
  ChainsBuilder B;

  vk::SwapchainKHR Swapchain{};
  std::vector<vk::Image> SwapchainImages{};

public:
  ChainsManager() = default;

  void init(PlatformHandler const &PltIn) {
    PltMgr = &PltIn;
    Swapchain = B.create<vk::SwapchainKHR>(*PltMgr);
    SwapchainImages = B.create<std::vector<vk::Image>>(*PltMgr, Swapchain);
  }
  // Believe this getters are temporary.
  vk::SwapchainKHR &getSwapchain() { return Swapchain; }
  std::vector<vk::Image> &getImages() { return SwapchainImages; }
};

} // namespace gEng
