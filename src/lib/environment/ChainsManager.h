#pragma once

#include "Builder.hpp"
#include "platform_handler.h"

namespace gEng {

struct Swapchains final {
  std::vector<vk::Image> Img{};
  std::vector<vk::ImageView> ImgView{};
};

class ChainsManager final {
  PlatformHandler const *PltMgr{nullptr};
  ChainsBuilder B;

  vk::SwapchainKHR Swapchain{};
  std::vector<vk::Image> SwapchainImages{};
  std::vector<vk::ImageView> SwapchainImageViews{};
  Swapchains SCs{};

public:
  ChainsManager() = default;

  void init(PlatformHandler const &PltIn) {
    PltMgr = &PltIn;
    Swapchain = B.create<vk::SwapchainKHR>(*PltMgr);
    SCs = B.create<Swapchains>(*PltMgr, Swapchain);
  }
  // Believe this getters are temporary.
  vk::SwapchainKHR &getSwapchain() { return Swapchain; }
  std::vector<vk::Image> &getImages() { return SCs.Img; }
  std::vector<vk::ImageView> &getImages() { return SCs.ImgView; }
};

} // namespace gEng
