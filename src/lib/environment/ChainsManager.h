#pragma once

#include "Builder.hpp"
#include "detail/Types.hpp"
#include "platform_handler.h"

namespace gEng {

class Window;

class ChainsManager final {
  // Maybe move to init
  ChainsBuilder B;

  vk::SwapchainKHR Swapchain{};
  detail::Swapchains SCs{};
  vk::RenderPass RPass{};
  vk::SampleCountFlagBits MSAA{};
  vk::DescriptorSetLayout DescSet{};
  vk::PipelineLayout PPL{};
  vk::Pipeline P{};

public:
  ChainsManager() = default;

  // TODO create constructor and remove ManagerBehavior
  void init(PlatformHandler const &PltMgr, Window const &W) {
    MSAA = B.create<vk::SampleCountFlagBits>(PltMgr);
    Swapchain = B.create<vk::SwapchainKHR>(PltMgr, W);
    SCs = B.create<detail::Swapchains>(PltMgr, Swapchain);
    RPass = B.create<vk::RenderPass>(PltMgr, MSAA);
    DescSet = B.create<vk::DescriptorSetLayout>(PltMgr);
    PPL = B.create<vk::PipelineLayout>(PltMgr, DescSet);
    P = B.create<vk::Pipeline>(PltMgr, MSAA, PPL, RPass);
  }
  // Believe this getters are temporary.
  vk::SwapchainKHR &getSwapchain() { return Swapchain; }
  vk::Format &getFormat() { return B.Fmt; }
  std::vector<vk::Image> &getImages() { return SCs.Img; }
  std::vector<vk::ImageView> &getImageViews() { return SCs.ImgView; }
  vk::Extent2D &getExtent() { return B.Ext; }
  vk::RenderPass &getRPass() { return RPass; };
  auto &getMSAA() { return MSAA; }
  auto &getDSL() { return DescSet; }
  auto &getPPL() { return PPL; }
  auto &getP() { return P; }
};

} // namespace gEng
