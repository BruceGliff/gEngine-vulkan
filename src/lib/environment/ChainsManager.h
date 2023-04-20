#pragma once

#include "Builder.hpp"
#include "detail/Types.hpp"
#include "platform_handler.h"

#include "../image/ImageBuilder.hpp"

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

  ImageBuilder::Type IM{};
  vk::ImageView IView{};

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

    ImageBuilder IB{PltMgr.get<vk::Device>(), PltMgr.get<vk::PhysicalDevice>()};
    IM = IB.create<ImageBuilder::Type>(
        B.Ext.width, B.Ext.height, 1u, MSAA, B.Fmt, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransientAttachment |
            vk::ImageUsageFlagBits::eColorAttachment,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    IView = IB.create<vk::ImageView>(std::get<0>(IM), B.Fmt,
                                     vk::ImageAspectFlagBits::eColor, 1u);

    // createColorResources();
    // createDepthResources();
    // createFramebuffers();
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
  auto &getColorRes() { return IM; }
  auto &getColorIView() { return IView; }
};

} // namespace gEng
