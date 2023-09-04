#pragma once

#include "Builder.hpp"
#include "detail/Types.hpp"
#include "platform_handler.h"

#include "../image/ImageBuilder.hpp"
#include <vulkan/vulkan_handles.hpp>

namespace gEng {

class Window;

class ChainsManager final {
  // Maybe move to constructor
  ChainsBuilder B;

  vk::SwapchainKHR Swapchain{};
  detail::Swapchains SCs{};
  vk::RenderPass RPass{};
  vk::SampleCountFlagBits MSAA{};
  vk::DescriptorSetLayout DescSet{};
  vk::DescriptorPool DescPool{};
  std::vector<vk::DescriptorSet> DSet{};
  vk::PipelineLayout PPL{};
  vk::Pipeline P{};

  ImageBuilder::Type IM{};
  vk::ImageView IView{};
  ImageBuilder::Type DP{};
  vk::ImageView DView{};

  ChainsBuilder::FrameBuffers FrameBuffers;

public:
  ChainsManager() = default;
  ChainsManager(PlatformHandler const &PltMgr, Window const &W) {
    MSAA = B.create<vk::SampleCountFlagBits>(PltMgr);
    Swapchain = B.create<vk::SwapchainKHR>(PltMgr, W);
    SCs = B.create<detail::Swapchains>(PltMgr, Swapchain);
    RPass = B.create<vk::RenderPass>(PltMgr, MSAA);
    DescSet = B.create<vk::DescriptorSetLayout>(PltMgr);
    DescPool = B.create<vk::DescriptorPool>(PltMgr);
    DSet = B.create<std::vector<vk::DescriptorSet>>(PltMgr, DescPool, DescSet);
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

    auto DepthFmt = B.findDepthFmt(PltMgr.get<vk::PhysicalDevice>());
    DP = IB.create<ImageBuilder::Type>(
        B.Ext.width, B.Ext.height, 1u, MSAA, DepthFmt,
        vk::ImageTiling::eOptimal,
        vk::Flags(vk::ImageUsageFlagBits::eDepthStencilAttachment),
        vk::MemoryPropertyFlagBits::eDeviceLocal);
    DView = IB.create<vk::ImageView>(std::get<0>(DP), DepthFmt,
                                     vk::ImageAspectFlagBits::eDepth, 1u);
    // As I understand this part is optional as we will take care of this in the
    // render pass.
    IB.transitionImageLayout(
        PltMgr, std::get<0>(DP), DepthFmt, vk::ImageLayout::eUndefined,
        vk::ImageLayout::eDepthStencilAttachmentOptimal, 1);

    FrameBuffers = B.create<ChainsBuilder::FrameBuffers>(PltMgr, IView, DView,
                                                         SCs.ImgView, RPass);
  }
  ChainsManager(ChainsManager &&) = default;
  ChainsManager &operator=(ChainsManager &&) = default;

  // Believe this getters are temporary.
  vk::SwapchainKHR &getSwapchain() { return Swapchain; }
  vk::Format &getFormat() { return B.Fmt; }
  std::vector<vk::Image> &getImages() { return SCs.Img; }
  std::vector<vk::ImageView> &getImageViews() { return SCs.ImgView; }
  vk::Extent2D &getExtent() { return B.Ext; }
  vk::RenderPass &getRPass() { return RPass; };
  auto &getDSL() { return DescSet; }
  auto &getDPool() { return DescPool; }
  auto &getDSet() { return DSet; }

  auto &getPPL() { return PPL; }
  auto &getP() { return P; }
  auto &getFrameBuffers() { return FrameBuffers; }

  void cleanup(vk::Device Dev) {
    Dev.destroyImageView(IView);
    Dev.destroyImage(std::get<vk::Image>(IM));
    Dev.freeMemory(std::get<vk::DeviceMemory>(IM));

    Dev.destroyImageView(DView);
    Dev.destroyImage(std::get<vk::Image>(DP));
    Dev.freeMemory(std::get<vk::DeviceMemory>(DP));

    for (auto &&SwapchainBuff : FrameBuffers)
      Dev.destroyFramebuffer(SwapchainBuff);

    Dev.destroyPipeline(P);
    Dev.destroyPipelineLayout(PPL);
    Dev.destroyRenderPass(RPass);

    // FIXME SCs.Img?
    for (auto &&ImgView : SCs.ImgView)
      Dev.destroyImageView(ImgView);

    Dev.destroySwapchainKHR(Swapchain);

    Dev.destroyDescriptorPool(DescPool);
    Dev.destroyDescriptorSetLayout(DescSet);
  }
};

} // namespace gEng
