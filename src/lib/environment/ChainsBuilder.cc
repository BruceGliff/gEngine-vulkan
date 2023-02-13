#include "Builder.hpp"

#include "detail/CommonForPltAndChains.hpp"
#include "platform_handler.h"

using namespace gEng;

template <> vk::SwapchainKHR create(PlatformHandler &PltMgr) {
  auto PhysDev = PltMgr.get<vk::PhysicalDevice>();
  auto Dev = PltMgr.get<vk::Device>();
  auto Surface = PltMgr.get<vk::SurfaceKHR>();

  SwapchainSupportDetails SwapchainSupport = querySwapchainSupport(PhysDev);

  vk::SurfaceFormatKHR SurfFmt =
      chooseSwapSurfaceFormat(SwapchainSupport.formats);
  vk::PresentModeKHR PresentMode =
      chooseSwapPresentMode(SwapchainSupport.presentModes);
  vk::Extent2D Extent = chooseSwapExtent(SwapchainSupport.capabilities);

  // min images count in swap chain(plus one).
  uint32_t ImageCount = SwapchainSupport.capabilities.minImageCount + 1;
  // Not to go throught max ImageCount (0 means no upper-bounds).
  if (SwapchainSupport.capabilities.maxImageCount > 0 &&
      ImageCount > SwapchainSupport.capabilities.maxImageCount)
    ImageCount = SwapchainSupport.capabilities.maxImageCount;

  QueueFamilyIndices Indices = findQueueFamilies(PhysDev);
  bool const IsFamiliesSame = Indices.GraphicsFamily == Indices.PresentFamily;
  // Next, we need to specify how to handle swap chain images that will be
  // used across multiple queue families.
  std::vector<uint32_t> FamilyIndices =
      !IsFamiliesSame ? std::vector<uint32_t>{Indices.GraphicsFamily.value(),
                                              Indices.PresentFamily.value()}
                      : std::vector<uint32_t>{};
  vk::SharingMode SMode = !IsFamiliesSame ? vk::SharingMode::eConcurrent
                                          : vk::SharingMode::eExclusive;

  vk::SwapchainCreateInfoKHR const CreateInfo{
      {},
      Surface,
      ImageCount,
      SurfFmt.format,
      SurfFmt.colorSpace,
      Extent,
      1,
      vk::ImageUsageFlagBits::eColorAttachment,
      SMode,
      FamilyIndices,
      SwapchainSupport.capabilities.currentTransform,
      vk::CompositeAlphaFlagBitsKHR::eOpaque,
      PresentMode,
      VK_TRUE};

  return Dev.createSwapchainKHR(CreateInfo);
}

template <>
Swapchains create(PlatformHandler &PltMgr, vk::SwapchainKHR &Swapchain) {

  m_swapchainImages = m_device.getSwapchainImagesKHR(m_swapchain);
}
