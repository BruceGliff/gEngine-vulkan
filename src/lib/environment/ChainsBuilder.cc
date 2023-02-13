#include "Builder.hpp"

#include "detail/CommonForPltAndChains.hpp"
#include "platform_handler.h"

using namespace gEng;

template <> vk::SwapchainKHR create(PlatformHandler &PltMgr) {
  auto PhysDev = PltMgr.get<vk::PhysicalDevice>();

  SwapchainSupportDetails SwapchainSupport =
      querySwapchainSupport(m_physicalDevice);

  vk::SurfaceFormatKHR SurfFmt =
      chooseSwapSurfaceFormat(SwapchainSupport.formats);
  vk::PresentModeKHR PresentMode =
      chooseSwapPresentMode(SwapchainSupport.presentModes);
  m_swapchainExtent = chooseSwapExtent(SwapchainSupport.capabilities);

  // min images count in swap chain(plus one).
  uint32_t ImageCount = SwapchainSupport.capabilities.minImageCount + 1;
  // Not to go throught max ImageCount (0 means no upper-bounds).
  if (SwapchainSupport.capabilities.maxImageCount > 0 &&
      ImageCount > SwapchainSupport.capabilities.maxImageCount)
    ImageCount = SwapchainSupport.capabilities.maxImageCount;

  QueueFamilyIndices Indices = findQueueFamilies(m_physicalDevice);
  bool const IsFamiliesSame = Indices.GraphicsFamily == Indices.PresentFamily;
  // Next, we need to specify how to handle swap chain images that will be
  // used across multiple queue families.
  std::vector<uint32_t> FamilyIndices =
      !IsFamiliesSame ? std::vector<uint32_t>{Indices.GraphicsFamily.value(),
                                              Indices.PresentFamily.value()}
                      : std::vector<uint32_t>{};
  vk::SharingMode SMode = !IsFamiliesSame ? vk::SharingMode::eConcurrent
                                          : vk::SharingMode::eExclusive;

  m_swapchainImageFormat = SurfFmt.format;
  vk::SwapchainCreateInfoKHR const CreateInfo{
      {},
      m_surface,
      ImageCount,
      m_swapchainImageFormat,
      SurfFmt.colorSpace,
      m_swapchainExtent,
      1,
      vk::ImageUsageFlagBits::eColorAttachment,
      SMode,
      FamilyIndices,
      SwapchainSupport.capabilities.currentTransform,
      vk::CompositeAlphaFlagBitsKHR::eOpaque,
      PresentMode,
      VK_TRUE};

  m_swapchain = m_device.createSwapchainKHR(CreateInfo);
  m_swapchainImages = m_device.getSwapchainImagesKHR(m_swapchain);
}
