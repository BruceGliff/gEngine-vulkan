#include "Builder.hpp"

#include <algorithm>

#include "detail/CommonForPltAndChains.hpp"
#include "gEng/window.h"
#include "platform_handler.h"

using namespace gEng;

// Presentation mode represents the actual conditions for showing images to
// the screen.
static vk::PresentModeKHR chooseSwapPresentMode(
    std::vector<vk::PresentModeKHR> const &AvailablePresentModes) {
  auto FindIt =
      std::find_if(AvailablePresentModes.begin(), AvailablePresentModes.end(),
                   [](auto &&PresentMode) {
                     return PresentMode == vk::PresentModeKHR::eMailbox;
                   });
  if (FindIt != AvailablePresentModes.end())
    return *FindIt;

  return vk::PresentModeKHR::eFifo;
}

static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(
    std::vector<vk::SurfaceFormatKHR> const &AvailableFormats) {
  // Some words in Swap chain part:
  // https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Swap_chain
  auto FindIt = std::find_if(
      AvailableFormats.begin(), AvailableFormats.end(), [](auto &&Format) {
        return Format.format == vk::Format::eB8G8R8A8Srgb &&
               Format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
      });
  if (FindIt != AvailableFormats.end())
    return *FindIt;

  std::cout << "The best format unavailable.\nUse available one.\n";
  return AvailableFormats[0];
}

// glfwWindows works with screen-coordinates. But Vulkan - with pixels.
// And not always they are corresponsible with each other.
// So we want to create a proper resolution.
static vk::Extent2D
chooseSwapExtent(vk::SurfaceCapabilitiesKHR const &Capabilities,
                 Window const &Window) {
  if (Capabilities.currentExtent.width != UINT32_MAX)
    return Capabilities.currentExtent;
  else {
    // TODO. maybe updExtent
    auto [Width, Height] = Window.getExtent();
    uint32_t const RealW = std::clamp(Width, Capabilities.minImageExtent.width,
                                      Capabilities.maxImageExtent.width);
    uint32_t const RealH =
        std::clamp(Height, Capabilities.minImageExtent.height,
                   Capabilities.maxImageExtent.height);
    return {RealW, RealH};
  }
}

template <>
vk::SwapchainKHR
ChainsBuilder::create<vk::SwapchainKHR>(PlatformHandler const &PltMgr,
                                        Window const &W) {
  auto PhysDev = PltMgr.get<vk::PhysicalDevice>();
  auto Dev = PltMgr.get<vk::Device>();
  auto Surface = PltMgr.get<vk::SurfaceKHR>();

  auto SwapchainSupport = detail::querySwapchainSupport(Surface, PhysDev);

  vk::SurfaceFormatKHR SurfFmt =
      chooseSwapSurfaceFormat(SwapchainSupport.formats);
  vk::PresentModeKHR PresentMode =
      chooseSwapPresentMode(SwapchainSupport.presentModes);
  Ext = chooseSwapExtent(SwapchainSupport.capabilities, W);

  // min images count in swap chain(plus one).
  uint32_t ImageCount = SwapchainSupport.capabilities.minImageCount + 1;
  // Not to go throught max ImageCount (0 means no upper-bounds).
  if (SwapchainSupport.capabilities.maxImageCount > 0 &&
      ImageCount > SwapchainSupport.capabilities.maxImageCount)
    ImageCount = SwapchainSupport.capabilities.maxImageCount;

  auto Indices = detail::findQueueFamilies(Surface, PhysDev);
  bool const AreFamiliesSame = Indices.GraphicsFamily == Indices.PresentFamily;
  // Next, we need to specify how to handle swap chain images that will be
  // used across multiple queue families.
  std::vector<uint32_t> FamilyIndices =
      !AreFamiliesSame ? std::vector<uint32_t>{Indices.GraphicsFamily.value(),
                                               Indices.PresentFamily.value()}
                       : std::vector<uint32_t>{};
  vk::SharingMode SMode = !AreFamiliesSame ? vk::SharingMode::eConcurrent
                                           : vk::SharingMode::eExclusive;

  Fmt = SurfFmt.format;
  vk::SwapchainCreateInfoKHR const CreateInfo{
      {},
      Surface,
      ImageCount,
      Fmt,
      SurfFmt.colorSpace,
      Ext,
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
detail::Swapchains
ChainsBuilder::create<detail::Swapchains>(PlatformHandler const &PltMgr,
                                          vk::SwapchainKHR &Swapchain) {
  auto Dev = PltMgr.get<vk::Device>();
  std::vector<vk::Image> Imgs = Dev.getSwapchainImagesKHR(Swapchain);

  std::vector<vk::ImageView> ImgViews;
  std::transform(Imgs.begin(), Imgs.end(), std::back_inserter(ImgViews),
                 [this, &Dev](vk::Image const &Img) {
                   return Dev.createImageView(
                       {{},
                        Img,
                        vk::ImageViewType::e2D,
                        Fmt,
                        {},
                        {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}});
                 });
  return detail::Swapchains{Imgs, ImgViews};
}
