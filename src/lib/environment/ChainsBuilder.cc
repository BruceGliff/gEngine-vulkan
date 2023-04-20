#include "Builder.hpp"

#include <algorithm>

#include "detail/CommonForPltAndChains.hpp"
#include "gEng/window.h"
#include "platform_handler.h"
#include "shader/shader.h"
#include "vertex.h"
// TODO make SysEnv global singleton
#include "gEng/environment.h"

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
vk::SampleCountFlagBits
ChainsBuilder::create<vk::SampleCountFlagBits>(PlatformHandler const &PltMgr) {
  using SC = vk::SampleCountFlagBits;
  auto PhysDev = PltMgr.get<vk::PhysicalDevice>();
  vk::PhysicalDeviceProperties DevProps = PhysDev.getProperties();

  vk::SampleCountFlags Counts = DevProps.limits.framebufferColorSampleCounts &
                                DevProps.limits.framebufferDepthSampleCounts;
  SC FlagBits = SC::e1;

  if (Counts & SC::e64)
    FlagBits = SC::e64;
  else if (Counts & SC::e32)
    FlagBits = SC::e32;
  else if (Counts & SC::e16)
    FlagBits = SC::e16;
  else if (Counts & SC::e8)
    FlagBits = SC::e8;
  else if (Counts & SC::e4)
    FlagBits = SC::e4;
  else if (Counts & SC::e2)
    FlagBits = SC::e2;

  return FlagBits;
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

// Takes a lists of candidate formats from most desireable to the least
// desirable and checks the first one is supported.
static vk::Format findSupportedFormat(vk::PhysicalDevice const &PhysDev,
                                      std::vector<vk::Format> const &Candidates,
                                      vk::ImageTiling Tiling,
                                      vk::FormatFeatureFlags Feats) {
  for (vk::Format const &Format : Candidates) {
    vk::FormatProperties Props = PhysDev.getFormatProperties(Format);
    if (Tiling == vk::ImageTiling::eLinear &&
        (Props.linearTilingFeatures & Feats) == Feats)
      return Format;
    else if (Tiling == vk::ImageTiling::eOptimal &&
             (Props.optimalTilingFeatures & Feats) == Feats)
      return Format;
  }

  throw std::runtime_error("failed to find supported format!");
}
static vk::Format findDepthFormat(vk::PhysicalDevice const &PhysDev) {
  using F = vk::Format;
  return findSupportedFormat(
      PhysDev, {F::eD32Sfloat, F::eD32SfloatS8Uint, F::eD24UnormS8Uint},
      vk::ImageTiling::eOptimal,
      vk::FormatFeatureFlagBits::eDepthStencilAttachment);
}
template <>
vk::RenderPass
ChainsBuilder::create<vk::RenderPass>(PlatformHandler const &PltMgr,
                                      vk::SampleCountFlagBits &msaa) {
  auto PhysDev = PltMgr.get<vk::PhysicalDevice>();
  vk::AttachmentDescription DepthAtt{
      {},
      findDepthFormat(PhysDev),
      msaa,
      vk::AttachmentLoadOp::eClear,
      vk::AttachmentStoreOp::eDontCare,
      vk::AttachmentLoadOp::eDontCare,
      vk::AttachmentStoreOp::eDontCare,
      vk::ImageLayout::eUndefined,
      vk::ImageLayout::eDepthStencilAttachmentOptimal};
  vk::AttachmentReference DepthAttRef{
      1, vk::ImageLayout::eDepthStencilAttachmentOptimal};

  vk::AttachmentDescription ColorAtt{{},
                                     Fmt,
                                     msaa,
                                     vk::AttachmentLoadOp::eClear,
                                     vk::AttachmentStoreOp::eStore,
                                     vk::AttachmentLoadOp::eDontCare,
                                     vk::AttachmentStoreOp::eDontCare,
                                     vk::ImageLayout::eUndefined,
                                     vk::ImageLayout::eColorAttachmentOptimal};
  vk::AttachmentReference ColorAttRef{0,
                                      vk::ImageLayout::eColorAttachmentOptimal};

  vk::AttachmentDescription ColorAttResolve{{},
                                            Fmt,
                                            vk::SampleCountFlagBits::e1,
                                            vk::AttachmentLoadOp::eDontCare,
                                            vk::AttachmentStoreOp::eStore,
                                            vk::AttachmentLoadOp::eDontCare,
                                            vk::AttachmentStoreOp::eDontCare,
                                            vk::ImageLayout::eUndefined,
                                            vk::ImageLayout::ePresentSrcKHR};
  vk::AttachmentReference ColorAttResolveRef{
      2, vk::ImageLayout::eColorAttachmentOptimal};

  // TODO: empty input attachments(3rd operand).
  vk::SubpassDescription Subpass{{},
                                 vk::PipelineBindPoint::eGraphics,
                                 {},
                                 ColorAttRef,
                                 ColorAttResolveRef,
                                 &DepthAttRef};

  using Fbits = vk::PipelineStageFlagBits;
  vk::SubpassDependency Dependency{
      VK_SUBPASS_EXTERNAL, 0,
      Fbits::eColorAttachmentOutput | Fbits::eEarlyFragmentTests,
      Fbits::eColorAttachmentOutput | Fbits::eEarlyFragmentTests,
      vk::AccessFlagBits::eColorAttachmentWrite |
          vk::AccessFlagBits::eDepthStencilAttachmentWrite};

  std::array<vk::AttachmentDescription, 3> Attachments{ColorAtt, DepthAtt,
                                                       ColorAttResolve};
  return PltMgr.get<vk::Device>().createRenderPass(
      {{}, Attachments, Subpass, Dependency});
}

template <>
vk::DescriptorSetLayout
ChainsBuilder::create<vk::DescriptorSetLayout>(PlatformHandler const &PltMgr) {
  auto Dev = PltMgr.get<vk::Device>();
  // TODO samplers are null, but descriptorCount=1!.
  // Even though there are no pImmutableSamplers in both LB, descriptorCount
  // has to be at least 1. TODO: find out why.
  vk::DescriptorSetLayoutBinding LayoutBindingUBO{
      0, vk::DescriptorType::eUniformBuffer, 1,
      vk::ShaderStageFlagBits::eVertex};
  vk::DescriptorSetLayoutBinding LayoutBindingSampler{
      1, vk::DescriptorType::eCombinedImageSampler, 1,
      vk::ShaderStageFlagBits::eFragment};

  std::array<vk::DescriptorSetLayoutBinding, 2> Bindings = {
      LayoutBindingUBO, LayoutBindingSampler};

  return Dev.createDescriptorSetLayout({{}, Bindings});
}

template <>
vk::PipelineLayout
ChainsBuilder::create<vk::PipelineLayout>(PlatformHandler const &PltMgr,
                                          vk::DescriptorSetLayout &DSL) {
  auto Dev = PltMgr.get<vk::Device>();
  return Dev.createPipelineLayout({{}, DSL});
}

template <>
vk::Pipeline ChainsBuilder::create<vk::Pipeline>(PlatformHandler const &PltMgr,
                                                 vk::SampleCountFlagBits &MSAA,
                                                 vk::PipelineLayout &PPL,
                                                 vk::RenderPass &RPass) {
  auto Dev = PltMgr.get<vk::Device>();
  auto const &EH = SysEnv::getInstance();

  fs::path ShadersPath{EH};
  ShadersPath /= "shaders/";
  Shader VShader{(ShadersPath / "basic.vert.spv").string()};
  Shader FShader{(ShadersPath / "basic.frag.spv").string()};

  vk::ShaderModule VShaderModule =
      Dev.createShaderModule({{}, VShader.getSPIRV()});
  vk::ShaderModule FShaderModule =
      Dev.createShaderModule({{}, FShader.getSPIRV()});

  vk::PipelineShaderStageCreateInfo VSInfo{
      {}, vk::ShaderStageFlagBits::eVertex, VShaderModule, "main"};
  vk::PipelineShaderStageCreateInfo FSInfo{
      {}, vk::ShaderStageFlagBits::eFragment, FShaderModule, "main"};

  std::array<vk::PipelineShaderStageCreateInfo, 2> ShaderStages{VSInfo, FSInfo};

  // Fill Vertex binding info.
  auto bindDescr = Vertex::getBindDescription();
  auto attrDescr = Vertex::getAttrDescription();

  vk::PipelineVertexInputStateCreateInfo VInputInfo{{}, bindDescr, attrDescr};

  // The rules how verticies will be treated(lines, points, triangles..)
  vk::PipelineInputAssemblyStateCreateInfo InputAssembly{
      {}, vk::PrimitiveTopology::eTriangleList, VK_FALSE};

  // Read this in FixedFunction part.
  vk::Viewport Viewport{0.0f, 0.0f, (float)Ext.width, (float)Ext.height,
                        0.0f, 1.0f};
  vk::Rect2D Scissor{{0, 0}, Ext};

  vk::PipelineViewportStateCreateInfo ViewportState{{}, Viewport, Scissor};

  vk::PipelineRasterizationStateCreateInfo Rast{
      {},
      VK_FALSE,
      VK_FALSE,
      vk::PolygonMode::eFill,
      vk::CullModeFlagBits::eBack,
      vk::FrontFace::eCounterClockwise};
  Rast.setLineWidth(1.f);

  vk::PipelineMultisampleStateCreateInfo Multisampling{
      {}, MSAA, VK_TRUE, .2f, nullptr};

  using CC = vk::ColorComponentFlagBits;
  vk::PipelineColorBlendAttachmentState ColorBlendAttachment{
      VK_FALSE,
      vk::BlendFactor::eOne,
      vk::BlendFactor::eZero,
      vk::BlendOp::eAdd,
      vk::BlendFactor::eOne,
      vk::BlendFactor::eZero,
      vk::BlendOp::eAdd,
      CC::eR | CC::eG | CC::eB | CC::eA};

  vk::PipelineColorBlendStateCreateInfo ColorBlending{
      {}, VK_FALSE, vk::LogicOp::eCopy, ColorBlendAttachment};

  // There are some states of pipeline that can be changed dynamicly.
  std::array<vk::DynamicState, 2> DynStates{vk::DynamicState::eViewport,
                                            vk::DynamicState::eLineWidth};

  vk::PipelineDynamicStateCreateInfo DynState{{}, DynStates};

  vk::PipelineDepthStencilStateCreateInfo DepthStencil{
      {}, VK_TRUE, VK_TRUE, vk::CompareOp::eLess, VK_FALSE, VK_FALSE, {},
      {}, 0.f,     1.f};

  vk::GraphicsPipelineCreateInfo PipelineInfo{{},
                                              ShaderStages,
                                              &VInputInfo,
                                              &InputAssembly,
                                              {},
                                              &ViewportState,
                                              &Rast,
                                              &Multisampling,
                                              &DepthStencil,
                                              &ColorBlending,
                                              {},
                                              PPL,
                                              RPass,
                                              {},
                                              {},
                                              -1};

  // It creates several pipelines.
  auto P =
      Dev.createGraphicsPipelines(VK_NULL_HANDLE, PipelineInfo).value.at(0);

  Dev.destroyShaderModule(FShaderModule);
  Dev.destroyShaderModule(VShaderModule);
  return P;
}
