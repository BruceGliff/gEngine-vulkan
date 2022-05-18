// To use designated initializers.
// #define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.hpp>

#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include "EnvHandler.h"
#include "decoy/decoy.h"
#include "image/image.h"
#include "shader/shader.h"
#include "vertex.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// This is some hack for a callback handling.
// VkDebugUtilsMessengerCreateInfoEXT struct should be passed to
// vkCreateDebugUtilsMessengerEXT, but as this function is an extension, it is
// not automatically loaded. So we have to look up by ourselfes via
// vkGetInstanceProcAddr.
// TODO think about C++ style.
VkResult createDebugUtilsMessengerEXT(
    vk::Instance const &Instance,
    VkDebugUtilsMessengerCreateInfoEXT const *pCreateInfo,
    VkAllocationCallbacks const *pAllocator,
    VkDebugUtilsMessengerEXT *pDebugMessenger) {
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      Instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr)
    return func(Instance, pCreateInfo, pAllocator, pDebugMessenger);
  else
    return VK_ERROR_EXTENSION_NOT_PRESENT;
}
// Some simillar hack.
void destroyDebugUtilsMessengerEXT(VkInstance instance,
                                   VkDebugUtilsMessengerEXT debugMessenger,
                                   const VkAllocationCallbacks *pAllocator) {
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func)
    func(instance, debugMessenger, pAllocator);
  else
    std::cout << "ERR: debug is not destroyed!\n";
}

struct UniformBufferObject {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
};

class HelloTriangleApplication {
  EnvHandler &EH;
  GLFWwindow *m_Window{};

  unsigned const m_Width{1600};
  unsigned const m_Height{900};

  // Hom many frames can be processed concurrently.
  const int MAX_FRAMES_IN_FLIGHT = 2;

  // TODO of course get rid of global code!.
  // This vector is responsible for listing requires validation layers for
  // debug.
  std::vector<char const *> const m_validationLayers = {
      "VK_LAYER_KHRONOS_validation"};
  // This vector is responsible for listing requires device extensions.
  // presentQueue by itself requires swapchain, so this is for
  // explicit check.
  std::vector<char const *> const m_deviceExtensions = {
      VK_KHR_SWAPCHAIN_EXTENSION_NAME};
#ifdef NDEBUG
  bool const m_enableValidationLayers = false;
#else
  bool const m_enableValidationLayers = true;
#endif

  // An instance needed for connection between app and VkLibrary
  // And adds a detailes about app to the driver
  vk::Instance m_instance;
  // Member for a call back handling.
  VkDebugUtilsMessengerEXT m_debugMessenger{};
  // Preferable device. Will be freed automatically.
  vk::PhysicalDevice m_physicalDevice;
  // Logical device.
  vk::Device m_device;
  // Queue automatically created with logical device, but we need to create a
  // handles. And queues automatically destroyed within device.
  vk::Queue m_graphicsQueue;
  // Presentation queue.
  vk::Queue m_presentQueue;
  // Surface to be rendered in.
  // It is actually platform-dependent, but glfw uses function which fills
  // platform-specific structures by itself.
  VkSurfaceKHR m_surface{};
  // swapchain;
  VkSwapchainKHR m_swapchain{};

  std::vector<vk::Image> m_swapchainImages;
  vk::Format m_swapchainImageFormat;
  vk::Extent2D m_swapchainExtent;
  // ImageView used to specify how to treat VkImage.
  // For each VkImage we create VkImageView.
  std::vector<vk::ImageView> m_swapchainImageViews;

  VkRenderPass m_renderPass;
  // Used for an uniform variable.
  VkPipelineLayout m_pipelineLayout;

  vk::Pipeline m_graphicsPipeline;

  // A framebuffer object references all of the VkImageView objects.
  std::vector<vk::Framebuffer> m_swapChainFramebuffers;

  VkCommandPool m_commandPool;

  std::vector<VkCommandBuffer> m_commandBuffers;

  std::vector<VkSemaphore> m_imageAvailableSemaphore;
  std::vector<VkSemaphore> m_renderFinishedSemaphore;

  std::vector<VkFence> m_inFlightFence;

  bool framebufferResized = false;

  std::vector<Vertex> Vertices;
  std::vector<uint32_t> Indices;

  VkBuffer VertexBuffer;
  VkDeviceMemory VertexBufferMemory;

  VkBuffer IndexBuffer;
  VkDeviceMemory IndexBufferMemory;

  std::vector<VkBuffer> uniformBuffers;
  std::vector<VkDeviceMemory> uniformBuffersMemory;

  vk::DescriptorSetLayout descriptorSetLayout;

  VkDescriptorPool descriptorPool;
  std::vector<vk::DescriptorSet> descriptorSets;

  uint32_t mipLevels;
  VkImage textureImage;
  VkDeviceMemory textureImageMemory;

  VkImageView textureImageView;
  VkSampler textureSampler;

  VkImage depthImage;
  VkDeviceMemory depthImageMemory;
  VkImageView depthImageView;

  // For msaa.
  vk::SampleCountFlagBits msaaSamples = vk::SampleCountFlagBits::e1;
  VkImage colorImage;
  VkDeviceMemory colorImageMemory;
  VkImageView colorImageView;

public:
  HelloTriangleApplication(EnvHandler &InEH) : EH{InEH} {}
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

private:
  void initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    // TODO Resize does not work properly.
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);
    m_Window = glfwCreateWindow(m_Width, m_Height, EH.getFilenameStr().c_str(),
                                nullptr, nullptr);
    assert(m_Window && "Window initializating falis!");

    glfwSetWindowUserPointer(m_Window, this);
    glfwSetFramebufferSizeCallback(m_Window, framebufferResizeCallback);
  }

  static void framebufferResizeCallback(GLFWwindow *window, int width,
                                        int height) {
    auto app = reinterpret_cast<HelloTriangleApplication *>(
        glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
  }

  void initVulkan() {
    createInstance();
    setupDebugMessenger();
    // It has to be placed here, because we need already created Instance
    // and picking PhysicalDevice can rely on Surface attributes.
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapchain();
    createImageViews();
    createRenderPass();
    createDescriptorSetLayout();
    createGraphicPipeline();
    createCommandPool();
    createColorResources();
    createDepthResources();
    createFramebuffers();
    createTextureImage();
    createTextureImageView();
    createTextureSampler();
    loadModel();
    createVertexBuffer();
    createIndexBuffer();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
    createSyncObjects();
  }

  void createColorResources() {
    vk::Format ColorFmt = m_swapchainImageFormat;

    std::tie(colorImage, colorImageMemory) =
        createImage(m_swapchainExtent.width, m_swapchainExtent.height, 1,
                    msaaSamples, ColorFmt, vk::ImageTiling::eOptimal,
                    vk::ImageUsageFlagBits::eTransientAttachment |
                        vk::ImageUsageFlagBits::eColorAttachment,
                    vk::MemoryPropertyFlagBits::eDeviceLocal);
    colorImageView = createImageView(colorImage, ColorFmt,
                                     vk::ImageAspectFlagBits::eColor, 1);
  }

  vk::SampleCountFlagBits getMaxUsableSampleCount() {
    using SC = vk::SampleCountFlagBits;
    vk::PhysicalDeviceProperties DevProps = m_physicalDevice.getProperties();

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

    std::cout << "Samples: " << static_cast<uint32_t>(FlagBits) << '\n';
    return FlagBits;
  }

  void loadModel() {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    fs::path ModelPath{EH};
    ModelPath /= "assets/models/viking_room.obj";
    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                          ModelPath.generic_string().c_str()))
      throw std::runtime_error(warn + err);

    // To avoid using Vertex duplications.
    std::unordered_map<Vertex, uint32_t> uniqueVertices{};
    for (auto const &shape : shapes)
      for (auto const &index : shape.mesh.indices) {
        Vertex vertex{};
        vertex.Position = {attrib.vertices[3 * index.vertex_index + 0],
                           attrib.vertices[3 * index.vertex_index + 1],
                           attrib.vertices[3 * index.vertex_index + 2]};
        vertex.TexCoord = {attrib.texcoords[2 * index.texcoord_index + 0],
                           1.0f -
                               attrib.texcoords[2 * index.texcoord_index + 1]};
        vertex.Color = {1.0f, 1.0f, 1.0f};

        if (uniqueVertices.count(vertex) == 0) {
          uniqueVertices[vertex] = static_cast<uint32_t>(Vertices.size());
          Vertices.push_back(vertex);
        }
        Indices.push_back(uniqueVertices[vertex]);
      }
  }

  void createDepthResources() {
    vk::Format DepthFmt = findDepthFormat();

    std::tie(depthImage, depthImageMemory) =
        createImage(m_swapchainExtent.width, m_swapchainExtent.height, 1,
                    msaaSamples, DepthFmt, vk::ImageTiling::eOptimal,
                    vk::ImageUsageFlagBits::eDepthStencilAttachment,
                    vk::MemoryPropertyFlagBits::eDeviceLocal);
    depthImageView = createImageView(depthImage, DepthFmt,
                                     vk::ImageAspectFlagBits::eDepth, 1);

    // As I understand this part is optional as we will take care of this in the
    // render pass.
    transitionImageLayout(depthImage, DepthFmt, vk::ImageLayout::eUndefined,
                          vk::ImageLayout::eDepthStencilAttachmentOptimal, 1);
  }

  vk::Format findDepthFormat() {
    using F = vk::Format;
    return findSupportedFormat(
        {F::eD32Sfloat, F::eD32SfloatS8Uint, F::eD24UnormS8Uint},
        vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eDepthStencilAttachment);
  }

  bool hasStencilComponent(vk::Format Fmt) {
    return Fmt == vk::Format::eD32SfloatS8Uint ||
           Fmt == vk::Format::eD24UnormS8Uint;
  }

  // Takes a lists of candidate formats from most desireable to the least
  // desirable and checks the first one is supported.
  vk::Format findSupportedFormat(std::vector<vk::Format> const &Candidates,
                                 vk::ImageTiling Tiling,
                                 vk::FormatFeatureFlags Feats) {
    for (vk::Format const &Format : Candidates) {
      vk::FormatProperties Props = m_physicalDevice.getFormatProperties(Format);
      if (Tiling == vk::ImageTiling::eLinear &&
          (Props.linearTilingFeatures & Feats) == Feats)
        return Format;
      else if (Tiling == vk::ImageTiling::eOptimal &&
               (Props.optimalTilingFeatures & Feats) == Feats)
        return Format;
    }

    throw std::runtime_error("failed to find supported format!");
  }

  void createTextureSampler() {
    // TODO retrieve properties once in program
    vk::PhysicalDeviceProperties Props = m_physicalDevice.getProperties();

    textureSampler = m_device.createSampler({{},
                                             vk::Filter::eNearest,
                                             vk::Filter::eLinear,
                                             vk::SamplerMipmapMode::eLinear,
                                             vk::SamplerAddressMode::eRepeat,
                                             vk::SamplerAddressMode::eRepeat,
                                             vk::SamplerAddressMode::eRepeat,
                                             0.f,
                                             VK_TRUE,
                                             Props.limits.maxSamplerAnisotropy,
                                             VK_FALSE,
                                             vk::CompareOp::eAlways,
                                             0.f,
                                             static_cast<float>(mipLevels),
                                             vk::BorderColor::eIntOpaqueBlack,
                                             VK_FALSE});
  }

  vk::ImageView createImageView(vk::Image const &Image,
                                vk::Format const &Format,
                                vk::ImageAspectFlags AspectFlags,
                                uint32_t MipLevels) {
    // TODO: do we really need function?
    return m_device.createImageView({{},
                                     Image,
                                     vk::ImageViewType::e2D,
                                     Format,
                                     {},
                                     {AspectFlags, 0, MipLevels, 0, 1}});
  }

  void createTextureImageView() {
    textureImageView =
        createImageView(textureImage, vk::Format::eR8G8B8A8Srgb,
                        vk::ImageAspectFlagBits::eColor, mipLevels);
  }

  void transitionImageLayout(vk::Image Image, vk::Format Fmt,
                             vk::ImageLayout OldLayout,
                             vk::ImageLayout NewLayout, uint32_t MipLvls) {
    vk::CommandBuffer CmdBuffer = beginSingleTimeCommands();

    vk::ImageMemoryBarrier Barrier{
        {}, {}, OldLayout, NewLayout,
        {}, {}, Image,     {vk::ImageAspectFlagBits::eColor, 0, MipLvls, 0, 1}};
    vk::PipelineStageFlags SrcStage;
    vk::PipelineStageFlags DstStage;

    using IL = vk::ImageLayout;
    if (NewLayout == IL::eDepthStencilAttachmentOptimal) {
      Barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;
      if (hasStencilComponent(Fmt))
        Barrier.subresourceRange.aspectMask |=
            vk::ImageAspectFlagBits::eStencil;
      // TODO this else causes validation error. But it is actually useless.
      // else
      //  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    }

    using AF = vk::AccessFlagBits;
    using PS = vk::PipelineStageFlagBits;
    if (OldLayout == IL::eUndefined && NewLayout == IL::eTransferDstOptimal) {
      Barrier.srcAccessMask = {};
      Barrier.dstAccessMask = AF::eTransferWrite;
      SrcStage = PS::eTopOfPipe;
      DstStage = PS::eTransfer;
    } else if (OldLayout == IL::eTransferDstOptimal &&
               NewLayout == IL::eShaderReadOnlyOptimal) {
      Barrier.srcAccessMask = AF::eTransferWrite;
      Barrier.dstAccessMask = AF::eShaderRead;
      SrcStage = PS::eTransfer;
      DstStage = PS::eFragmentShader;
    } else if (OldLayout == IL::eUndefined &&
               NewLayout == IL::eDepthStencilAttachmentOptimal) {
      Barrier.srcAccessMask = {};
      Barrier.dstAccessMask =
          AF::eDepthStencilAttachmentRead | AF::eDepthStencilAttachmentWrite;
      SrcStage = PS::eTopOfPipe;
      DstStage = PS::eEarlyFragmentTests;
    } else
      throw std::invalid_argument("unsupported layout transition!");

    CmdBuffer.pipelineBarrier(SrcStage, DstStage, {}, nullptr, nullptr,
                              Barrier);

    endSingleTimeCommands(CmdBuffer);
  }

  void copyBufferToImage(vk::Buffer Buffer, vk::Image Image, uint32_t Width,
                         uint32_t Height) {
    vk::CommandBuffer CmdBuff = beginSingleTimeCommands();

    vk::BufferImageCopy Reg{
        0,         0,
        0,         {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        {0, 0, 0}, {Width, Height, 1}};
    CmdBuff.copyBufferToImage(Buffer, Image,
                              vk::ImageLayout::eTransferDstOptimal, Reg);

    endSingleTimeCommands(CmdBuff);
  }

  void generateMipmaps(vk::Image Img, vk::Format Fmt, uint32_t Width,
                       uint32_t Height, uint32_t MipLvls) {
    // Check if image format supports linear blitting.
    vk::FormatProperties FmtProps = m_physicalDevice.getFormatProperties(Fmt);

    if (!(FmtProps.optimalTilingFeatures &
          vk::FormatFeatureFlagBits::eSampledImageFilterLinear))
      throw std::runtime_error(
          "texture image format does not support linear blitting!");

    vk::CommandBuffer CmdBuff = beginSingleTimeCommands();

    vk::ImageMemoryBarrier Barrier{
        {},  {},
        {},  {},
        {},  {},
        Img, {vk::ImageAspectFlagBits::eColor, {/*miplevel*/}, 1, 0, 1}};

    int32_t MipWidth = static_cast<int32_t>(Width);
    int32_t MipHeight = static_cast<int32_t>(Height);

    using IL = vk::ImageLayout;
    using AF = vk::AccessFlagBits;
    using Off = vk::Offset3D;
    for (uint32_t i = 1; i != MipLvls; ++i) {
      Barrier.subresourceRange.baseMipLevel = i - 1;
      Barrier.oldLayout = IL::eTransferDstOptimal;
      Barrier.newLayout = IL::eTransferSrcOptimal;
      Barrier.srcAccessMask = AF::eTransferWrite;
      Barrier.dstAccessMask = AF::eTransferRead;
      CmdBuff.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                              vk::PipelineStageFlagBits::eTransfer, {}, nullptr,
                              nullptr, Barrier);

      auto GetHalf = [](int32_t MipDim) { return MipDim > 1 ? MipDim / 2 : 1; };
      vk::ImageBlit Blit{
          /*Src*/ {vk::ImageAspectFlagBits::eColor, i - 1, 0, 1},
          {Off{0, 0, 0}, Off{MipWidth, MipHeight, 1}},
          /*Dst*/ {vk::ImageAspectFlagBits::eColor, i, 0, 1},
          {Off{0, 0, 0}, Off{GetHalf(MipWidth), GetHalf(MipHeight), 1}}};
      // must be submitted to a queue with graphics capability.
      CmdBuff.blitImage(Img, IL::eTransferSrcOptimal, Img,
                        IL::eTransferDstOptimal, Blit, vk::Filter::eLinear);

      Barrier.oldLayout = IL::eTransferSrcOptimal;
      Barrier.newLayout = IL::eShaderReadOnlyOptimal;
      Barrier.srcAccessMask = AF::eTransferRead;
      Barrier.dstAccessMask = AF::eShaderRead;
      CmdBuff.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                              vk::PipelineStageFlagBits::eFragmentShader, {},
                              nullptr, nullptr, Barrier);

      if (MipWidth > 1)
        MipWidth /= 2;
      if (MipHeight > 1)
        MipHeight /= 2;
    }

    Barrier.subresourceRange.baseMipLevel = MipLvls - 1;
    Barrier.oldLayout = IL::eTransferDstOptimal;
    Barrier.newLayout = IL::eShaderReadOnlyOptimal;
    Barrier.srcAccessMask = AF::eTransferWrite;
    Barrier.dstAccessMask = AF::eShaderRead;
    CmdBuff.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eFragmentShader, {},
                            nullptr, nullptr, Barrier);

    endSingleTimeCommands(CmdBuff);
  }

  void createTextureImage() {
    fs::path ImagePath{EH};
    ImagePath /= "assets/textures/viking_room.png";
    image Image(ImagePath.generic_string());

    uint32_t const ImageSize = Image.getSize();
    uint32_t const Width = Image.getWidth();
    uint32_t const Height = Image.getHeight();
    mipLevels =
        static_cast<uint32_t>(std::floor(std::log2(std::max(Width, Height)))) +
        1;

    auto [StagingBuff, StagingBuffMem] =
        createBuffer(ImageSize, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible |
                         vk::MemoryPropertyFlagBits::eHostCoherent);

    void *Data = m_device.mapMemory(StagingBuffMem, 0, ImageSize);
    memcpy(Data, Image.getRawData(), ImageSize);
    m_device.unmapMemory(StagingBuffMem);

    std::tie(textureImage, textureImageMemory) =
        createImage(Width, Height, mipLevels, vk::SampleCountFlagBits::e1,
                    vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal,
                    vk::ImageUsageFlagBits::eTransferSrc |
                        vk::ImageUsageFlagBits::eTransferDst |
                        vk::ImageUsageFlagBits::eSampled,
                    vk::MemoryPropertyFlagBits::eDeviceLocal);

    transitionImageLayout(textureImage, vk::Format::eR8G8B8A8Srgb,
                          vk::ImageLayout::eUndefined,
                          vk::ImageLayout::eTransferDstOptimal, mipLevels);
    copyBufferToImage(StagingBuff, textureImage, Width, Height);
    // Transitioning to SHADER_READ_ONLY while generating mipmaps.
    generateMipmaps(textureImage, vk::Format::eR8G8B8A8Srgb, Width, Height,
                    mipLevels);

    m_device.destroyBuffer(StagingBuff);
    m_device.freeMemory(StagingBuffMem);
  }

  std::pair<vk::Image, vk::DeviceMemory>
  createImage(uint32_t Width, uint32_t Height, uint32_t MipLevls,
              vk::SampleCountFlagBits NumSample, vk::Format Fmt,
              vk::ImageTiling Tiling, vk::ImageUsageFlags Usage,
              vk::MemoryPropertyFlags Props) {
    vk::ImageCreateInfo ImageInfo{{},        vk::ImageType::e2D,
                                  Fmt,       {Width, Height, 1},
                                  MipLevls,  1,
                                  NumSample, Tiling,
                                  Usage,     vk::SharingMode::eExclusive};
    vk::Image Image = m_device.createImage(ImageInfo);
    vk::MemoryRequirements MemReq = m_device.getImageMemoryRequirements(Image);
    vk::DeviceMemory ImageMem = m_device.allocateMemory(
        {MemReq.size, findMemoryType(MemReq.memoryTypeBits, Props)});

    m_device.bindImageMemory(Image, ImageMem, 0);

    return {Image, ImageMem};
  }

  VkCommandBuffer beginSingleTimeCommands() {
    VkCommandBufferAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = m_commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1};

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(m_device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
  }

  void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                            .commandBufferCount = 1,
                            .pCommandBuffers = &commandBuffer};

    vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_graphicsQueue);

    vkFreeCommandBuffers(m_device, m_commandPool, 1, &commandBuffer);
  }

  void createDescriptorSets() {
    std::vector<vk::DescriptorSetLayout> Layouts(MAX_FRAMES_IN_FLIGHT,
                                                 descriptorSetLayout);

    descriptorSets = m_device.allocateDescriptorSets({descriptorPool, Layouts});

    for (size_t i = 0; i != MAX_FRAMES_IN_FLIGHT; ++i) {
      vk::DescriptorBufferInfo BufInfo{uniformBuffers[i], 0,
                                        sizeof(UniformBufferObject)};
      // TODO. move from loop.
      vk::DescriptorImageInfo ImgInfo{textureSampler, textureImageView,
                                      vk::ImageLayout::eShaderReadOnlyOptimal};

      vk::WriteDescriptorSet BufWrite{
          descriptorSets[i], 0,       0, vk::DescriptorType::eUniformBuffer,
          nullptr,           BufInfo};
      vk::WriteDescriptorSet ImgWrite{descriptorSets[i],1,0,vk::DescriptorType::eCombinedImageSampler,ImgInfo,nullptr};

      std::array<vk::WriteDescriptorSet, 2> descriptorWrites{
          std::move(BufWrite), std::move(ImgWrite)};

      m_device.updateDescriptorSets(descriptorWrites, nullptr);
    }
  }

  void createDescriptorPool() {
    uint32_t const Frames = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

    std::array<vk::DescriptorPoolSize, 2> PoolSizes = {
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, Frames},
        vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler,
                               Frames}};
    descriptorPool = m_device.createDescriptorPool({{}, Frames, PoolSizes});
  }

  void createUniformBuffers() {
    vk::DeviceSize BuffSize = sizeof(UniformBufferObject);

    // TODO. rethink this approach.
    uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i != MAX_FRAMES_IN_FLIGHT; ++i)
      std::tie(uniformBuffers[i], uniformBuffersMemory[i]) =
          createBuffer(BuffSize, vk::BufferUsageFlagBits::eUniformBuffer,
                       vk::MemoryPropertyFlagBits::eHostVisible |
                           vk::MemoryPropertyFlagBits::eHostCoherent);
  }

  void createDescriptorSetLayout() {
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

    descriptorSetLayout = m_device.createDescriptorSetLayout({{}, Bindings});
  }

  void createVertexBuffer() {
    vk::DeviceSize BuffSize = sizeof(Vertex) * Vertices.size();

    auto [StagingBuff, StagingBuffMem] =
        createBuffer(BuffSize, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible |
                         vk::MemoryPropertyFlagBits::eHostCoherent);

    void *Data = m_device.mapMemory(StagingBuffMem, 0, BuffSize);
    memcpy(Data, Vertices.data(), (size_t)BuffSize);
    m_device.unmapMemory(StagingBuffMem);

    std::tie(VertexBuffer, VertexBufferMemory) =
        createBuffer(BuffSize,
                     vk::BufferUsageFlagBits::eTransferDst |
                         vk::BufferUsageFlagBits::eVertexBuffer,
                     vk::MemoryPropertyFlagBits::eDeviceLocal);

    copyBuffer(StagingBuff, VertexBuffer, BuffSize);

    m_device.destroyBuffer(StagingBuff);
    m_device.freeMemory(StagingBuffMem);
  }

  void createIndexBuffer() {
    vk::DeviceSize BuffSize = sizeof(uint32_t) * Indices.size();

    auto [StagingBuff, StagingBuffMem] =
        createBuffer(BuffSize, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible |
                         vk::MemoryPropertyFlagBits::eHostCoherent);
    void *Data = m_device.mapMemory(StagingBuffMem, 0, BuffSize);
    memcpy(Data, Indices.data(),
           BuffSize); // TODO why just data == Indices.data()?
    m_device.unmapMemory(StagingBuffMem);

    std::tie(IndexBuffer, IndexBufferMemory) =
        createBuffer(BuffSize,
                     vk::BufferUsageFlagBits::eTransferDst |
                         vk::BufferUsageFlagBits::eIndexBuffer,
                     vk::MemoryPropertyFlagBits::eDeviceLocal);

    copyBuffer(StagingBuff, IndexBuffer, BuffSize);

    m_device.destroyBuffer(StagingBuff);
    m_device.freeMemory(StagingBuffMem);
  }

  void copyBuffer(vk::Buffer Src, vk::Buffer Dst, vk::DeviceSize Size) {
    // TODO. Maybe separate command pool is to be created for these kinds of
    // short-lived buffers, because the implementation may be able to apply
    // memory allocation optimizations. VK_COMMAND_POOL_CREATE_TRANSIENT_BIT for
    // commandPool generation in that case.
    vk::CommandBuffer CmdBuff = beginSingleTimeCommands();
    CmdBuff.copyBuffer(Src, Dst, vk::BufferCopy{{}, {}, Size});
    endSingleTimeCommands(CmdBuff);
  }

  // Creates new swap chain when smth goes wrong or resizing.
  void recreateSwapchain() {
    // Wait till proper size.
    int width = 0, height = 0;
    glfwGetFramebufferSize(m_Window, &width, &height);
    while (width == 0 || height == 0) {
      glfwGetFramebufferSize(m_Window, &width, &height);
      glfwWaitEvents();
    }

    vkDeviceWaitIdle(m_device);

    createSwapchain();
    createImageViews();
    createRenderPass();
    createGraphicPipeline();
    createColorResources();
    createDepthResources();
    createFramebuffers();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
  }

  void updateUniformBuffer(uint32_t currentImage) {
    static auto const startTime = std::chrono::high_resolution_clock::now();

    auto const currentTime = std::chrono::high_resolution_clock::now();
    float const time =
        std::chrono::duration<float, std::chrono::seconds::period>(currentTime -
                                                                   startTime)
            .count();
    UniformBufferObject ubo{
        .model = glm::rotate(glm::mat4(1.f), time * glm::radians(90.f),
                             glm::vec3(0.f, 0.f, 1.f)),
        .view = glm::lookAt(glm::vec3(2.f, 2.f, 2.f), glm::vec3(0.f, 0.f, 0.f),
                            glm::vec3(0.f, 0.f, 1.f)),
        .proj = glm::perspective(glm::radians(45.f),
                                 m_swapchainExtent.width /
                                     (float)m_swapchainExtent.height,
                                 0.1f, 10.f)};
    ubo.proj[1][1] *= -1; // because GLM designed for OpenGL.

    void *data;
    vkMapMemory(m_device, uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0,
                &data);
    memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(m_device, uniformBuffersMemory[currentImage]);
  }

  std::pair<vk::Buffer, vk::DeviceMemory>
  createBuffer(vk::DeviceSize Size, vk::BufferUsageFlags Usage,
               vk::MemoryPropertyFlags Properties) {
    // TODO for transfering VK_QUEUE_TRANSFER_BIT is needed, but it included in
    // VK_QUEUE_GRAPHICS_BIT or COMPUTE_BIT. But it would be nice to create
    // queue family specially with TRANSFER_BIT.
    vk::Buffer Buffer =
        m_device.createBuffer({{}, Size, Usage, vk::SharingMode::eExclusive});
    vk::MemoryRequirements MemReq =
        m_device.getBufferMemoryRequirements(Buffer);
    vk::DeviceMemory Memory = m_device.allocateMemory(
        {MemReq.size, findMemoryType(MemReq.memoryTypeBits, Properties)});
    m_device.bindBufferMemory(Buffer, Memory, 0);

    return std::make_pair(Buffer, Memory);
  }

  void createSyncObjects() {
    m_imageAvailableSemaphore.resize(MAX_FRAMES_IN_FLIGHT);
    m_renderFinishedSemaphore.resize(MAX_FRAMES_IN_FLIGHT);
    m_inFlightFence.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semaphoreInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};

    VkFenceCreateInfo fenceInfo{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                                .flags = VK_FENCE_CREATE_SIGNALED_BIT};

    for (int i = 0; i != MAX_FRAMES_IN_FLIGHT; ++i)
      if (vkCreateSemaphore(m_device, &semaphoreInfo, nullptr,
                            &m_imageAvailableSemaphore[i]) != VK_SUCCESS ||
          vkCreateSemaphore(m_device, &semaphoreInfo, nullptr,
                            &m_renderFinishedSemaphore[i]) != VK_SUCCESS ||
          vkCreateFence(m_device, &fenceInfo, nullptr, &m_inFlightFence[i]) !=
              VK_SUCCESS) {
        throw std::runtime_error("failed to create semaphores!");
      }
  }

  void createCommandBuffers() {
    m_commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

    VkCommandBufferAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = m_commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = (uint32_t)m_commandBuffers.size()};

    if (vkAllocateCommandBuffers(m_device, &allocInfo,
                                 m_commandBuffers.data()) != VK_SUCCESS)
      throw std::runtime_error("failed to allocate command buffers!");
  }

  void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
    VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = 0,
        .pInheritanceInfo = nullptr};

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
      throw std::runtime_error("failed to begin recording command buffer!");
    }

    // Order of clear values should be indentical to the order of attachments.
    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};
    VkRenderPassBeginInfo renderPassInfo{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = m_renderPass,
        .framebuffer = m_swapChainFramebuffers[imageIndex],
        .clearValueCount = static_cast<uint32_t>(clearValues.size()),
        .pClearValues = clearValues.data()};
    // This part is to avoid "nested designators extension".
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent =
        static_cast<VkExtent2D>(m_swapchainExtent);

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo,
                         VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      m_graphicsPipeline);

    VkBuffer vertexBuffers[] = {VertexBuffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(commandBuffer, IndexBuffer, 0, VK_INDEX_TYPE_UINT32);

    // TODO. temprorary.
    vk::CommandBuffer CmdBuff = commandBuffer;
    CmdBuff.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                               m_pipelineLayout, 0,
                               descriptorSets[currentFrame], nullptr);

    // vertexCount, instanceCount, fitstVertex, firstInstance
    vkCmdDrawIndexed(commandBuffer, Indices.size(), 1, 0, 0, 0);
    // vkCmdDraw(commandBuffer, static_cast<uint32_t>(Vertices.size()), 1, 0,
    // 0);

    vkCmdEndRenderPass(commandBuffer);
    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
      throw std::runtime_error("failed to record command buffer!");
    }
  }

  void createCommandPool() {
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(m_physicalDevice);
    m_commandPool = m_device.createCommandPool(
        {vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
         queueFamilyIndices.GraphicsFamily.value()});
  }

  void createFramebuffers() {
    // Order of the attachments is essential!
    // It is reverse from created in createRenderPass
    std::array<vk::ImageView, 3> Atts = {colorImageView, depthImageView, {}};
    std::vector<vk::Framebuffer> swapchainBuffrs;
    std::transform(m_swapchainImageViews.begin(), m_swapchainImageViews.end(),
                   std::back_inserter(swapchainBuffrs),
                   [&Atts, this](vk::ImageView &ImgV) {
                     Atts[2] = ImgV;
                     return m_device.createFramebuffer(
                         {{},
                          m_renderPass,
                          Atts,
                          m_swapchainExtent.width,
                          m_swapchainExtent.height,
                          1});
                   });

    m_swapChainFramebuffers = std::move(swapchainBuffrs);
  }

  void createRenderPass() {
    vk::AttachmentDescription DepthAtt{
        {},
        findDepthFormat(),
        msaaSamples,
        vk::AttachmentLoadOp::eClear,
        vk::AttachmentStoreOp::eDontCare,
        vk::AttachmentLoadOp::eDontCare,
        vk::AttachmentStoreOp::eDontCare,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eDepthStencilAttachmentOptimal};
    vk::AttachmentReference DepthAttRef{
        1, vk::ImageLayout::eDepthStencilAttachmentOptimal};

    vk::AttachmentDescription ColorAtt{
        {},
        m_swapchainImageFormat,
        msaaSamples,
        vk::AttachmentLoadOp::eClear,
        vk::AttachmentStoreOp::eStore,
        vk::AttachmentLoadOp::eDontCare,
        vk::AttachmentStoreOp::eDontCare,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal};
    vk::AttachmentReference ColorAttRef{
        0, vk::ImageLayout::eColorAttachmentOptimal};

    vk::AttachmentDescription ColorAttResolve{{},
                                              m_swapchainImageFormat,
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
    m_renderPass =
        m_device.createRenderPass({{}, Attachments, Subpass, Dependency});
  }

  void createGraphicPipeline() {
    fs::path ShadersPath{EH};
    ShadersPath /= "shaders/";
    Shader VShader{(ShadersPath / "basic.vert.spv").string()};
    Shader FShader{(ShadersPath / "basic.frag.spv").string()};

    vk::ShaderModule VShaderModule =
        m_device.createShaderModule({{}, VShader.getSPIRV()});
    vk::ShaderModule FShaderModule =
        m_device.createShaderModule({{}, FShader.getSPIRV()});

    vk::PipelineShaderStageCreateInfo VSInfo{
        {}, vk::ShaderStageFlagBits::eVertex, VShaderModule, "main"};
    vk::PipelineShaderStageCreateInfo FSInfo{
        {}, vk::ShaderStageFlagBits::eFragment, FShaderModule, "main"};

    std::array<vk::PipelineShaderStageCreateInfo, 2> ShaderStages{VSInfo,
                                                                  FSInfo};

    // Fill Vertex binding info.
    auto bindDescr = Vertex::getBindDescription();
    auto attrDescr = Vertex::getAttrDescription();

    vk::PipelineVertexInputStateCreateInfo VInputInfo{{}, bindDescr, attrDescr};

    // The rules how verticies will be treated(lines, points, triangles..)
    vk::PipelineInputAssemblyStateCreateInfo InputAssembly{
        {}, vk::PrimitiveTopology::eTriangleList, VK_FALSE};

    // Read this in FixedFunction part.
    vk::Viewport Viewport{0.0f,
                          0.0f,
                          (float)m_swapchainExtent.width,
                          (float)m_swapchainExtent.height,
                          0.0f,
                          1.0f};
    vk::Rect2D Scissor{{0, 0}, m_swapchainExtent};

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
        {}, msaaSamples, VK_TRUE, .2f, nullptr};

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

    m_pipelineLayout = m_device.createPipelineLayout({{}, descriptorSetLayout});

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
                                                m_pipelineLayout,
                                                m_renderPass,
                                                {},
                                                {},
                                                -1};

    // It creates several pipelines.
    m_graphicsPipeline =
        m_device.createGraphicsPipelines(VK_NULL_HANDLE, PipelineInfo)
            .value.at(0);

    m_device.destroyShaderModule(FShaderModule);
    m_device.destroyShaderModule(VShaderModule);
  }

  // Finds right type of memory to use.
  uint32_t findMemoryType(uint32_t TypeFilter,
                          vk::MemoryPropertyFlags Properties) {
    vk::PhysicalDeviceMemoryProperties MemProps =
        m_physicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i != MemProps.memoryTypeCount; ++i)
      if ((TypeFilter & (1 << i)) &&
          (MemProps.memoryTypes[i].propertyFlags & Properties) == Properties)
        return i;

    throw std::runtime_error("failed to find suitable memory type!");
  }

  void createImageViews() {
    std::vector<vk::ImageView> ImgViews;
    std::transform(m_swapchainImages.begin(), m_swapchainImages.end(),
                   std::back_inserter(ImgViews), [this](vk::Image const &Img) {
                     return createImageView(Img, m_swapchainImageFormat,
                                            vk::ImageAspectFlagBits::eColor, 1);
                   });

    m_swapchainImageViews = std::move(ImgViews);
  }

  void createSwapchain() {
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

  void createSurface() {
    if (glfwCreateWindowSurface(m_instance, m_Window, nullptr, &m_surface) !=
        VK_SUCCESS)
      throw std::runtime_error("failed to create window surface!");
  }

  void createLogicalDevice() {
    // TODO rethink about using findQueueFamilieses once.
    QueueFamilyIndices Indices = findQueueFamilies(m_physicalDevice);

    // Each queue family has to have own VkDeviceQueueCreateInfo.
    std::vector<vk::DeviceQueueCreateInfo> QueueCreateInfos{};
    // This is the worst way of doing it. Rethink!
    std::set<uint32_t> UniqueIdc = {Indices.GraphicsFamily.value(),
                                    Indices.PresentFamily.value()};
    // TODO: what is this?
    std::array<float, 1> const QueuePriority = {1.f};
    for (uint32_t Family : UniqueIdc)
      QueueCreateInfos.push_back(
          vk::DeviceQueueCreateInfo{{}, Family, QueuePriority});

    vk::PhysicalDeviceFeatures DevFeat{};
    DevFeat.samplerAnisotropy = VK_TRUE;
    DevFeat.sampleRateShading = VK_TRUE;

    // TODO: rethink this approach. May be use smth like std::optional.
    std::vector<char const *> const &Layers = m_enableValidationLayers
                                                  ? m_validationLayers
                                                  : std::vector<char const *>{};
    m_device = m_physicalDevice.createDevice(
        {{}, QueueCreateInfos, Layers, m_deviceExtensions, &DevFeat});

    // We can use the vkGetDeviceQueue function to retrieve queue handles for
    // each queue family. The parameters are the logical device, queue family,
    // queue index and a pointer to the variable to store the queue handle in.
    // Because we're only creating a single queue from this family, we'll simply
    // use index 0.
    m_graphicsQueue = m_device.getQueue(Indices.GraphicsFamily.value(), 0);
    m_presentQueue = m_device.getQueue(Indices.PresentFamily.value(), 0);
  }

  void pickPhysicalDevice() {
    std::vector<vk::PhysicalDevice> Devices =
        m_instance.enumeratePhysicalDevices();

    auto FindIt =
        std::find_if(Devices.begin(), Devices.end(), [this](auto &&Device) {
          return isDeviceSuitable(Device);
        });
    if (FindIt == Devices.end())
      throw std::runtime_error("failed to find a suitable GPU!");

    m_physicalDevice = *FindIt;
    msaaSamples = getMaxUsableSampleCount();
  }

  // Checks which queue families are supported by the device and which one of
  // these supports the commands that we want to use.
  struct QueueFamilyIndices {
    // optional just because we may be want to select GPU with some family, but
    // it is not strictly necessary.
    std::optional<uint32_t> GraphicsFamily{};
    // Not every device can support presentation of the image, so need to
    // check that divece has proper family queue.
    std::optional<uint32_t> PresentFamily{};

    bool isComplete() {
      return GraphicsFamily.has_value() && PresentFamily.has_value();
    }
  };

  // Swapchain requires more details to be checked.
  // - basic surface capabilities.
  // - surface format (pixel format, color space).
  // - available presentation mode.
  struct SwapchainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
  };

  // This section covers how to query the structs that include this information.
  SwapchainSupportDetails
  querySwapchainSupport(vk::PhysicalDevice const &Device) {
    return SwapchainSupportDetails{
        .capabilities = Device.getSurfaceCapabilitiesKHR(m_surface),
        .formats = Device.getSurfaceFormatsKHR(m_surface),
        .presentModes = Device.getSurfacePresentModesKHR(m_surface)};
  }

  QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice const &Device) {
    QueueFamilyIndices Indices;
    std::vector<vk::QueueFamilyProperties> QueueFamilies =
        Device.getQueueFamilyProperties();

    // TODO: rething this approach.
    uint32_t i{0};
    for (auto &&queueFamily : QueueFamilies) {
      // For better performance one queue family has to support all requested
      // queues at once, but we also can treat them as different families for
      // unified approach.
      if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)
        Indices.GraphicsFamily = i;

      // Checks for presentation family support.
      if (Device.getSurfaceSupportKHR(i, m_surface))
        Indices.PresentFamily = i;

      // Not quite sure why the hell we need this early-break.
      if (Indices.isComplete())
        return Indices;
      ++i;
    }

    return QueueFamilyIndices{};
  }

  // We want to select format.
  vk::SurfaceFormatKHR chooseSwapSurfaceFormat(
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

  // Presentation mode represents the actual conditions for showing images to
  // the screen.
  vk::PresentModeKHR chooseSwapPresentMode(
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

  // glfwWindows works with screen-coordinates. But Vulkan - with pixels.
  // And not always they are corresponsible with each other.
  // So we want to create a proper resolution.
  vk::Extent2D
  chooseSwapExtent(vk::SurfaceCapabilitiesKHR const &Capabilities) {
    if (Capabilities.currentExtent.width != UINT32_MAX)
      return Capabilities.currentExtent;
    else {
      int Width{0}, Height{0};
      glfwGetFramebufferSize(m_Window, &Width, &Height);

      uint32_t const RealW = std::clamp(static_cast<uint32_t>(Width),
                                        Capabilities.minImageExtent.width,
                                        Capabilities.maxImageExtent.width);
      uint32_t const RealH = std::clamp(static_cast<uint32_t>(Height),
                                        Capabilities.minImageExtent.height,
                                        Capabilities.maxImageExtent.height);
      return {RealW, RealH};
    }
  }

  // Checks if device is suitable for our extensions and purposes.
  bool isDeviceSuitable(vk::PhysicalDevice const &Device) {
    // name, type, supported Vulkan version can be quired via
    // GetPhysicalDeviceProperties.
    vk::PhysicalDeviceProperties DeviceProps = Device.getProperties();

    // optional features like texture compressing, 64bit floating operations,
    // multiview-port and so one...
    vk::PhysicalDeviceFeatures DeviceFeat = Device.getFeatures();

    // Right now I have only one INTEGRATED GPU(on linux). But it will be more
    // suitable to calculate score and select preferred GPU with the highest
    // score. (eg. discrete GPU has +1000 score..)

    // Swap chain support is sufficient for this tutorial if there is at least
    // one supported image format and one supported presentation mode given
    // the window surface we have.
    auto swapchainSupport = [](SwapchainSupportDetails swapchainDetails) {
      return !swapchainDetails.formats.empty() &&
             !swapchainDetails.presentModes.empty();
    };

    // But we want to find out if GPU is graphicFamily. (?)
    return findQueueFamilies(Device).isComplete() &&
           checkDeviceExtensionSupport(Device) &&
           swapchainSupport(querySwapchainSupport(Device)) &&
           DeviceFeat.samplerAnisotropy;
    // All three ckecks are different. WTF!
  }

  bool checkDeviceExtensionSupport(vk::PhysicalDevice const &Device) {
    // Some bad code. Rethink!
    std::vector<vk::ExtensionProperties> AvailableExtensions =
        Device.enumerateDeviceExtensionProperties();
    std::unordered_set<std::string_view> UniqieExt;
    std::for_each(AvailableExtensions.begin(), AvailableExtensions.end(),
                  [&UniqieExt](auto &&Extension) {
                    UniqieExt.insert(Extension.extensionName);
                  });
    return std::all_of(m_deviceExtensions.begin(), m_deviceExtensions.end(),
                       [&UniqieExt](char const *RequireExt) {
                         return UniqieExt.contains(RequireExt);
                       });
  }

  vk::DebugUtilsMessengerCreateInfoEXT populateDebugMessengerInfo() {
    using MessageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT;
    using MessageType = vk::DebugUtilsMessageTypeFlagBitsEXT;
    return {{},
            MessageSeverity::eVerbose | MessageSeverity::eWarning |
                MessageSeverity::eError,
            MessageType::eGeneral | MessageType::ePerformance |
                MessageType::eValidation,
            debugCallback};
  }

  void setupDebugMessenger() {
    if (!m_enableValidationLayers)
      return;

    // TODO think about C++ style.
    VkDebugUtilsMessengerCreateInfoEXT CreateInfo =
        static_cast<VkDebugUtilsMessengerCreateInfoEXT>(
            populateDebugMessengerInfo());
    if (createDebugUtilsMessengerEXT(m_instance, &CreateInfo, nullptr,
                                     &m_debugMessenger) != VK_SUCCESS)
      throw std::runtime_error("failed to set up debug messenger!");
  }

  std::vector<char const *> getRequiredExtensions() {
    uint32_t ExtensionCount{};
    const char **Extensions =
        glfwGetRequiredInstanceExtensions(&ExtensionCount);

    std::vector<const char *> AllExtensions{Extensions,
                                            Extensions + ExtensionCount};
    // Addition for callback.
    if (m_enableValidationLayers)
      AllExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    return AllExtensions;
  }

  void createInstance() {
    // TODO believe this logic has to be moved somewhere to call constructor of
    // m_instance only once, fe: m_instance{createInstance}.
    if (m_enableValidationLayers && !checkValidationLayers())
      throw std::runtime_error{
          "Requestred validation layers are not available!"};

    vk::ApplicationInfo AppInfo{"Hello triangle", VK_MAKE_VERSION(1, 0, 0),
                                "No Engine", VK_MAKE_VERSION(1, 0, 0),
                                VK_API_VERSION_1_0};

    std::vector<char const *> Extensions{getRequiredExtensions()};
    std::vector<char const *> const &Layers = m_enableValidationLayers
                                                  ? m_validationLayers
                                                  : std::vector<char const *>{};

    vk::InstanceCreateInfo CreateInfo{{}, &AppInfo, Layers, Extensions};

    vk::DebugUtilsMessengerCreateInfoEXT DebugCreateInfo =
        m_enableValidationLayers ? populateDebugMessengerInfo()
                                 : vk::DebugUtilsMessengerCreateInfoEXT{};
    if (m_enableValidationLayers)
      CreateInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT *)&DebugCreateInfo;

    m_instance = vk::createInstance(CreateInfo);
  }

  // Checks all required extensions listed in ReqExt for occurance in VK
  void checkExtensions(char const **ReqExt, uint32_t ExtCount) {
    assert(ReqExt);
    assert(ExtCount);

    uint32_t AvailableExtCount{};
    vkEnumerateInstanceExtensionProperties(nullptr, &AvailableExtCount,
                                           nullptr);
    std::vector<VkExtensionProperties> AvailableExts(AvailableExtCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &AvailableExtCount,
                                           AvailableExts.data());

    uint32_t idx = 0;
    while (idx != ExtCount) {
      bool isFounded{false};
      for (auto &&Ext : AvailableExts)
        if (!strcmp(Ext.extensionName, ReqExt[idx])) {
          idx++;
          isFounded = true;
        }
      if (!isFounded)
        std::cerr << "Extension: " << ReqExt[idx] << " is not founded!\n";
    }
  }

  // Checks if all requested layers are available.
  bool checkValidationLayers() {
    auto AvailableLayers = vk::enumerateInstanceLayerProperties();
    std::unordered_set<std::string_view> UniqueLayers;
    std::for_each(AvailableLayers.begin(), AvailableLayers.end(),
                  [&UniqueLayers](auto const &LayerProperty) {
                    UniqueLayers.insert(LayerProperty.layerName);
                  });
    return std::all_of(m_validationLayers.begin(), m_validationLayers.end(),
                       [&UniqueLayers](char const *RequireLayer) {
                         return UniqueLayers.contains(RequireLayer);
                       });
  }

  static VKAPI_ATTR VkBool32 VKAPI_CALL
  debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                VkDebugUtilsMessageTypeFlagsEXT messageType,
                const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
                void *pUserData) {

    // The way to control dumping  validation info.
    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
      // Message is important enough to show
      std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
    }

    return VK_FALSE;
  }

  void mainLoop() {

    uint32_t MaxFrames{5};
    uint32_t Frame{0};

    while (!glfwWindowShouldClose(m_Window) && Frame++ < MaxFrames) {
      glfwPollEvents();
      drawFrame();
    }

    vkDeviceWaitIdle(m_device);
  }

  uint32_t currentFrame = 0;
  void drawFrame() {
    vkWaitForFences(m_device, 1, &m_inFlightFence[currentFrame], VK_TRUE,
                    UINT64_MAX);

    uint32_t imageIndex{0};

    VkResult res = vkAcquireNextImageKHR(
        m_device, m_swapchain, UINT64_MAX,
        m_imageAvailableSemaphore[currentFrame], VK_NULL_HANDLE, &imageIndex);
    if (res == VK_ERROR_OUT_OF_DATE_KHR) {
      recreateSwapchain();
    } else if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR) {
      throw std::runtime_error("failed to acquire swap chain image!");
    }

    updateUniformBuffer(currentFrame);

    // Only reset the fence if we are submitting work
    vkResetFences(m_device, 1, &m_inFlightFence[currentFrame]);

    vkResetCommandBuffer(m_commandBuffers[currentFrame],
                         /*VkCommandBufferResetFlagBits*/ 0);
    recordCommandBuffer(m_commandBuffers[currentFrame], imageIndex);

    VkSemaphore waitSemaphores[] = {m_imageAvailableSemaphore[currentFrame]};
    VkSemaphore signalSemaphores[] = {m_renderFinishedSemaphore[currentFrame]};
    VkPipelineStageFlags waitStages[] = {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    VkSubmitInfo submitInfo{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                            .waitSemaphoreCount = 1,
                            .pWaitSemaphores = waitSemaphores,
                            .pWaitDstStageMask = waitStages,
                            .commandBufferCount = 1,
                            .pCommandBuffers = &m_commandBuffers[currentFrame],
                            .signalSemaphoreCount = 1,
                            .pSignalSemaphores = signalSemaphores};

    if (vkQueueSubmit(m_graphicsQueue, 1, &submitInfo,
                      m_inFlightFence[currentFrame]) != VK_SUCCESS) {
      throw std::runtime_error("failed to submit draw command buffer!");
    }

    VkSwapchainKHR swapChains[] = {m_swapchain};
    VkPresentInfoKHR presentInfo{.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                                 .waitSemaphoreCount = 1,
                                 .pWaitSemaphores = signalSemaphores,
                                 .swapchainCount = 1,
                                 .pSwapchains = swapChains,
                                 .pImageIndices = &imageIndex,
                                 .pResults = nullptr};
    vkQueuePresentKHR(m_presentQueue, &presentInfo);
    if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR ||
        framebufferResized) {
      recreateSwapchain();
      framebufferResized = false;
    } else if (res != VK_SUCCESS)
      throw std::runtime_error("failed to present swap chain image!");

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
  }

  void cleanupSwapchain() {
    vkDestroyImageView(m_device, colorImageView, nullptr);
    vkDestroyImage(m_device, colorImage, nullptr);
    vkFreeMemory(m_device, colorImageMemory, nullptr);

    vkDestroyImageView(m_device, depthImageView, nullptr);
    vkDestroyImage(m_device, depthImage, nullptr);
    vkFreeMemory(m_device, depthImageMemory, nullptr);

    for (size_t i = 0; i < m_swapChainFramebuffers.size(); i++)
      vkDestroyFramebuffer(m_device, m_swapChainFramebuffers[i], nullptr);

    vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyRenderPass(m_device, m_renderPass, nullptr);

    for (size_t i = 0; i < m_swapchainImageViews.size(); i++)
      vkDestroyImageView(m_device, m_swapchainImageViews[i], nullptr);

    vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);

    for (size_t i = 0; i != MAX_FRAMES_IN_FLIGHT; ++i) {
      vkDestroyBuffer(m_device, uniformBuffers[i], nullptr);
      vkFreeMemory(m_device, uniformBuffersMemory[i], nullptr);
    }
  }

  void cleanup() {
    cleanupSwapchain();

    vkDestroySampler(m_device, textureSampler, nullptr);
    vkDestroyImageView(m_device, textureImageView, nullptr);

    vkDestroyImage(m_device, textureImage, nullptr);
    vkFreeMemory(m_device, textureImageMemory, nullptr);

    vkDestroyDescriptorPool(m_device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(m_device, descriptorSetLayout, nullptr);
    vkDestroyBuffer(m_device, VertexBuffer, nullptr);
    vkFreeMemory(m_device, VertexBufferMemory, nullptr);
    vkDestroyBuffer(m_device, IndexBuffer, nullptr);
    vkFreeMemory(m_device, IndexBufferMemory, nullptr);

    for (int i = 0; i != MAX_FRAMES_IN_FLIGHT; ++i) {
      vkDestroySemaphore(m_device, m_renderFinishedSemaphore[i], nullptr);
      vkDestroySemaphore(m_device, m_imageAvailableSemaphore[i], nullptr);
      vkDestroyFence(m_device, m_inFlightFence[i], nullptr);
    }
    vkDestroyCommandPool(m_device, m_commandPool, nullptr);
    vkDestroyDevice(m_device, nullptr);
    if (m_enableValidationLayers)
      destroyDebugUtilsMessengerEXT(m_instance, m_debugMessenger, nullptr);
    vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
    // TODO: Do I need this destroy?
    m_instance.destroy();

    glfwDestroyWindow(m_Window);

    glfwTerminate();
  }
};

int main(int argc, char *argv[]) {
#ifndef NDEBUG
  std::cout << "Debug\n";
#endif // Debug
  Decoy::Dump();

  EnvHandler EH{argv[0]};

  HelloTriangleApplication app{EH};
  try {
    app.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
