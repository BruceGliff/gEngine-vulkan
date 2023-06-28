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

// BAD. just a placeholder
#include "lib/environment/ChainsManager.h"
#include "lib/environment/platform_handler.h"

#include "image/image.h"
#include "shader/shader.h"
#include "vertex.h"

#include "gEng/environment.h"
#include "gEng/global.h"
#include "gEng/image.h"
#include "gEng/window.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <ranges>
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
// TODO this dbg creation has to be in two steps to avoid I = Instance:
//  1 - find pnfVk...
//  2 - call func(...) which would be pnfVk..
VkResult vkCreateDebugUtilsMessengerEXT(
    VkInstance Instance, VkDebugUtilsMessengerCreateInfoEXT const *pCreateInfo,
    VkAllocationCallbacks const *pAllocator,
    VkDebugUtilsMessengerEXT *pDebugMessenger) {
  vk::Instance I = Instance;
  auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
      I.getProcAddr("vkCreateDebugUtilsMessengerEXT"));
  if (func != nullptr)
    return func(Instance, pCreateInfo, pAllocator, pDebugMessenger);
  else
    return VK_ERROR_EXTENSION_NOT_PRESENT;
}
// Some simillar hack.
void vkDestroyDebugUtilsMessengerEXT(VkInstance Instance,
                                     VkDebugUtilsMessengerEXT debugMessenger,
                                     const VkAllocationCallbacks *pAllocator) {
  vk::Instance I = Instance;
  auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
      I.getProcAddr("vkDestroyDebugUtilsMessengerEXT"));
  if (func)
    func(Instance, debugMessenger, pAllocator);
  else
    std::cout << "ERR: debug is not destroyed!\n";
}

struct UniformBufferObject {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
};

// TODO. move Window in UserWindow.
class HelloTriangleApplication : gEng::UserWindow {
  gEng::SysEnv &EH;

  gEng::Window m_Window;

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
  vk::DebugUtilsMessengerEXT m_debugMessenger;
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
  vk::SurfaceKHR m_surface{};

  gEng::ChainsManager Chains{};
  vk::SwapchainKHR m_swapchain{};

  vk::Extent2D m_swapchainExtent;
  // ImageView used to specify how to treat VkImage.
  // For each VkImage we create VkImageView.

  vk::RenderPass m_renderPass;
  // Used for an uniform variable.
  vk::PipelineLayout m_pipelineLayout;

  vk::Pipeline m_graphicsPipeline;

  // A framebuffer object references all of the VkImageView objects.
  std::vector<vk::Framebuffer> m_swapChainFramebuffers;

  vk::CommandPool m_commandPool;

  std::vector<vk::CommandBuffer> m_commandBuffers;

  std::vector<vk::Semaphore> m_imageAvailableSemaphore;
  std::vector<vk::Semaphore> m_renderFinishedSemaphore;

  std::vector<vk::Fence> m_inFlightFence;

  std::vector<Vertex> Vertices;
  std::vector<uint32_t> Indices;

  vk::Buffer VertexBuffer;
  vk::DeviceMemory VertexBufferMemory;

  vk::Buffer IndexBuffer;
  vk::DeviceMemory IndexBufferMemory;

  std::vector<vk::Buffer> uniformBuffers;
  std::vector<vk::DeviceMemory> uniformBuffersMemory;

  vk::DescriptorSetLayout descriptorSetLayout;

  vk::DescriptorPool descriptorPool;
  std::vector<vk::DescriptorSet> descriptorSets;

  uint32_t mipLevels;
  vk::Image textureImage;
  vk::DeviceMemory textureImageMemory;

  vk::ImageView textureImageView;
  vk::Sampler textureSampler;
public:
  HelloTriangleApplication(gEng::SysEnv &InEH)
      : EH{InEH}, m_Window{1600u, 900u, "gEngine", this} {}
  void run() {

    // TODO this is just to test GlbManager interface.
    auto &G = gEng::GlbManager::getInstance();
    G.registerEntity<int>(4);
    std::cout << G.getEntity<int>();

    initVulkan();
    mainLoop();
    cleanup();
  }

private:
  void fillFromChainManager() {
    m_swapchain = Chains.getSwapchain();
    m_swapchainExtent = Chains.getExtent();
    m_renderPass = Chains.getRPass();
    descriptorSetLayout = Chains.getDSL();
    m_pipelineLayout = Chains.getPPL();
    m_graphicsPipeline = Chains.getP();

    m_swapChainFramebuffers = Chains.getFrameBuffers();
  }

  void initVulkan() {
    auto &PltMgn = gEng::PlatformHandler::getInstance();
    PltMgn.init(m_Window);

    m_instance = PltMgn.get<vk::Instance>();
    m_surface = PltMgn.get<vk::SurfaceKHR>();
    m_physicalDevice = PltMgn.get<vk::PhysicalDevice>();
    m_device = PltMgn.get<vk::Device>();
    m_commandPool = PltMgn.get<vk::CommandPool>();

    std::tie(m_graphicsQueue, m_presentQueue) =
        PltMgn.get<gEng::detail::GraphPresentQ>();

    Chains = gEng::ChainsManager{PltMgn, m_Window};
    fillFromChainManager();

    fs::path ImagePath{EH};
    ImagePath /= "assets/textures/viking_room.png";
    gEng::Image Img(ImagePath.generic_string());
    gEng::setImg(textureImage, textureImageMemory, mipLevels, textureImageView,
                 textureSampler, Img);
    loadModel();
    createVertexBuffer();
    createIndexBuffer();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
    createSyncObjects();
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

  // TODO combine with image.cc ImageBuilder
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

  void transitionImageLayout(vk::Image Image, vk::Format Fmt,
                             vk::ImageLayout OldLayout,
                             vk::ImageLayout NewLayout, uint32_t MipLvls) {
    auto &PltMgn = gEng::PlatformHandler::getInstance();
    auto [SSTC, CmdBuffer] = PltMgn.getSSTC();

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

  }

  void copyBufferToImage(vk::Buffer Buffer, vk::Image Image, uint32_t Width,
                         uint32_t Height) {
    auto &PltMgn = gEng::PlatformHandler::getInstance();
    auto [SSTC, CmdBuff] = PltMgn.getSSTC();

    vk::BufferImageCopy Reg{
        0,         0,
        0,         {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        {0, 0, 0}, {Width, Height, 1}};
    CmdBuff.copyBufferToImage(Buffer, Image,
                              vk::ImageLayout::eTransferDstOptimal, Reg);

  }

  void generateMipmaps(vk::Image Img, vk::Format Fmt, uint32_t Width,
                       uint32_t Height, uint32_t MipLvls) {
    // Check if image format supports linear blitting.
    vk::FormatProperties FmtProps = m_physicalDevice.getFormatProperties(Fmt);

    if (!(FmtProps.optimalTilingFeatures &
          vk::FormatFeatureFlagBits::eSampledImageFilterLinear))
      throw std::runtime_error(
          "texture image format does not support linear blitting!");

    auto &PltMgn = gEng::PlatformHandler::getInstance();
    auto [SSTC, CmdBuff] = PltMgn.getSSTC();

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
          descriptorSets[i], 0,      0, vk::DescriptorType::eUniformBuffer,
          nullptr,           BufInfo};
      vk::WriteDescriptorSet ImgWrite{descriptorSets[i],
                                      1,
                                      0,
                                      vk::DescriptorType::eCombinedImageSampler,
                                      ImgInfo,
                                      nullptr};

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
    auto &PltMgn = gEng::PlatformHandler::getInstance();
    auto [SSTC, CmdBuff] = PltMgn.getSSTC();
    CmdBuff.copyBuffer(Src, Dst, vk::BufferCopy{{}, {}, Size});
  }

  // Creates new swap chain when smth goes wrong or resizing.
  void recreateSwapchain() {
    // Wait till proper size.
    m_Window.updExtent();

    m_device.waitIdle();
    auto &PltMgn = gEng::PlatformHandler::getInstance();
    Chains = gEng::ChainsManager{PltMgn, m_Window};
    fillFromChainManager();

    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
  }

  void updateUniformBuffer(uint32_t CurrImg) {
    static auto const StartTime = std::chrono::high_resolution_clock::now();

    auto const CurrTime = std::chrono::high_resolution_clock::now();
    float const Time =
        std::chrono::duration<float, std::chrono::seconds::period>(CurrTime -
                                                                   StartTime)
            .count();
    UniformBufferObject ubo{
        .model = glm::rotate(glm::mat4(1.f), Time * glm::radians(90.f),
                             glm::vec3(0.f, 0.f, 1.f)),
        .view = glm::lookAt(glm::vec3(2.f, 2.f, 2.f), glm::vec3(0.f, 0.f, 0.f),
                            glm::vec3(0.f, 0.f, 1.f)),
        .proj =
            glm::perspective(glm::radians(45.f),
                             m_swapchainExtent.width /
                                 static_cast<float>(m_swapchainExtent.height),
                             0.1f, 10.f)};
    ubo.proj[1][1] *= -1; // because GLM designed for OpenGL.

    auto const &Memory = uniformBuffersMemory[CurrImg];
    void *Data = m_device.mapMemory(Memory, 0, sizeof(ubo));
    memcpy(Data, &ubo, sizeof(ubo));
    m_device.unmapMemory(Memory);
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
    std::vector<vk::Semaphore> sem1;
    std::vector<vk::Semaphore> sem2;
    std::vector<vk::Fence> fence;

    for (int i = 0; i != MAX_FRAMES_IN_FLIGHT; ++i) {
      sem1.push_back(m_device.createSemaphore({}));
      sem2.push_back(m_device.createSemaphore({}));
      fence.push_back(
          m_device.createFence({vk::FenceCreateFlagBits::eSignaled}));
    }
    m_imageAvailableSemaphore = std::move(sem1);
    m_renderFinishedSemaphore = std::move(sem2);
    m_inFlightFence = std::move(fence);
  }

  void createCommandBuffers() {
    uint32_t const Frames = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    m_commandBuffers = m_device.allocateCommandBuffers(
        {m_commandPool, vk::CommandBufferLevel::ePrimary, Frames});
  }

  void recordCommandBuffer(vk::CommandBuffer CmdBuff, uint32_t ImgIdx) {
    // TODO. what differencies btwn ImgIdx and currentFrame?
    vk::CommandBufferBeginInfo CmdBeginInfo{};
    if (CmdBuff.begin(&CmdBeginInfo) != vk::Result::eSuccess)
      throw std::runtime_error("Fail to begin cmd buff");

    // Order of clear values should be indentical to the order of attachments.
    std::array<vk::ClearValue, 2> ClearValues{};
    std::array ColorVal = {0.0f, 0.0f, 0.0f, 1.0f};
    ClearValues[0].setColor(ColorVal);
    ClearValues[1].setDepthStencil({1.0f, 0});

    CmdBuff.beginRenderPass({m_renderPass,
                             m_swapChainFramebuffers[ImgIdx],
                             {{}, m_swapchainExtent},
                             ClearValues},
                            vk::SubpassContents::eInline);
    CmdBuff.bindPipeline(vk::PipelineBindPoint::eGraphics, m_graphicsPipeline);

    vk::DeviceSize Offsets{0};
    CmdBuff.bindVertexBuffers(0, VertexBuffer, Offsets);
    CmdBuff.bindIndexBuffer(IndexBuffer, 0, vk::IndexType::eUint32);

    CmdBuff.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                               m_pipelineLayout, 0,
                               descriptorSets[currentFrame], nullptr);

    // vertexCount, instanceCount, fitstVertex, firstInstance
    CmdBuff.drawIndexed(Indices.size(), 1, 0, 0, 0);

    CmdBuff.endRenderPass();
    CmdBuff.end();
  }

  void createCommandPool() {
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(m_physicalDevice);
    m_commandPool = m_device.createCommandPool(
        {vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
         queueFamilyIndices.GraphicsFamily.value()});
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
      // TODO. maybe updExtent
      auto [Width, Height] = m_Window.getExtent();
      uint32_t const RealW =
          std::clamp(Width, Capabilities.minImageExtent.width,
                     Capabilities.maxImageExtent.width);
      uint32_t const RealH =
          std::clamp(Height, Capabilities.minImageExtent.height,
                     Capabilities.maxImageExtent.height);
      return {RealW, RealH};
    }
  }

  // Checks if device is suitable for our extensions and purposes.
  bool isDeviceSuitable(vk::PhysicalDevice const &Device) {
    // name, type, supported Vulkan version can be quired via
    // GetPhysicalDeviceProperties.
    // vk::PhysicalDeviceProperties DeviceProps = Device.getProperties();

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

    m_debugMessenger =
        m_instance.createDebugUtilsMessengerEXT(populateDebugMessengerInfo());
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
      CreateInfo.setPNext(&DebugCreateInfo);

    m_instance = vk::createInstance(CreateInfo);
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
    uint32_t Frame{0};

    auto FrameLimit = EH.getFramesLimit();

    while (!m_Window.isShouldClose() &&
           !(FrameLimit && Frame++ > FrameLimit.value())) {
      glfwPollEvents();
      drawFrame();
    }

    m_device.waitIdle();
  }

  uint32_t currentFrame = 0;
  void drawFrame() {
    auto &PltMgn = gEng::PlatformHandler::getInstance();
    vk::Device Dev = PltMgn.get<vk::Device>();

    if (Dev.waitForFences(m_inFlightFence[currentFrame], VK_TRUE, UINT64_MAX) !=
        vk::Result::eSuccess)
      throw std::runtime_error("Fail to wait fance");

    auto [Res, ImgIdx] = Dev.acquireNextImageKHR(
        m_swapchain, UINT64_MAX, m_imageAvailableSemaphore[currentFrame], {});
    if (Res == vk::Result::eErrorOutOfDateKHR)
      recreateSwapchain();
    else if (Res != vk::Result::eSuccess && Res != vk::Result::eSuboptimalKHR)
      throw std::runtime_error("failed to acquire swap chain image!");

    updateUniformBuffer(currentFrame);

    // Only reset the fence if we are submitting work
    Dev.resetFences(m_inFlightFence[currentFrame]);
    m_commandBuffers[currentFrame].reset();
    recordCommandBuffer(m_commandBuffers[currentFrame], ImgIdx);

    vk::PipelineStageFlags WaitStage =
        vk::PipelineStageFlagBits::eColorAttachmentOutput;

    m_graphicsQueue.submit(
        vk::SubmitInfo{m_imageAvailableSemaphore[currentFrame], WaitStage,
                       m_commandBuffers[currentFrame],
                       m_renderFinishedSemaphore[currentFrame]},
        m_inFlightFence[currentFrame]);

    if (m_presentQueue.presentKHR(
            {m_renderFinishedSemaphore[currentFrame], m_swapchain, ImgIdx}) !=
        vk::Result::eSuccess)
      throw std::runtime_error("Fail to wait fance");

    if (Res == vk::Result::eErrorOutOfDateKHR ||
        Res == vk::Result::eSuboptimalKHR || IsResized) {
      recreateSwapchain();
      IsResized = false;
    } else if (Res != vk::Result::eSuccess)
      throw std::runtime_error("failed to present swap chain image!");

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
  }

  void cleanupSwapchain() {
    Chains.cleanup(m_device);
    for (auto &&UniBuff : uniformBuffers)
      m_device.destroyBuffer(UniBuff);
    for (auto &&UniBuffMem : uniformBuffersMemory)
      m_device.freeMemory(UniBuffMem);
  }

  void cleanup() {
    cleanupSwapchain();

    m_device.destroySampler(textureSampler);
    m_device.destroyImageView(textureImageView);

    m_device.destroyImage(textureImage);
    m_device.freeMemory(textureImageMemory);

    m_device.destroyDescriptorPool(descriptorPool);
    m_device.destroyDescriptorSetLayout(descriptorSetLayout);
    m_device.destroyBuffer(VertexBuffer);
    m_device.freeMemory(VertexBufferMemory);
    m_device.destroyBuffer(IndexBuffer);
    m_device.freeMemory(IndexBufferMemory);

    for (auto &&Sem : m_renderFinishedSemaphore)
      m_device.destroySemaphore(Sem);
    for (auto &&Sem : m_imageAvailableSemaphore)
      m_device.destroySemaphore(Sem);
    for (auto &&Fence : m_inFlightFence)
      m_device.destroyFence(Fence);

    m_device.destroyCommandPool(m_commandPool);
  }
};

int main(int argc, char *argv[]) {
#ifndef NDEBUG
  std::cout << "Debug\n";
#endif // Debug

  // TODO think about this interface in singleton
  // gEng::SysEnv::init(argc, argv);
  gEng::SysEnv &EH = gEng::SysEnv::getInstance();
  EH.init(argc, argv);
  if (EH.getHelp())
    return EXIT_SUCCESS;

  HelloTriangleApplication app{EH};
  try {
    app.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
