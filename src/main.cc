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
  static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

  // TODO of course get rid of global code!.

  // An instance needed for connection between app and VkLibrary
  // And adds a detailes about app to the driver
  vk::Instance m_instance;
  // Preferable device. Will be freed automatically.
  vk::PhysicalDevice m_physicalDevice;
  // Logical device.
  vk::Device m_device;
  // Queue automatically created with logical device, but we need to create a
  // handles. And queues automatically destroyed within device.
  vk::Queue m_graphicsQueue;
  // Presentation queue.
  vk::Queue m_presentQueue;

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
    std::cout << G.getEntity<int>() << '\n';

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
