// To use designated initializers.
// #define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.hpp>

#include <GLFW/glfw3.h>

// BAD. just a placeholder
#include "lib/environment/ChainsManager.h"
#include "lib/environment/platform_handler.h"
#include "lib/image/Image.h"
#include "lib/model/Model.h"
#include "lib/uniform_buffer/UniformBuffer.hpp"

#include "shader/shader.h"
#include "vertex.h"

#include "gEng/environment.h"
#include "gEng/global.h"
#include "gEng/window.h"

#include <algorithm>
#include <cassert>
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

  std::optional<gEng::ChainsManager> Chains{};
  vk::SwapchainKHR m_swapchain{};

  vk::Extent2D m_swapchainExtent;
  // ImageView used to specify how to treat VkImage.
  // For each VkImage we create VkImageView.

  vk::RenderPass m_renderPass;

  vk::Pipeline m_graphicsPipeline;

  // A framebuffer object references all of the VkImageView objects.
  std::vector<vk::Framebuffer> m_swapChainFramebuffers;

  vk::CommandPool m_commandPool;

  std::vector<vk::CommandBuffer> m_commandBuffers;

  std::vector<vk::Semaphore> m_imageAvailableSemaphore;
  std::vector<vk::Semaphore> m_renderFinishedSemaphore;

  std::vector<vk::Fence> m_inFlightFence;

  std::optional<gEng::ModelVk> M;

  gEng::Image Img;

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
    m_swapchain = Chains->getSwapchain();
    m_swapchainExtent = Chains->getExtent();
    m_renderPass = Chains->getRPass();
    m_graphicsPipeline = Chains->getP();

    m_swapChainFramebuffers = Chains->getFrameBuffers();
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

    Chains.emplace(PltMgn, m_Window);
    fillFromChainManager();

    fs::path ImagePath{EH};
    ImagePath /= "assets/textures/viking_room.png";
    Img.setImg(ImagePath.generic_string());

    fs::path ModelPath{EH};
    ModelPath /= "assets/models/viking_room.obj";
    M.emplace(gEng::Model{ModelPath.generic_string()});

    Chains->getShader().connect(*M, Img);

    createCommandBuffers();
    createSyncObjects();
  }

  // Creates new swap chain when smth goes wrong or resizing.
  void recreateSwapchain() {
    // Wait till proper size.
    m_Window.updExtent();

    m_device.waitIdle();
    auto &PltMgn = gEng::PlatformHandler::getInstance();
    Chains.emplace(PltMgn, m_Window);
    fillFromChainManager();

    createCommandBuffers();
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

    M->bind(CmdBuff);
    Chains->getShader().bind(CmdBuff, currentFrame);
    M->draw(CmdBuff);

    CmdBuff.endRenderPass();
    CmdBuff.end();
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

    M->updateUniformBuffer(currentFrame,
                           m_swapchainExtent.width /
                               static_cast<float>(m_swapchainExtent.height));

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

  void cleanup() {
    Chains->cleanup(m_device);

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
