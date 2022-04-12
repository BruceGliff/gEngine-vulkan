#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "EnvHandler.h"
#include "decoy/decoy.h"
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
#include <vector>

// This is some hack for a callback handling.
// VkDebugUtilsMessengerCreateInfoEXT struct should be passed to
// vkCreateDebugUtilsMessengerEXT, but as this function is an extension, it is
// not automatically loaded. So we have to look up by ourselfes via
// vkGetInstanceProcAddr.
VkResult createDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkDebugUtilsMessengerEXT *pDebugMessenger) {
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr)
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
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
  GLFWwindow * m_Window {};

  unsigned const m_Width {1600};
  unsigned const m_Height {900};

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
  VkInstance m_instance {};
  // Member for a call back handling.
  VkDebugUtilsMessengerEXT m_debugMessenger{};
  // Preferable device. Will be freed automatically.
  VkPhysicalDevice m_physicalDevice{VK_NULL_HANDLE};
  // Logical device.
  VkDevice m_device{};
  // Queue automatically created with logical device, but we need to create a
  // handles. And queues automatically destroyed within device.
  VkQueue m_graphicsQueue{};
  // Presentation queue.
  VkQueue m_presentQueue{};
  // Surface to be rendered in.
  // It is actually platform-dependent, but glfw uses function which fills
  // platform-specific structures by itself.
  VkSurfaceKHR m_surface{};
  // swapchain;
  VkSwapchainKHR m_swapchain{};

  std::vector<VkImage> m_swapchainImages;
  VkFormat m_swapchainImageFormat;
  VkExtent2D m_swapchainExtent;
  // ImageView used to specify how to treat VkImage.
  // For each VkImage we create VkImageView.
  std::vector<VkImageView> m_swapchainImageViews;

  VkRenderPass m_renderPass;
  // Used for an uniform variable.
  VkPipelineLayout m_pipelineLayout;

  VkPipeline m_graphicsPipeline;

  // A framebuffer object references all of the VkImageView objects.
  std::vector<VkFramebuffer> m_swapChainFramebuffers;

  VkCommandPool m_commandPool;

  std::vector<VkCommandBuffer> m_commandBuffers;

  std::vector<VkSemaphore> m_imageAvailableSemaphore;
  std::vector<VkSemaphore> m_renderFinishedSemaphore;

  std::vector<VkFence> m_inFlightFence;

  bool framebufferResized = false;

  VkBuffer VertexBuffer;
  VkDeviceMemory VertexBufferMemory;

  VkBuffer IndexBuffer;
  VkDeviceMemory IndexBufferMemory;

  std::vector<VkBuffer> uniformBuffers;
  std::vector<VkDeviceMemory> uniformBuffersMemory;

  VkDescriptorSetLayout descriptorSetLayout;

  VkDescriptorPool descriptorPool;
  std::vector<VkDescriptorSet> descriptorSets;

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
    createFramebuffers();
    createCommandPool();
    createVertexBuffer();
    createIndexBuffer();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
    createSyncObjects();
  }

  void createDescriptorSets() {
    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT,
                                               descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
        .pSetLayouts = layouts.data()};

    descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    if (vkAllocateDescriptorSets(m_device, &allocInfo, descriptorSets.data()) !=
        VK_SUCCESS)
      throw std::runtime_error("failed to allocate descriptor sets");

    for (size_t i = 0; i != MAX_FRAMES_IN_FLIGHT; ++i) {
      VkDescriptorBufferInfo bufferInfo{.buffer = uniformBuffers[i],
                                        .offset = 0,
                                        .range = sizeof(UniformBufferObject)};

      VkWriteDescriptorSet descriptorWrite{
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = descriptorSets[i],
          .dstBinding = 0,
          .dstArrayElement = 0,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          .pImageInfo = nullptr,
          .pBufferInfo = &bufferInfo,
          .pTexelBufferView = nullptr};

      vkUpdateDescriptorSets(m_device, 1, &descriptorWrite, 0, nullptr);
    }
  }

  void createDescriptorPool() {
    VkDescriptorPoolSize poolSize{
        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)};

    VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize};

    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &descriptorPool) !=
        VK_SUCCESS)
      throw std::runtime_error("failed to create descriptor pool");
  }

  void createUniformBuffers() {
    VkDeviceSize bufferSize = sizeof(UniformBufferObject);

    uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i != MAX_FRAMES_IN_FLIGHT; ++i) {
      createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   uniformBuffers[i], uniformBuffersMemory[i]);
    }
  }

  void createDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding uboLayoutBinding{
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .pImmutableSamplers = nullptr};

    VkDescriptorSetLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &uboLayoutBinding};

    if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr,
                                    &descriptorSetLayout) != VK_SUCCESS)
      throw std::runtime_error("failed to create descriptor set layout");
  }

  // Rainbow triangle description.
  std::vector<Vertex> const Vertices = {{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
                                        {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
                                        {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
                                        {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}};
  void createVertexBuffer() {
    VkDeviceSize BuffSize = sizeof(Vertex) * Vertices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(BuffSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer, stagingBufferMemory);

    void *data;
    vkMapMemory(m_device, stagingBufferMemory, 0, BuffSize, 0, &data);
    memcpy(data, Vertices.data(), (size_t)BuffSize);
    vkUnmapMemory(m_device, stagingBufferMemory);

    createBuffer(
        BuffSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VertexBuffer, VertexBufferMemory);
    copyBuffer(stagingBuffer, VertexBuffer, BuffSize);

    vkDestroyBuffer(m_device, stagingBuffer, nullptr);
    vkFreeMemory(m_device, stagingBufferMemory, nullptr);
  }

  std::vector<uint32_t> const Indices = {0, 1, 2, 2, 3, 0};
  void createIndexBuffer() {
    VkDeviceSize bufferSize = sizeof(uint32_t) * Indices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer, stagingBufferMemory);

    void *data;
    vkMapMemory(m_device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, Indices.data(),
           bufferSize); // TODO why just data == Indices.data()?
    vkUnmapMemory(m_device, stagingBufferMemory);

    createBuffer(
        bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, IndexBuffer, IndexBufferMemory);

    copyBuffer(stagingBuffer, IndexBuffer, bufferSize);

    vkDestroyBuffer(m_device, stagingBuffer, nullptr);
    vkFreeMemory(m_device, stagingBufferMemory, nullptr);
  }

  void copyBuffer(VkBuffer srcBuff, VkBuffer dstBuff, VkDeviceSize Size) {
    // TODO. Maybe separate command pool is to be created for these kinds of
    // short-lived buffers, because the implementation may be able to apply
    // memory allocation optimizations. VK_COMMAND_POOL_CREATE_TRANSIENT_BIT for
    // commandPool generation in that case.
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

    VkBufferCopy copyReg{.srcOffset = 0, .dstOffset = 0, .size = Size};
    vkCmdCopyBuffer(commandBuffer, srcBuff, dstBuff, 1, &copyReg);
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                            .commandBufferCount = 1,
                            .pCommandBuffers = &commandBuffer};

    vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    // TODO sync may be done with Fences for scheduling multiple tranfers
    // simultaneously.
    vkQueueWaitIdle(m_graphicsQueue);

    vkFreeCommandBuffers(m_device, m_commandPool, 1, &commandBuffer);
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
    createFramebuffers();

    createFramebuffers();
    createUniformBuffers();
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

  void createBuffer(VkDeviceSize Size, VkBufferUsageFlags Usage,
                    VkMemoryPropertyFlags Properties, VkBuffer /*OUT*/ &Buffer,
                    VkDeviceMemory /*OUT*/ &Memory) {
    // TODO for transfering VK_QUEUE_TRANSFER_BIT is needed, but it included in
    // VK_QUEUE_GRAPHICS_BIT or COMPUTE_BIT. But it would be nice to create
    // queue family specially with TRANSFER_BIT.
    VkBufferCreateInfo bufferInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                  .size = Size,
                                  .usage = Usage,
                                  .sharingMode = VK_SHARING_MODE_EXCLUSIVE};

    if (vkCreateBuffer(m_device, &bufferInfo, nullptr, &Buffer) != VK_SUCCESS) {
      throw std::runtime_error("failed to create buffer");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(m_device, Buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = memRequirements.size,
        .memoryTypeIndex =
            findMemoryType(memRequirements.memoryTypeBits, Properties)};

    if (vkAllocateMemory(m_device, &allocInfo, nullptr, &Memory) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to allocate buffer memory");
    }

    vkBindBufferMemory(m_device, Buffer, Memory, 0);
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
                                 m_commandBuffers.data()) != VK_SUCCESS) {
      throw std::runtime_error("failed to allocate command buffers!");
    }
  }

  void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
    VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = 0,
        .pInheritanceInfo = nullptr};

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
      throw std::runtime_error("failed to begin recording command buffer!");
    }

    VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
    VkRenderPassBeginInfo renderPassInfo{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = m_renderPass,
        .framebuffer = m_swapChainFramebuffers[imageIndex],
        .clearValueCount = 1,
        .pClearValues = &clearColor};
    // This part is to avoid "nested designators extension".
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = m_swapchainExtent;

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo,
                         VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      m_graphicsPipeline);

    VkBuffer vertexBuffers[] = {VertexBuffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(commandBuffer, IndexBuffer, 0, VK_INDEX_TYPE_UINT32);

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_pipelineLayout, 0, 1,
                            &descriptorSets[currentFrame], 0, nullptr);
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

    VkCommandPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, // Optional
        .queueFamilyIndex = queueFamilyIndices.graphicsFamily.value(),
    };

    if (vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_commandPool) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create command pool!");
    }
  }

  void createFramebuffers() {
    m_swapChainFramebuffers.resize(m_swapchainImageViews.size());
    int idx{0};
    for (auto &&ImageView : m_swapchainImageViews) {

      VkFramebufferCreateInfo framebufferInfo{
          .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
          .renderPass = m_renderPass,
          .attachmentCount = 1,
          .pAttachments = &ImageView,
          .width = m_swapchainExtent.width,
          .height = m_swapchainExtent.height,
          .layers = 1};

      if (vkCreateFramebuffer(m_device, &framebufferInfo, nullptr,
                              &m_swapChainFramebuffers[idx++]) != VK_SUCCESS) {
        throw std::runtime_error("failed to create framebuffer!");
      }
    }
  }

  void createRenderPass() {
    VkAttachmentDescription colorAttachment{
        .format = m_swapchainImageFormat,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR};

    VkAttachmentReference colorAttachmentRef{
        .attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass{.pipelineBindPoint =
                                     VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 .colorAttachmentCount = 1,
                                 .pColorAttachments = &colorAttachmentRef};

    VkSubpassDependency dependency{
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT};

    VkRenderPassCreateInfo renderPassInfo{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &colorAttachment,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &dependency};

    if (vkCreateRenderPass(m_device, &renderPassInfo, nullptr, &m_renderPass) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create render pass!");
    }
  }

  void createGraphicPipeline() {
    fs::path ShadersPath{EH};
    ShadersPath /= "shaders/";
    Shader VShader{(ShadersPath / "basic.vert.spv").string()};
    Shader FShader{(ShadersPath / "basic.frag.spv").string()};

    VkShaderModule vertShaderModule = createShaderModule(VShader);
    VkShaderModule fragShaderModule = createShaderModule(FShader);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = vertShaderModule,
        .pName = "main"};

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = fragShaderModule,
        .pName = "main"};

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo,
                                                      fragShaderStageInfo};

    // Fill Vertex binding info.
    auto bindDescr = Vertex::getBindDescription();
    auto attrDescr = Vertex::getAttrDescription();
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &bindDescr,
        .vertexAttributeDescriptionCount = attrDescr.size(),
        .pVertexAttributeDescriptions = attrDescr.data()};

    // The rules how verticies will be treated(lines, points, triangles..)
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE};

    // Read this in FixedFunction part.
    VkViewport viewport{.x = 0.0f,
                        .y = 0.0f,
                        .width = (float)m_swapchainExtent.width,
                        .height = (float)m_swapchainExtent.height,
                        .minDepth = 0.0f,
                        .maxDepth = 1.0f};

    VkRect2D scissor{.offset = {0, 0}, .extent = m_swapchainExtent};

    VkPipelineViewportStateCreateInfo viewportState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor};

    VkPipelineRasterizationStateCreateInfo rasterizer{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .depthBiasConstantFactor = 0.0f, // Optional
        .depthBiasClamp = 0.0f,          // Optional
        .depthBiasSlopeFactor = 0.0f,    // Optional
        .lineWidth = 1.0f,
    };

    VkPipelineMultisampleStateCreateInfo multisampling{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = VK_FALSE,
        .minSampleShading = 1.0f,          // Optional
        .pSampleMask = nullptr,            // Optional
        .alphaToCoverageEnable = VK_FALSE, // Optional
        .alphaToOneEnable = VK_FALSE       // Optional
    };

    VkPipelineColorBlendAttachmentState colorBlendAttachment{
        .blendEnable = VK_FALSE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,  // Optional
        .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO, // Optional
        .colorBlendOp = VK_BLEND_OP_ADD,             // Optional
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,  // Optional
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO, // Optional
        .alphaBlendOp = VK_BLEND_OP_ADD,             // Optional
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f; // Optional
    colorBlending.blendConstants[1] = 0.0f; // Optional
    colorBlending.blendConstants[2] = 0.0f; // Optional
    colorBlending.blendConstants[3] = 0.0f; // Optional

    // There are some states of pipeline that can be changed dynamicly.
    VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT,
                                      VK_DYNAMIC_STATE_LINE_WIDTH};

    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    // Pipeline layout is for using uniform values.
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptorSetLayout};

    if (vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr,
                               &m_pipelineLayout) != VK_SUCCESS)
      throw std::runtime_error("failed to create pipeline layout!");

    VkGraphicsPipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = shaderStages,
        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = nullptr, // Optional
        .pColorBlendState = &colorBlending,
        .pDynamicState = nullptr, // Optional
        .layout = m_pipelineLayout,
        .renderPass = m_renderPass,
        .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
    };

    if (vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                  nullptr, &m_graphicsPipeline) != VK_SUCCESS) {
      throw std::runtime_error("failed to create graphics pipeline!");
    }

    vkDestroyShaderModule(m_device, fragShaderModule, nullptr);
    vkDestroyShaderModule(m_device, vertShaderModule, nullptr);
  }

  VkShaderModule createShaderModule(Shader const &Ker) {
    VkShaderModuleCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = Ker.getSize(),
        .pCode = Ker.getRawData()};

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(m_device, &createInfo, nullptr, &shaderModule) !=
        VK_SUCCESS)
      throw std::runtime_error("failed to create shader module!");

    return shaderModule;
  }

  // Finds right type of memory to use.
  uint32_t findMemoryType(uint32_t typeFilter,
                          VkMemoryAllocateFlags Properties) {
    VkPhysicalDeviceMemoryProperties MemProperties;
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &MemProperties);

    for (uint32_t i = 0; i < MemProperties.memoryTypeCount; i++)
      if ((typeFilter & (1 << i)) &&
          (MemProperties.memoryTypes[i].propertyFlags & Properties) ==
              Properties)
        return i;

    throw std::runtime_error("failed to find suitable memory type!");
  }

  void createImageViews() {
    m_swapchainImageViews.resize(m_swapchainImages.size());
    {
      int i = 0;
      for (VkImage const &Image : m_swapchainImages) {
        VkImageViewCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = Image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = m_swapchainImageFormat,
            .components = {VK_COMPONENT_SWIZZLE_IDENTITY,
                           VK_COMPONENT_SWIZZLE_IDENTITY,
                           VK_COMPONENT_SWIZZLE_IDENTITY,
                           VK_COMPONENT_SWIZZLE_IDENTITY}, // rgba
        };
        // subresourceRange defines what is image purpose and which part of
        // image should be accessed. Image will be used as color target without
        // mipmapping levels or multiple layers.
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(m_device, &createInfo, nullptr,
                              &m_swapchainImageViews[i]) != VK_SUCCESS)
          throw std::runtime_error("failed to create image views!");
        ++i;
      }
    }
  }

  void createSwapchain() {
    SwapchainSupportDetails swapchainSupport =
        querySwapchainSupport(m_physicalDevice);

    VkSurfaceFormatKHR surfaceFormat =
        chooseSwapSurfaceFormat(swapchainSupport.formats);
    VkPresentModeKHR presentMode =
        chooseSwapPresentMode(swapchainSupport.presentModes);
    VkExtent2D extent = chooseSwapExtent(swapchainSupport.capabilities);

    // min images count in swap chain(plus one).
    uint32_t imageCount = swapchainSupport.capabilities.minImageCount + 1;
    // Not to go throught max ImageCount (0 means no upper-bounds).
    if (swapchainSupport.capabilities.maxImageCount > 0 &&
        imageCount > swapchainSupport.capabilities.maxImageCount)
      imageCount = swapchainSupport.capabilities.maxImageCount;

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = m_surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    QueueFamilyIndices indices = findQueueFamilies(m_physicalDevice);
    uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(),
                                     indices.presentFamily.value()};

    // Next, we need to specify how to handle swap chain images that will be
    // used across multiple queue families.
    if (indices.graphicsFamily != indices.presentFamily) {
      createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
      createInfo.queueFamilyIndexCount = 2;
      createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
      createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
      createInfo.queueFamilyIndexCount = 0;     // Optional
      createInfo.pQueueFamilyIndices = nullptr; // Optional
    }
    createInfo.preTransform = swapchainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(m_device, &createInfo, nullptr, &m_swapchain) !=
        VK_SUCCESS)
      throw std::runtime_error("failed to create swap chain!");

    vkGetSwapchainImagesKHR(m_device, m_swapchain, &imageCount, nullptr);
    m_swapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &imageCount,
                            m_swapchainImages.data());
    m_swapchainImageFormat = surfaceFormat.format;
    m_swapchainExtent = extent;
  }

  void createSurface() {
    if (glfwCreateWindowSurface(m_instance, m_Window, nullptr, &m_surface) !=
        VK_SUCCESS)
      throw std::runtime_error("failed to create window surface!");
  }

  void createLogicalDevice() {
    QueueFamilyIndices indices = findQueueFamilies(m_physicalDevice);

    // Each queue family has to have own VkDeviceQueueCreateInfo.
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos{};
    // This is the worst way of doing it. Rethink!
    std::set<uint32_t> setUniqueInfoIdx = {indices.graphicsFamily.value(),
                                           indices.presentFamily.value()};

    float queuePriority{1.f};
    for (uint32_t queueFamily : setUniqueInfoIdx) {
      VkDeviceQueueCreateInfo queueCreateInfo{
          .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
          .queueFamilyIndex = queueFamily,
          .queueCount = 1,
          .pQueuePriorities = &queuePriority,
      };
      queueCreateInfos.push_back(std::move(queueCreateInfo));
    }

    // Right now we do not need this.
    VkPhysicalDeviceFeatures deviceFeatures{};

    // The main deviceIndo structure.
    VkDeviceCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
        .pQueueCreateInfos = queueCreateInfos.data(),
        .enabledLayerCount = 0,
        .enabledExtensionCount =
            static_cast<uint32_t>(m_deviceExtensions.size()),
        .ppEnabledExtensionNames = m_deviceExtensions.data(),
        .pEnabledFeatures = &deviceFeatures,
    };
    // To be compatible with older implementations, as new Vulcan version
    // does not require ValidaionLayers
    if (m_enableValidationLayers) {
      createInfo.enabledLayerCount =
          static_cast<uint32_t>(m_validationLayers.size());
      createInfo.ppEnabledLayerNames = m_validationLayers.data();
    }

    if (vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &m_device) !=
        VK_SUCCESS)
      throw std::runtime_error("failed to create logical device!");

    // We can use the vkGetDeviceQueue function to retrieve queue handles for
    // each queue family. The parameters are the logical device, queue family,
    // queue index and a pointer to the variable to store the queue handle in.
    // Because we're only creating a single queue from this family, we'll simply
    // use index 0.
    vkGetDeviceQueue(m_device, indices.graphicsFamily.value(), 0,
                     &m_graphicsQueue);
    vkGetDeviceQueue(m_device, indices.presentFamily.value(), 0,
                     &m_presentQueue);
  }

  void pickPhysicalDevice() {
    uint32_t deviceCount{0};
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr);
    if (!deviceCount)
      throw std::runtime_error{"No physical device supports Vulkan!"};

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, devices.data());

    for (const auto &device : devices)
      if (isDeviceSuitable(device)) {
        m_physicalDevice = device;
        break;
      }

    if (!m_physicalDevice)
      throw std::runtime_error("failed to find a suitable GPU!");
  }

  // Checks which queue families are supported by the device and which one of
  // these supports the commands that we want to use.
  struct QueueFamilyIndices {
    // optional just because we may be want to select GPU with some family, but
    // it is not strictly necessary.
    std::optional<uint32_t> graphicsFamily{};
    // Not every device can support presentation of the image, so need to
    // check that divece has proper family queue.
    std::optional<uint32_t> presentFamily{};

    bool isComplete() {
      return graphicsFamily.has_value() && presentFamily.has_value();
    }
  };

  // Swapchain requires more details to be checked.
  // - basic surface capabilities.
  // - surface format (pixel format, color space).
  // - available presentation mode.
  struct SwapchainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
  };

  // This section covers how to query the structs that include this information.
  SwapchainSupportDetails querySwapchainSupport(VkPhysicalDevice device) {
    SwapchainSupportDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, m_surface,
                                              &details.capabilities);

    // The next step is about querying the supported surface formats.
    uint32_t formatCount{0};
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, m_surface, &formatCount,
                                         nullptr);
    if (formatCount != 0) {
      details.formats.resize(
          formatCount); // can it fill an array and return formatCount at once?
      vkGetPhysicalDeviceSurfaceFormatsKHR(device, m_surface, &formatCount,
                                           details.formats.data());
    }

    // And finally, querying the supported presentation modes.
    uint32_t presentModeCount{0};
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, m_surface,
                                              &presentModeCount, nullptr);
    if (presentModeCount != 0) {
      details.presentModes.resize(presentModeCount);
      vkGetPhysicalDeviceSurfacePresentModesKHR(
          device, m_surface, &presentModeCount, details.presentModes.data());
    }

    return details;
  }

  QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
    QueueFamilyIndices indices;
    // Logic to find queue family indices to populate struct with.
    uint32_t queueFamilyCount{0};
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                             nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                             queueFamilies.data());
    {
      int i{0};
      for (const auto &queueFamily : queueFamilies) {
        // For better performance one queue famili has to support all requested
        // queues at once, but we also can treat them as different families for
        // unified approach.
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
          indices.graphicsFamily = i;

        // Checks for presentation family support.
        VkBool32 presentSupport{false};
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, m_surface,
                                             &presentSupport);
        if (presentSupport)
          indices.presentFamily = i;

        // Not quite sure why the hell we need this early-break.
        if (indices.isComplete())
          break;
        i++;
      }
    }
    return indices;
  }

  // We want to select format.
  VkSurfaceFormatKHR chooseSwapSurfaceFormat(
      std::vector<VkSurfaceFormatKHR> const &availableFormats) {
    // Some words in Swap chain part:
    // https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Swap_chain
    for (auto const &availableFormat : availableFormats)
      if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
          availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        return availableFormat;

    std::cout << "The best format unavailable.\nUse available one.\n";
    return availableFormats[0];
  }

  // Presentation mode represents the actual conditions for showing images to
  // the screen.
  VkPresentModeKHR chooseSwapPresentMode(
      std::vector<VkPresentModeKHR> const &availablePresentModes) {
    for (auto const &availablePresentMode : availablePresentModes)
      // simillar to FIFO, but when queue is full - replaces images.
      // called triplebuffered.
      if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
        return availablePresentMode;
    // this is vertical sync. images are shown from the queue and stored in
    // queue. if queue is full - wait.
    return VK_PRESENT_MODE_FIFO_KHR;
  }

  // glfwWindows works with screen-coordinates. But Vulkan - with pixels.
  // And not always they are corresponsible with each other.
  // So we want to create a proper resolution.
  VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities) {
    if (capabilities.currentExtent.width != UINT32_MAX)
      return capabilities.currentExtent;
    else {
      int width{0}, height{0};
      glfwGetFramebufferSize(m_Window, &width, &height);

      VkExtent2D actualExtent = {static_cast<uint32_t>(width),
                                 static_cast<uint32_t>(height)};

      actualExtent.width =
          std::clamp(actualExtent.width, capabilities.minImageExtent.width,
                     capabilities.maxImageExtent.width);
      actualExtent.height =
          std::clamp(actualExtent.height, capabilities.minImageExtent.height,
                     capabilities.maxImageExtent.height);

      return actualExtent;
    }
  }

  // Checks if device is suitable for our extensions and purposes.
  bool isDeviceSuitable(VkPhysicalDevice device) {
    // name, type, supported Vulcan version can be quired via
    // GetPhysicalDeviceProperties.
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);

    // optional features like texture compressing, 64bit floating operations,
    // multiview-port and so one..
    VkPhysicalDeviceFeatures deviceFeatures;
    vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

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
    return findQueueFamilies(device).isComplete() &&
           checkDeviceExtensionSupport(device) &&
           swapchainSupport(querySwapchainSupport(device));
    // All three ckecks are different. WTF!
  }

  bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
    // Some bad code. Rethink!
    uint32_t extensionCount{0};
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                         nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                         availableExtensions.data());

    std::set<std::string> requiredExtensions(m_deviceExtensions.begin(),
                                             m_deviceExtensions.end());

    for (const auto &extension : availableExtensions) {
      requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
  }

  VkDebugUtilsMessengerCreateInfoEXT populateDebugMessengerInfo() {
    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
    createInfo.pUserData = nullptr; // Optional

    return createInfo;
  }

  void setupDebugMessenger() {
    if (!m_enableValidationLayers)
      return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo{populateDebugMessengerInfo()};

    if (createDebugUtilsMessengerEXT(m_instance, &createInfo, nullptr,
                                     &m_debugMessenger) != VK_SUCCESS)
      throw std::runtime_error("failed to set up debug messenger!");
  }

  std::vector<char const *> getRequiredExtensions() {
    uint32_t glfwExtensionCount = 0;
    const char **glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char *> extensions(glfwExtensions,
                                         glfwExtensions + glfwExtensionCount);

    // Addition for callback
    if (m_enableValidationLayers)
      extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    return extensions;
  }

  void createInstance() {
    if (m_enableValidationLayers && !checkValidationLayers())
      throw std::runtime_error{
          "Requestred validation layers are not available!"};

    // Some optional information to pass to driver
    // for some optimizations
    VkApplicationInfo appInfo {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "Hello triangle",
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName = "No Engine",
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion = VK_API_VERSION_1_0
    };

    std::vector<char const *> Extensions{getRequiredExtensions()};
    // Required struct tells the Vulkan driver whick global
    // extension and validation level to use
    VkInstanceCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = 0,
        .enabledExtensionCount = static_cast<uint32_t>(Extensions.size()),
        .ppEnabledExtensionNames = Extensions.data()};

    // It definetly has to be created before vkCreateInstane. BUT! Check for
    // vkDestroyInstance. Will it be valid there too?
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo =
        m_enableValidationLayers ? populateDebugMessengerInfo()
                                 : VkDebugUtilsMessengerCreateInfoEXT{};
    if (m_enableValidationLayers) {
      createInfo.enabledLayerCount =
          static_cast<uint32_t>(m_validationLayers.size());
      createInfo.ppEnabledLayerNames = m_validationLayers.data();

      // pNext is set to call debug messages for checking vkCreateInstance and
      // vkDestoryInstance.
      createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT *)&debugCreateInfo;
    }

    // TODO make macros vkSafeCall()
    if (vkCreateInstance(&createInfo, nullptr, &m_instance) != VK_SUCCESS) {
      throw std::runtime_error("failed to create instance!");
    }
    
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
    uint32_t LayersCount{0};
    vkEnumerateInstanceLayerProperties(&LayersCount, nullptr);
    std::vector<VkLayerProperties> AvailableLayers(LayersCount);
    vkEnumerateInstanceLayerProperties(&LayersCount, AvailableLayers.data());

    for (const char *layerName : m_validationLayers) {
      bool layerFound = false;

      for (const auto &layerProperties : AvailableLayers)
        if (strcmp(layerName, layerProperties.layerName) == 0) {
          layerFound = true;
          break;
        }

      if (!layerFound)
        return false;
    }
    return true;
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
    while (!glfwWindowShouldClose(m_Window)) {
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
    } else if (res != VK_SUCCESS) {
      throw std::runtime_error("failed to present swap chain image!");
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
  }

  void cleanupSwapchain() {
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
    // for (auto framebuffer : m_swapChainFramebuffers)
    //   vkDestroyFramebuffer(m_device, framebuffer, nullptr);
    // vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);
    // vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    // vkDestroyRenderPass(m_device, m_renderPass, nullptr);
    // for (auto imageView : m_swapchainImageViews)
    //   vkDestroyImageView(m_device, imageView, nullptr);
    // vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
    vkDestroyDevice(m_device, nullptr);
    if (m_enableValidationLayers)
      destroyDebugUtilsMessengerEXT(m_instance, m_debugMessenger, nullptr);
    vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
    vkDestroyInstance(m_instance, nullptr);

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
