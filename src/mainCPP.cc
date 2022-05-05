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

  std::vector<Vertex> Vertices;
  std::vector<uint32_t> Indices;

  VkBuffer VertexBuffer;
  VkDeviceMemory VertexBufferMemory;

  VkBuffer IndexBuffer;
  VkDeviceMemory IndexBufferMemory;

  std::vector<VkBuffer> uniformBuffers;
  std::vector<VkDeviceMemory> uniformBuffersMemory;

  VkDescriptorSetLayout descriptorSetLayout;

  VkDescriptorPool descriptorPool;
  std::vector<VkDescriptorSet> descriptorSets;

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
    VkFormat colorFormat = m_swapchainImageFormat;

    createImage(m_swapchainExtent.width, m_swapchainExtent.height, 1,
                msaaSamples, colorFormat, VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, colorImage,
                colorImageMemory);
    colorImageView =
        createImageView(colorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
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
    VkFormat depthFormat = findDepthFormat();
    createImage(m_swapchainExtent.width, m_swapchainExtent.height, 1,
                msaaSamples, depthFormat, VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage,
                depthImageMemory);
    depthImageView =
        createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 1);

    // As I understand this part is optional as we will take care of this in the
    // render pass.
    transitionImageLayout(depthImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1);
  }

  VkFormat findDepthFormat() {
    return findSupportedFormat({VK_FORMAT_D32_SFLOAT,
                                VK_FORMAT_D32_SFLOAT_S8_UINT,
                                VK_FORMAT_D24_UNORM_S8_UINT},
                               VK_IMAGE_TILING_OPTIMAL,
                               VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
  }

  bool hasStencilComponent(VkFormat format) {
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
           format == VK_FORMAT_D24_UNORM_S8_UINT;
  }

  // Takes a lists of candidate formats from most desireable to the least
  // desirable and checks the first one is supported.
  VkFormat findSupportedFormat(std::vector<VkFormat> const &candidates,
                               VkImageTiling tiling,
                               VkFormatFeatureFlags features) {
    for (VkFormat format : candidates) {
      VkFormatProperties props;
      vkGetPhysicalDeviceFormatProperties(m_physicalDevice, format, &props);
      if (tiling == VK_IMAGE_TILING_LINEAR &&
          (props.linearTilingFeatures & features) == features)
        return format;
      else if (tiling == VK_IMAGE_TILING_OPTIMAL &&
               (props.optimalTilingFeatures & features) == features)
        return format;
    }

    throw std::runtime_error("failed to find supported format!");
  }

  void createTextureSampler() {
    VkPhysicalDeviceProperties properties{};
    // TODO retrieve properties once in program
    vkGetPhysicalDeviceProperties(m_physicalDevice, &properties);

    VkSamplerCreateInfo samplerInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_NEAREST, // VK_FILTER_LINEAR
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .mipLodBias = 0.f,
        .anisotropyEnable = VK_TRUE,
        .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_ALWAYS,
        .minLod = 0.f,
        .maxLod = static_cast<float>(mipLevels),
        .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
        .unnormalizedCoordinates = VK_FALSE};

    if (vkCreateSampler(m_device, &samplerInfo, nullptr, &textureSampler) !=
        VK_SUCCESS)
      throw std::runtime_error("failed to create texture sampler!");
  }

  VkImageView createImageView(VkImage image, VkFormat format,
                              VkImageAspectFlags aspectFlags,
                              uint32_t mipLevels) {
    VkImageViewCreateInfo viewInfo{.sType =
                                       VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                   .image = image,
                                   .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                   .format = format};
    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VkImageView imageView;
    if (vkCreateImageView(m_device, &viewInfo, nullptr, &imageView) !=
        VK_SUCCESS)
      throw std::runtime_error("failed to create texture image view!");

    return imageView;
  }

  void createTextureImageView() {
    textureImageView = createImageView(textureImage, VK_FORMAT_R8G8B8A8_SRGB,
                                       VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
  }

  void transitionImageLayout(VkImage image, VkFormat format,
                             VkImageLayout oldLayout, VkImageLayout newLayout,
                             uint32_t mipLevels) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkImageMemoryBarrier barrier{.sType =
                                     VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                 .oldLayout = oldLayout,
                                 .newLayout = newLayout,
                                 .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                 .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                 .image = image};
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = mipLevels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
      barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
      if (hasStencilComponent(format))
        barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
      // TODO this else causes validation error. But it is actually useless.
      // else
      //  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    }

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
        newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
      barrier.srcAccessMask = 0;
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
      destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
               newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
      destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
               newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
      barrier.srcAccessMask = 0;
      barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                              VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

      sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
      destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    } else
      throw std::invalid_argument("unsupported layout transition!");

    vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0,
                         nullptr, 0, nullptr, 1, &barrier);

    endSingleTimeCommands(commandBuffer);
  }

  void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width,
                         uint32_t height) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkBufferImageCopy region{
        .bufferOffset = 0, .bufferRowLength = 0, .bufferImageHeight = 0};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;

    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};
    vkCmdCopyBufferToImage(commandBuffer, buffer, image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    endSingleTimeCommands(commandBuffer);
  }

  void generateMipmaps(VkImage image, VkFormat imageFormat, uint32_t Width,
                       uint32_t Height, uint32_t mipLevels) {

    // Check if image format supports linear blitting.
    VkFormatProperties formatProps;
    vkGetPhysicalDeviceFormatProperties(m_physicalDevice, imageFormat,
                                        &formatProps);
    if (!(formatProps.optimalTilingFeatures &
          VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
      throw std::runtime_error(
          "texture image format does not support linear blitting!");

    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkImageMemoryBarrier barrier{.sType =
                                     VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                 .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                 .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                 .image = image};
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;

    int32_t mipWidth = static_cast<int32_t>(Width);
    int32_t mipHeight = static_cast<int32_t>(Height);
    for (uint32_t i = 1; i != mipLevels; ++i) {
      barrier.subresourceRange.baseMipLevel = i - 1;
      barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                           nullptr, 1, &barrier);

      VkImageBlit blit{};
      blit.srcOffsets[0] = {0, 0, 0};
      blit.srcOffsets[1] = {mipWidth, mipHeight, 1};
      blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      blit.srcSubresource.mipLevel = i - 1;
      blit.srcSubresource.baseArrayLayer = 0;
      blit.srcSubresource.layerCount = 1;
      blit.dstOffsets[0] = {0, 0, 0};
      blit.dstOffsets[1] = {mipWidth > 1 ? mipWidth / 2 : 1,
                            mipHeight > 1 ? mipHeight / 2 : 1, 1};
      blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      blit.dstSubresource.mipLevel = i;
      blit.dstSubresource.baseArrayLayer = 0;
      blit.dstSubresource.layerCount = 1;

      // must be submitted to a queue with graphics capability.
      vkCmdBlitImage(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                     VK_FILTER_LINEAR);

      barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                           0, nullptr, 1, &barrier);

      if (mipWidth > 1)
        mipWidth /= 2;
      if (mipHeight > 1)
        mipHeight /= 2;
    }

    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &barrier);

    endSingleTimeCommands(commandBuffer);
  }

  void createTextureImage() {
    fs::path ImagePath{EH};
    ImagePath /= "assets/textures/viking_room.png";
    image Image(ImagePath.generic_string());

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    uint32_t const ImageSize = Image.getSize();
    uint32_t const Width = Image.getWidth();
    uint32_t const Height = Image.getHeight();
    mipLevels =
        static_cast<uint32_t>(std::floor(std::log2(std::max(Width, Height)))) +
        1;
    createBuffer(ImageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer, stagingBufferMemory);
    void *data;
    vkMapMemory(m_device, stagingBufferMemory, 0, ImageSize, 0, &data);
    memcpy(data, Image.getRawData(), ImageSize);
    vkUnmapMemory(m_device, stagingBufferMemory);

    createImage(
        Width, Height, mipLevels, vk::SampleCountFlagBits::e1,
        VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

    transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB,
                          VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);
    copyBufferToImage(stagingBuffer, textureImage, Width, Height);
    // Transitioning to SHADER_READ_ONLY while generating mipmaps.
    generateMipmaps(textureImage, VK_FORMAT_R8G8B8A8_SRGB, Width, Height,
                    mipLevels);

    vkDestroyBuffer(m_device, stagingBuffer, nullptr);
    vkFreeMemory(m_device, stagingBufferMemory, nullptr);
  }

  void createImage(uint32_t width, uint32_t height, uint32_t mipLevels,
                   vk::SampleCountFlagBits numSample, VkFormat format,
                   VkImageTiling tiling, VkImageUsageFlags usage,
                   VkMemoryPropertyFlags properties, VkImage &image,
                   VkDeviceMemory &imageMemory) {

    // vk::ImageCreateInfo ImageInfo { {}, vk::ImageType::e2D, format,
    // vk::Extent3D{width, height, 1}, mipLevels, 1, numSample, tiling, usage,
    // vk::SharingMode::eExclusive, {} };

    VkImageCreateInfo imageInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .mipLevels = mipLevels,
        .arrayLayers = 1,
        .samples = static_cast<VkSampleCountFlagBits>(numSample),
        .tiling = tiling,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;

    if (vkCreateImage(m_device, &imageInfo, nullptr, &image) != VK_SUCCESS)
      throw std::runtime_error("failed to create image!");

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(m_device, image, &memRequirements);

    VkMemoryAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = memRequirements.size,
        .memoryTypeIndex =
            findMemoryType(memRequirements.memoryTypeBits, properties)};

    if (vkAllocateMemory(m_device, &allocInfo, nullptr, &imageMemory) !=
        VK_SUCCESS)
      throw std::runtime_error("failed to allocate image memory!");

    vkBindImageMemory(m_device, image, imageMemory, 0);
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
      VkDescriptorImageInfo imageInfo{
          .sampler = textureSampler,
          .imageView = textureImageView,
          .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      std::array<VkWriteDescriptorSet, 2> descriptorWrites{};
      descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrites[0].dstSet = descriptorSets[i];
      descriptorWrites[0].dstBinding = 0;
      descriptorWrites[0].dstArrayElement = 0;
      descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      descriptorWrites[0].descriptorCount = 1;
      descriptorWrites[0].pBufferInfo = &bufferInfo;

      descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrites[1].dstSet = descriptorSets[i];
      descriptorWrites[1].dstBinding = 1;
      descriptorWrites[1].dstArrayElement = 0;
      descriptorWrites[1].descriptorType =
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      descriptorWrites[1].descriptorCount = 1;
      descriptorWrites[1].pImageInfo = &imageInfo;

      vkUpdateDescriptorSets(m_device,
                             static_cast<uint32_t>(descriptorWrites.size()),
                             descriptorWrites.data(), 0, nullptr);
    }
  }

  void createDescriptorPool() {
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

    VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data()};

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

    VkDescriptorSetLayoutBinding samplerLayoutBinding{
        .binding = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        .pImmutableSamplers = nullptr};
    std::array<VkDescriptorSetLayoutBinding, 2> bindings = {
        uboLayoutBinding, samplerLayoutBinding};

    VkDescriptorSetLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data()};

    if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr,
                                    &descriptorSetLayout) != VK_SUCCESS)
      throw std::runtime_error("failed to create descriptor set layout");
  }

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
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();
    VkBufferCopy copyRegion{.size = Size};
    vkCmdCopyBuffer(commandBuffer, srcBuff, dstBuff, 1, &copyRegion);
    endSingleTimeCommands(commandBuffer);
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
            .count() *
        0;

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
        .queueFamilyIndex = queueFamilyIndices.GraphicsFamily.value(),
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
      // Order of the attachments is essential!
      // It is reverse from created in createRenderPass
      std::array<VkImageView, 3> attachments = {colorImageView, depthImageView,
                                                ImageView};
      VkFramebufferCreateInfo framebufferInfo{
          .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
          .renderPass = m_renderPass,
          .attachmentCount = static_cast<uint32_t>(attachments.size()),
          .pAttachments = attachments.data(),
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
    VkAttachmentDescription depthAttachment{
        .format = findDepthFormat(),
        .samples = static_cast<VkSampleCountFlagBits>(msaaSamples),
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkAttachmentReference depthAttachmentRef{
        .attachment = 1,
        .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkAttachmentDescription colorAttachment{
        .format = m_swapchainImageFormat,
        .samples = static_cast<VkSampleCountFlagBits>(msaaSamples),
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkAttachmentReference colorAttachmentRef{
        .attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkAttachmentDescription colorAttachmentResolve{
        .format = m_swapchainImageFormat,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR};

    VkAttachmentReference colorAttachmentResolveRef{
        .attachment = 2, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass{
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachmentRef,
        .pResolveAttachments = &colorAttachmentResolveRef,
        .pDepthStencilAttachment = &depthAttachmentRef};

    VkSubpassDependency dependency{
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT};

    std::array<VkAttachmentDescription, 3> attachments = {
        colorAttachment, depthAttachment, colorAttachmentResolve};
    VkRenderPassCreateInfo renderPassInfo{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = static_cast<uint32_t>(attachments.size()),
        .pAttachments = attachments.data(),
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
        .rasterizationSamples = static_cast<VkSampleCountFlagBits>(msaaSamples),
        .sampleShadingEnable = VK_TRUE,
        .minSampleShading = .2f,           // Optional
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

    VkPipelineDepthStencilStateCreateInfo depthStencil{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable = VK_FALSE,
        .front = {}, // Optional
        .back = {},  // Optional
        .minDepthBounds = 0.f,
        .maxDepthBounds = 1.f};

    VkGraphicsPipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = shaderStages,
        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = &depthStencil,
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
    for (uint32_t i = 0; i < m_swapchainImages.size(); i++)
      m_swapchainImageViews[i] =
          createImageView(m_swapchainImages[i], m_swapchainImageFormat,
                          VK_IMAGE_ASPECT_COLOR_BIT, 1);
  }

  void createSwapchain() {
    SwapchainSupportDetails swapchainSupport =
        querySwapchainSupport(m_physicalDevice);

    vk::SurfaceFormatKHR surfaceFormat =
        chooseSwapSurfaceFormat(swapchainSupport.formats);
    vk::PresentModeKHR presentMode =
        chooseSwapPresentMode(swapchainSupport.presentModes);
    vk::Extent2D extent = chooseSwapExtent(swapchainSupport.capabilities);

    // min images count in swap chain(plus one).
    uint32_t imageCount = swapchainSupport.capabilities.minImageCount + 1;
    // Not to go throught max ImageCount (0 means no upper-bounds).
    if (swapchainSupport.capabilities.maxImageCount > 0 &&
        imageCount > swapchainSupport.capabilities.maxImageCount)
      imageCount = swapchainSupport.capabilities.maxImageCount;

    vk::SwapchainCreateInfoKHR CreateInfo{
        {},
        m_surface,
        imageCount,
        surfaceFormat.format,
        surfaceFormat.colorSpace,
        extent,
        1,
        vk::ImageUsageFlagBits::eColorAttachment};

    QueueFamilyIndices Indices = findQueueFamilies(m_physicalDevice);
    uint32_t queueFamilyIndices[] = {Indices.GraphicsFamily.value(),
                                     Indices.PresentFamily.value()};

    // Next, we need to specify how to handle swap chain images that will be
    // used across multiple queue families.
    if (Indices.GraphicsFamily != Indices.PresentFamily) {
      CreateInfo.imageSharingMode = vk::SharingMode::eConcurrent;
      CreateInfo.queueFamilyIndexCount = 2;
      CreateInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
      CreateInfo.imageSharingMode = vk::SharingMode::eExclusive;
      CreateInfo.queueFamilyIndexCount = 0;     // Optional
      CreateInfo.pQueueFamilyIndices = nullptr; // Optional
    }
    CreateInfo.preTransform = swapchainSupport.capabilities.currentTransform;
    CreateInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    CreateInfo.presentMode = presentMode;
    CreateInfo.clipped = VK_TRUE;
    CreateInfo.oldSwapchain = VK_NULL_HANDLE;

    VkSwapchainCreateInfoKHR CreateInfoTmp =
        static_cast<VkSwapchainCreateInfoKHR>(CreateInfo);
    if (vkCreateSwapchainKHR(m_device, &CreateInfoTmp, nullptr, &m_swapchain) !=
        VK_SUCCESS)
      throw std::runtime_error("failed to create swap chain!");

    vkGetSwapchainImagesKHR(m_device, m_swapchain, &imageCount, nullptr);
    m_swapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &imageCount,
                            m_swapchainImages.data());
    m_swapchainImageFormat = static_cast<VkFormat>(surfaceFormat.format);
    m_swapchainExtent = static_cast<VkExtent2D>(extent);
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
    std::set<uint32_t> setUniqueInfoIdx = {indices.GraphicsFamily.value(),
                                           indices.PresentFamily.value()};

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
    VkPhysicalDeviceFeatures deviceFeatures{
        .sampleRateShading = VK_TRUE,
        .samplerAnisotropy = VK_TRUE,
    };

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
    vkGetDeviceQueue(m_device, indices.GraphicsFamily.value(), 0,
                     &m_graphicsQueue);
    vkGetDeviceQueue(m_device, indices.PresentFamily.value(), 0,
                     &m_presentQueue);
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
