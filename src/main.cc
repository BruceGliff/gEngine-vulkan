#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

#include "decoy/decoy.h"

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

class HelloTriangleApplication {
  GLFWwindow * m_Window {};

  unsigned const m_Width {1600};
  unsigned const m_Height {900};

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

public:
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
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);  
    m_Window = glfwCreateWindow(m_Width, m_Height, "Vulkan", nullptr, nullptr);
    assert(m_Window && "Window initializating falis!");
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
    createGraphicPipeline();
  }

  void createGraphicPipeline() {}

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
      break;
    }
  }

  void cleanup() {
    for (auto imageView : m_swapchainImageViews)
      vkDestroyImageView(m_device, imageView, nullptr);
    vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
    vkDestroyDevice(m_device, nullptr);
    if (m_enableValidationLayers)
      destroyDebugUtilsMessengerEXT(m_instance, m_debugMessenger, nullptr);
    vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
    vkDestroyInstance(m_instance, nullptr);

    glfwDestroyWindow(m_Window);

    glfwTerminate();
  }
};

int main() {
#ifndef NDEBUG
  std::cout << "Debug\n";
#endif // Debug
  Decoy::Dump();

  HelloTriangleApplication app;
  try {
    app.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
