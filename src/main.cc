#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cassert>
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
  // This part is responsible for enabling validation layers for debug.
  std::vector<char const *> const m_ValidationLayers = {
      "VK_LAYER_KHRONOS_validation"};
#ifdef NDEBUG
  bool const m_EnableValidationLayers = false;
#else
  bool const m_EnableValidationLayers = true;
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
  // platform-specific structures by itselfes.
  VkSurfaceKHR m_surface{};

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
        .enabledExtensionCount = 0,
        .pEnabledFeatures = &deviceFeatures,
    };
    // To be compatible with older implementations, as new Vulcan version
    // does not require ValidaionLayers
    if (m_EnableValidationLayers) {
      createInfo.enabledLayerCount =
          static_cast<uint32_t>(m_ValidationLayers.size());
      createInfo.ppEnabledLayerNames = m_ValidationLayers.data();
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

  QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
    QueueFamilyIndices indices;
    // Logic to find queue family indices to populate struct with.
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                             nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                             queueFamilies.data());
    {
      int i = 0;
      for (const auto &queueFamily : queueFamilies) {
        // For better performance one queue famili has to support all requested
        // queues at once, but we also can treat them as different families for
        // unified approach.
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
          indices.graphicsFamily = i;

        // Checks for presentation family support.
        VkBool32 presentSupport = false;
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

    // But we want to find out if GPU is graphicFamily. (?)
    return findQueueFamilies(device).isComplete();
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
    if (!m_EnableValidationLayers)
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
    if (m_EnableValidationLayers)
      extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    return extensions;
  }

  void createInstance() {
    if (m_EnableValidationLayers && !checkValidationLayers())
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
        m_EnableValidationLayers ? populateDebugMessengerInfo()
                                 : VkDebugUtilsMessengerCreateInfoEXT{};
    if (m_EnableValidationLayers) {
      createInfo.enabledLayerCount =
          static_cast<uint32_t>(m_ValidationLayers.size());
      createInfo.ppEnabledLayerNames = m_ValidationLayers.data();

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

    for (const char *layerName : m_ValidationLayers) {
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
    vkDestroyDevice(m_device, nullptr);
    if (m_EnableValidationLayers)
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
