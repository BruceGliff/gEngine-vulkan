#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "decoy/decoy.h"

class HelloTriangleApplication {
  GLFWwindow * m_Window {};

  unsigned const m_Width {1600};
  unsigned const m_Height {900};

  // An instance needed for connection between app and VkLibrary
  // And adds a detailes about app to the driver
  VkInstance m_instance {};
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
  }

  void createInstance() {
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

    uint32_t glfwExtensionCount {0};
    char const **glfwExtensions =
        glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    checkExtensions(glfwExtensions, glfwExtensionCount);

    // Required struct tells the Vulkan driver whick global
    // extension and validation level to use
    VkInstanceCreateInfo createInfo {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &appInfo,
      .enabledLayerCount = 0,
      .enabledExtensionCount = glfwExtensionCount,
      .ppEnabledExtensionNames = glfwExtensions
    };

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

  void mainLoop() {
    while (!glfwWindowShouldClose(m_Window)) {
      glfwPollEvents();
      break;
    }
  }

  void cleanup() {
    vkDestroyInstance(m_instance, nullptr);

    glfwDestroyWindow(m_Window);

    glfwTerminate();
  }
};

int main() {
  #ifdef Debug
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
