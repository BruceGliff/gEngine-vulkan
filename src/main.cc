#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <cassert>

#include "decoy/decoy.h"

class HelloTriangleApplication {
  GLFWwindow * m_Window {};

  unsigned const m_Width {1600};
  unsigned const m_Height {900};

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

  void initVulkan() {}

  void mainLoop() {
    while (!glfwWindowShouldClose(m_Window)) {
      glfwPollEvents();
    }
  }

  void cleanup() {
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