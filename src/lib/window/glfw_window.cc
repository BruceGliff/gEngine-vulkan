#include "gEng/window.h"

#include "glfw_window.h"

#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

static void framebufferResizeCallback(GLFWwindow *Window, int Width,
                                      int Height) {
  auto User =
      reinterpret_cast<gEng::UserWindow *>(glfwGetWindowUserPointer(Window));
  User->IsResized = true;
}

using namespace gEng;

UserWindow::~UserWindow() {}

Window::Window(uint32_t WidthIn, uint32_t HeightIn, std::string_view TitleIn,
               UserWindow *UserIn)
    : Width{WidthIn}, Height{HeightIn}, Title{TitleIn}, User{UserIn} {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  // TODO Resize does not work properly.
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);
  WrapWindow =
      glfwCreateWindow(Width, Height, TitleIn.data(), nullptr, nullptr);

  assert(WrapWindow && "Window initializating falis!");

  glfwSetWindowUserPointer(WrapWindow, User);
  glfwSetFramebufferSizeCallback(WrapWindow, framebufferResizeCallback);
}

vk::SurfaceKHR Window::createSurface(vk::Instance const &Instance) const {
  // TODO in wrap.
  VkSurfaceKHR Surface;
  if (glfwCreateWindowSurface(Instance, WrapWindow, nullptr, &Surface) !=
      VK_SUCCESS)
    throw std::runtime_error("failed to create window surface!");
  return Surface;
}

Window::~Window() {
  // TODO. maybe wrong idea as there can be multiple windows.
  glfwDestroyWindow(WrapWindow);
  glfwTerminate();
}

std::pair<uint32_t, uint32_t> Window::updExtent() {
  if (User->IsResized)
    std::tie(Width, Height) = getNativeExtent();

  return {Width, Height};
}

std::pair<uint32_t, uint32_t> Window::getExtent() const {
  assert(!User->IsResized && "We cannot update extent of the constant Window");

  return {Width, Height};
}

std::pair<uint32_t, uint32_t> Window::getNativeExtent() const {
  int Wth{};
  int Hth{};

  glfwGetFramebufferSize(WrapWindow, &Wth, &Hth);
  while (Hth == 0 || Wth == 0) {
    glfwGetFramebufferSize(WrapWindow, &Wth, &Hth);
    glfwWaitEvents();
  }

  return {static_cast<uint32_t>(Wth), static_cast<uint32_t>(Hth)};
}

bool Window::isShouldClose() const { return glfwWindowShouldClose(WrapWindow); }

std::vector<char const *> gEng::getRequiredExtensions(bool EnableDebug) {
  uint32_t ExtensionCount{};
  const char **Extensions = glfwGetRequiredInstanceExtensions(&ExtensionCount);

  std::vector<const char *> AllExtensions{Extensions,
                                          Extensions + ExtensionCount};
  // Addition for callback.
  if (EnableDebug)
    AllExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

  return AllExtensions;
}
