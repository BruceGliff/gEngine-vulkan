#include "gEng/window.h"

#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

static void framebufferResizeCallback(GLFWwindow *Window, int Width,
                                      int Height) {
  auto User =
      reinterpret_cast<gEng::UserWindow *>(glfwGetWindowUserPointer(Window));
  User->IsResized = true;
}

namespace gEng {

Window::Window(uint32_t WidthIn, uint32_t HeightIn, std::string_view TitleIn,
               UserWindow *UserIn, vk::Instance const &Instance,
               vk::SurfaceKHR *Surface)
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

  // TODO in wrap.
  if (glfwCreateWindowSurface(Instance, WrapWindow, nullptr,
                              reinterpret_cast<VkSurfaceKHR *>(Surface)) !=
      VK_SUCCESS)
    throw std::runtime_error("failed to create window surface!");
}

Window::~Window() {
  // TODO. maybe wrong idea as there can be multiple windows.
  glfwDestroyWindow(WrapWindow);
  glfwTerminate();
}

Window::Window(Window &&Other)
    : Width{Other.Width}, Height{Other.Height}, Title{std::move(Other.Title)},
      WrapWindow{Other.WrapWindow} {
  Other.WrapWindow = nullptr;
}

Window &Window::operator=(Window &&Other) {
  std::swap(Width, Other.Width);
  std::swap(Height, Other.Height);
  std::swap(Title, Other.Title);
  std::swap(WrapWindow, Other.WrapWindow);

  return *this;
}

std::pair<uint32_t, uint32_t> Window::getExtent() {
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

  return {Wth, Hth};
}

bool Window::isShouldClose() const { return glfwWindowShouldClose(WrapWindow); }

} // namespace gEng
