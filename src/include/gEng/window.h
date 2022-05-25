#pragma once

#include <cstdint>
#include <string>
#include <string_view>

// Forward declaration.
// Wrap.
struct GLFWwindow;

namespace vk {
class Instance;
class SurfaceKHR;
} // namespace vk

namespace gEng {

// Every application works with window should be UserWindow.
// It is passed as user to GLFW.
// Has to be protected inheritance.
struct UserWindow {
  bool IsResized{false};
  virtual ~UserWindow();
};

// Wrap of the glfw_window to hide implementation of the glfw.
class Window final {
  uint32_t Width{0};
  uint32_t Height{0};
  std::string Title{};
  GLFWwindow *WrapWindow{nullptr};
  UserWindow *User{nullptr};

public:
  Window() = delete;
  Window(Window const &) = delete;
  Window &operator=(Window const &) = delete;
  ~Window();

  Window(Window &&);
  Window &operator=(Window &&);

  Window(uint32_t WidthIn, uint32_t HeightIn, std::string_view TitleIn,
         UserWindow *User, vk::Instance const &Instance,
         vk::SurfaceKHR *Surface);

  std::pair<uint32_t, uint32_t> getExtent();
  std::pair<uint32_t, uint32_t> getExtent() const;
  std::pair<uint32_t, uint32_t> getNativeExtent() const;
  bool isShouldClose() const;
};
} // namespace gEng
