#pragma once

#include "PlatformBuilder.h"

#include <vulkan.hpp>

#include <tuple>

// template <typename T>
// check for static function create
// concept Builder = requires { T::create; };

namespace gEng {

template <typename Builder = gEng::PlatformBuilder>
class PlatformManager final {
  Builder B;
  using TypeReg =
      std::tuple<vk::Instance, vk::SurfaceKHR, vk::PhysicalDevice, vk::Device>;
  TypeReg Collection;

  bool isDeviceSuitable(vk::PhysicalDevice const &) const;

public:
  template <typename T, typename... Args> T &record(Args &&...args) {
    return record<T>(B, std::forward<Args>(args)...);
  }

  template <typename T, typename CustomBuilder, typename... Args>
  T &record(CustomBuilder &CB, Args &&...args) {
    auto &&Rcd = std::get<T>(Collection);
    return Rcd = CB.template create<T>(std::forward<Args>(args)...);
  }

  template <typename T> T &get() { return std::get<T>(Collection); }

  ~PlatformManager() {
    // TODO delete in reverse.
  }
};

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
QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice const &);

} // namespace gEng
