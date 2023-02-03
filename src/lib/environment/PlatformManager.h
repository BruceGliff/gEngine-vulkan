#pragma once

#include "PlatformBuilder.h"

// TODO this can be move to PlatformManager.cc
#include "gEng/utils/singleton.h"
#include "debug_callback.h"
#include "gEng/window.h"

#include <tuple>

#include <vulkan/vulkan.hpp>

// template <typename T>
// check for static function create
// concept Builder = requires { T::create; };

namespace gEng {

class PlatformManager final : public singleton<PlatformManager> {
    friend singleton;
    PlatformManager() {}
  PlatformBuilder B;
  using TypeReg =
    std::tuple<vk::Instance, vk::SurfaceKHR, vk::PhysicalDevice, vk::Device>;
  TypeReg Collection;

private:
  DebugMessenger<PlatformBuilder::EnableDebug> DbgMsger;
  bool isDeviceSuitable(vk::PhysicalDevice const &) const;

public:

  template <typename T, typename... Args>
  T &record(Args &&...args) {
    auto &&Rcd = std::get<T>(Collection);
    return Rcd = B.template create<T>(std::forward<Args>(args)...);
  }

  template <typename T> T &get() { return std::get<T>(Collection); }

    void init(gEng::Window const &W) {
      auto &Ins = record<vk::Instance>(DbgMsger);
      auto &Surf = record<vk::SurfaceKHR>(Ins, W);
      auto &PDev = record<vk::PhysicalDevice>(Ins);
      auto &Dev = record<vk::Device>(Surf, PDev);
    }

  ~PlatformManager() {
    // TODO delete in reverse.
  }
};

} // namespace gEng
