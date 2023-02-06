#pragma once

#include <algorithm>
#include <iostream>
#include <optional>
#include <tuple>
#include <typeindex>
#include <unordered_map>

#include <vulkan/vulkan.hpp>

namespace gEng {

// This is designed for single filling all platform-dependent vk-handles and
// accessing to it from different part of code.
class PlatformHandler final {
  template <typename VKType> using Optional = std::optional<VKType>;

  template <typename... Tys> using PackedTys = std::tuple<Optional<Tys>...>;

  using Collection =
      PackedTys<vk::Instance, vk::PhysicalDevice, vk::Device, vk::SurfaceKHR>;
  static Collection HandledEntities;

  template <typename T> static std::type_index getTypeIndex() {
    return std::type_index(typeid(T));
  }

  template <typename T, typename Cnt> static auto &get(Cnt &&C) {
    return std::get<std::optional<T>>(C);
  }

public:
  PlatformHandler() = delete;
  ~PlatformHandler() = delete;

  // Setting vk-handle for global access.
  // gEng::PlatformHandler::set<vk::HandleType>(Handle);
  template <typename T> static void set(T Entity) {
    auto &Ent = get<T>(HandledEntities);
    if (Ent)
      // TODO normal error
      std::cerr << "setting already existing entity\n";
    else
      Ent = Entity;
  }

  // Getting vk-handle from global access.
  // vk::HandleType Handle = gEng::PlatformHandler::get<vk::HandleType>();
  template <typename T> static T get() {
    auto &Ent = get<T>(HandledEntities);
    if (!Ent) {
      // TODO normal error
      std::cerr << "No setted entity\n";
      throw std::runtime_error{"No setted entity\n"};
    }
    return Ent.value();
  }
};

// access to pltHandler -> vk::Device Dev = PlHdl::get<vk::Device>(); -> should
// generate warning if access to empty field.
//                      ->
//                      PlHdl::set<vk::Device>(Generator::Default<vk::Device>());
//                      -> should done once. and generate warning if more.
//                                                Generator::Construct<vk::Device>(..params);
//   Who should delete: PlHdl or Generator?
//                         vk::PhysicalDevice Dev =
//                         PlHdl::get<vk::PhysicalDevice>() and so one

} // namespace gEng
