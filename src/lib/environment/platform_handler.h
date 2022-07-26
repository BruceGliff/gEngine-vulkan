#pragma once

#include <algorithm>
#include <iostream>
#include <typeindex>
#include <unordered_map>
#include <variant>

#include <vulkan/vulkan.hpp>

namespace gEng {

// This is designed for single filling all platform-dependent vk-handles and
// accessing to it from different part of code.
class PlatformHandler {
  using Variant = std::variant<vk::Instance, vk::PhysicalDevice, vk::Device,
                               vk::SurfaceKHR>;
  using Collection = std::unordered_map<std::type_index, Variant>;
  static Collection HandledEntities;

  template <typename T> static std::type_index getTypeIndex() {
    return std::type_index(typeid(T));
  }

public:
  PlatformHandler() = delete;
  ~PlatformHandler() = delete;

  // Setting vk-handle for global access.
  // gEng::PlatformHandler::set<vk::HandleType>(Handle);
  template <typename T> static void set(T Entity) {
    auto [It, IsInserted] =
        HandledEntities.try_emplace(getTypeIndex<T>(), Entity);

    if (!IsInserted) {
      // TODO normal error
      std::cerr << "setting already existing entity\n";
      return;
    }
  }

  // Getting vk-handle from global access.
  // vk::HandleType Handle = gEng::PlatformHandler::get<vk::HandleType>();
  template <typename T> static T get() {
    auto FindIt = HandledEntities.find(getTypeIndex<T>());
    if (FindIt == HandledEntities.end()) {
      // TODO normal error
      std::cerr << "No setted entity\n";
      throw std::runtime_error{"No setted entity\n"};
    }
    return std::get<T>(FindIt->second);
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