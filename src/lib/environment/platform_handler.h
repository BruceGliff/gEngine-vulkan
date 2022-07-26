#pragma once

#include <algorithm>
#include <iostream>
#include <typeindex>
#include <unordered_map>
#include <variant>

#include <vulkan/vulkan.hpp>

namespace gEng {

class PlatformHandler {

  using Variant = std::variant<vk::Instance, vk::PhysicalDevice, vk::Device>;
  using Collection = std::unordered_map<std::type_index, Variant>;
  static Collection HandledEntities;

  template <typename T> static std::type_index getTypeIndex() {
    return std::type_index(typeid(T));
  }

public:
  PlatformHandler() = delete;
  ~PlatformHandler() = delete;

  template <typename T> static void set(T Entity) {
    auto [It, IsInserted] =
        HandledEntities.try_emplace(getTypeIndex<T>(), Entity);

    if (!IsInserted) {
      // TODO normal error
      std::cerr << "setting already existing entity\n";
      return;
    }
  }

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
//                         PlHdl::get<vk::PhusicalDevice>() and so one

} // namespace gEng