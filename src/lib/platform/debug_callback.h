#pragma once

#include <vulkan/vulkan.hpp>

// Here I'll move everythink connected with debug callback function

namespace gEng {

vk::DebugUtilsMessengerCreateInfoEXT populateDebugMessengerInfo();

// IsValidation is either validation layers on or off.
template <bool IsValidation> class DebugInfo {
  vk::DebugUtilsMessengerCreateInfoEXT Info{};

public:
  DebugInfo(vk::InstanceCreateInfo &InstInfo) {}
};
template <> class DebugInfo<true> {
  vk::DebugUtilsMessengerCreateInfoEXT Info{populateDebugMessengerInfo()};

public:
  DebugInfo(vk::InstanceCreateInfo &InstInfo) { InstInfo.setPNext(&Info); }
};

template <bool IsValidation> class DebugMessenger {
  DebugMessenger(vk::Instance const &Inst) {}
  void destroy(vk::Instance const &Inst) {}
};
template <> class DebugMessenger<true> {
  vk::DebugUtilsMessengerEXT DbgMsger;

public:
  DebugMessenger(vk::Instance const &Inst)
      : DbgMsger{
            Inst.createDebugUtilsMessengerEXT(populateDebugMessengerInfo())} {}
  void destroy(vk::Instance const &Inst) {
    Inst.destroyDebugUtilsMessengerEXT(DbgMsger);
  }
};

} // namespace gEng