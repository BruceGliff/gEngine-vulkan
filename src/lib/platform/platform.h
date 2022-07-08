#pragma once

#include <vulkan/vulkan.hpp>

#include "debug_callback.h"

#ifndef NDEBUG
bool constexpr EnableValidation = true;
#else
bool constexpr EnableValidation = false;
#endif

namespace gEng {

// This class is aggregation of physical and logical devices and so one of
// Vulkan
class Platform final {

  vk::Instance Instance;
  DebugMessenger<EnableValidation> DbgMsger;

public:
  Platform();

  ~Platform();
};

} // namespace gEng
