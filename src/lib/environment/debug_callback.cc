#include "debug_callback.h"

#include <iostream>

static VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              VkDebugUtilsMessageTypeFlagsEXT messageType,
              const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
              void *pUserData) {

  // The way to control dumping  validation info.
  if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    // Message is important enough to show
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
  }

  return VK_FALSE;
}

namespace gEng {

vk::DebugUtilsMessengerCreateInfoEXT populateDebugMessengerInfo() {
  using MessageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT;
  using MessageType = vk::DebugUtilsMessageTypeFlagBitsEXT;
  return {{},
          MessageSeverity::eVerbose | MessageSeverity::eWarning |
              MessageSeverity::eError,
          MessageType::eGeneral | MessageType::ePerformance |
              MessageType::eValidation,
          debugCallback};
}

} // namespace gEng