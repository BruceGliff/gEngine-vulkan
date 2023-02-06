#include "Builder.hpp"

#include "../window/glfw_window.h"
#include "debug_callback.h"
#include "gEng/window.h"

#include <unordered_set>

#include <vulkan/vulkan.hpp>

using namespace gEng;

std::vector<char const *> const DeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

// In case if debug mode is disabled returns nullptr.
// That means no validation layers.
template <bool EnableDebug> auto constexpr getValidationLayers() {
  return nullptr;
}
// In case if in debug mode here should be specified all validation layers.
template <> auto constexpr getValidationLayers<true>() {
  return std::array{"VK_LAYER_KHRONOS_validation"};
}

// In case if there no layers - returns true.
template <typename Other> static bool checkValidationLayers(Other const &) {
  return true;
}

// Check all available layers.
static bool
checkValidationLayers(std::ranges::range auto const &ValidationLayers) {
  auto AvailableLayers = vk::enumerateInstanceLayerProperties();
  std::unordered_set<std::string_view> UniqueLayers;
  std::for_each(AvailableLayers.begin(), AvailableLayers.end(),
                [&UniqueLayers](auto const &LayerProperty) {
                  UniqueLayers.insert(LayerProperty.layerName);
                });
  return std::all_of(ValidationLayers.begin(), ValidationLayers.end(),
                     [&UniqueLayers](char const *RequireLayer) {
                       return UniqueLayers.contains(RequireLayer);
                     });
}

template <>
vk::Instance
PltBuilder::create<vk::Instance>(DebugMessenger<EnableDebug> &DbgMsger) {
  vk::ApplicationInfo AppInfo{"Hello triangle", VK_MAKE_VERSION(1, 0, 0),
                              "No Engine", VK_MAKE_VERSION(1, 0, 0),
                              VK_API_VERSION_1_0};

  std::vector<char const *> Extensions{getRequiredExtensions(EnableDebug)};

  auto constexpr Layers = getValidationLayers<EnableDebug>();
  if (!checkValidationLayers(Layers))
    throw std::runtime_error{"Requestred validation layers are not available!"};

  vk::InstanceCreateInfo CreateInfo{{}, &AppInfo, Layers, Extensions};

  // In DebugInfo pointer is uses. It has to be investigated, but for now I
  // assume that DebugInfo and createInstance has to be in the same stack frame.
  DebugInfo<EnableDebug> DI{CreateInfo};

  vk::Instance Instance = vk::createInstance(CreateInfo);

  // FIXME DbgMsger as argument is dirty hack.
  DbgMsger = DebugMessenger<EnableDebug>{Instance};
  return Instance;
}

template <>
vk::SurfaceKHR PltBuilder::create<vk::SurfaceKHR>(vk::Instance &Inst,
                                                  gEng::Window const &Window) {
  return Window.createSurface(Inst);
}
