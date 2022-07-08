#include "platform.h"

#include "gEng/window.h"

#include <algorithm>
#include <string_view>
#include <unordered_set>

using namespace gEng;

// TODO of course get rid of global code!.
// TODO Later this has to be a automatic filling structure.
std::vector<char const *> const ValidationLayers{"VK_LAYER_KHRONOS_validation"};

// Checks if all requested layers are available.
static bool checkValidationLayers() {
  // This vector is responsible for listing requires validation layers for
  // debug.
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

static std::vector<char const *> getRequiredExtensions() {
  std::vector<char const *> AllExtensions{Window::getExtensions()};
  // Addition for callback.
  if constexpr (EnableValidation)
    AllExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  return AllExtensions;
}

static vk::Instance createInstance() {
  if constexpr (EnableValidation)
    if (!checkValidationLayers())
      // TODO this is temporary solution just to see bugs.
      throw std::runtime_error{
          "Requestred validation layers are not available!"};

  vk::ApplicationInfo AppInfo{"Hello triangle", VK_MAKE_VERSION(1, 0, 0),
                              "No Engine", VK_MAKE_VERSION(1, 0, 0),
                              VK_API_VERSION_1_0};

  std::vector<char const *> Extensions{getRequiredExtensions()};
  // TODO may be somehow rewrite this with constexpr?
  std::vector<char const *> const &Layers =
      EnableValidation ? ValidationLayers : std::vector<char const *>{};

  vk::InstanceCreateInfo CreateInfo{{}, &AppInfo, Layers, Extensions};

  // In DebugInfo pointer is uses. It has to be investigated, but for now I
  // assume that DebugInfo and createInstance has to be in the same stack frame.
  DebugInfo<EnableValidation> DI{CreateInfo};
  return vk::createInstance(CreateInfo);
}

Platform::Platform() : Instance{createInstance()}, DbgMsger{Instance} {}

Platform::~Platform() {
  DbgMsger.destroy(Instance);
  Instance.destroy();
}
