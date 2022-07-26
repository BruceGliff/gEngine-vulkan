#include "plarform_manager.h"

#include "../window/glfw_window.h"

#include <algorithm>
#include <iostream>
#include <string_view>
#include <type_traits>
#include <unordered_set>

using namespace gEng;

std::unique_ptr<PltManager> PltManager::Mgr{nullptr};
PltManager &PltManager::getMgrInstance() {
  PltManager *Ptr = Mgr.get();
  if (Ptr)
    return *Ptr;

  Mgr = std::unique_ptr<PltManager>(new PltManager{});
  return *Mgr.get();
}

template <bool EnableDebug> static auto constexpr getValidationLayers() {
  if constexpr (EnableDebug)
    return std::array{"VK_LAYER_KHRONOS_validation"};
  else
    return nullptr;
}

template <typename ValLayer>
static bool checkValidationLayers(ValLayer const &ValidationLayers) {
  // If there is no validation layers then no need to check.
  if constexpr (!std::is_class_v<ValLayer>)
    return true;

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

vk::Instance PltManager::createInstance() {
  if (Instance) {
    std::cerr << "Instance already has been created.\n";
    return Instance.value();
  }

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

  Instance = vk::createInstance(CreateInfo);

  DbgMsger = DebugMessenger<EnableDebug>{Instance.value()};
  return Instance.value();
}

vk::SurfaceKHR PltManager::createSurface(gEng::Window const &Window) {
  if (!Instance)
    throw std::runtime_error("Cannot create Surface without Instance.");
  vk::Instance &Inst = Instance.value();

  if (Surface) {
    std::cerr << "Surface already has been created.\n";
    return Surface.value();
  }

  Surface = Window.createSurface(Inst);
  return Surface.value();
}

vk::PhysicalDevice PltManager::createPhysicalDevice() {
  if (!Instance)
    throw std::runtime_error("Cannot create physical device without Instance.");
  vk::Instance &Inst = Instance.value();

  std::vector<vk::PhysicalDevice> Devices = Inst.enumeratePhysicalDevices();

  // TODO stops here!
  auto FindIt =
      std::find_if(Devices.begin(), Devices.end(),
                   [this](auto &&Device) { return isDeviceSuitable(Device); });
  if (FindIt == Devices.end())
    throw std::runtime_error("failed to find a suitable GPU!");

  m_physicalDevice = *FindIt;
  msaaSamples = getMaxUsableSampleCount();
}

vk::Device PltManager::createDevice() {}

PltManager::~PltManager() {
  // Dev.destroy();
  if (Instance) {
    vk::Instance &Inst = Instance.value();
    DbgMsger.destroy(Inst);
    if (Surface)
      Inst.destroySurfaceKHR(Surface.value());
    // TODO: Do I need destroy this?
    Inst.destroy();
  }
}