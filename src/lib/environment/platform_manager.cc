#include "platform_manager.h"

#include "Builder.hpp"

#include "../window/glfw_window.h"

#include <algorithm>
#include <iostream>
#include <ranges>
#include <set>
#include <string_view>
#include <type_traits>
#include <unordered_set>

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

vk::Instance PltManager::createInstance() {
  gEng::PltBuilder B;
  Instance = B.create<vk::Instance>(DbgMsger);
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

QueueFamilyIndices
PltManager::findQueueFamilies(vk::PhysicalDevice const &PhysDev) const {
  QueueFamilyIndices Indices;
  std::vector<vk::QueueFamilyProperties> QueueFamilies =
      PhysDev.getQueueFamilyProperties();

  // TODO: rething this approach.
  uint32_t i{0};
  for (auto &&queueFamily : QueueFamilies) {
    // For better performance one queue family has to support all requested
    // queues at once, but we also can treat them as different families for
    // unified approach.
    if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics)
      Indices.GraphicsFamily = i;

    // Checks for presentation family support.
    if (PhysDev.getSurfaceSupportKHR(i, Surface.value()))
      Indices.PresentFamily = i;

    // Not quite sure why the hell we need this early-break.
    if (Indices.isComplete())
      return Indices;
    ++i;
  }

  return QueueFamilyIndices{};
}

// Swapchain requires more details to be checked.
// - basic surface capabilities.
// - surface format (pixel format, color space).
// - available presentation mode.
struct SwapchainSupportDetails {
  vk::SurfaceCapabilitiesKHR capabilities;
  std::vector<vk::SurfaceFormatKHR> formats;
  std::vector<vk::PresentModeKHR> presentModes;
};

// TODO maybe everything regarding swapchain move to another module.
// This section covers how to query the structs that include this information.
static SwapchainSupportDetails
querySwapchainSupport(vk::PhysicalDevice const &Device,
                      vk::SurfaceKHR const &Surface) {
  return SwapchainSupportDetails{
      .capabilities = Device.getSurfaceCapabilitiesKHR(Surface),
      .formats = Device.getSurfaceFormatsKHR(Surface),
      .presentModes = Device.getSurfacePresentModesKHR(Surface)};
}

static bool checkDeviceExtensionSupport(vk::PhysicalDevice const &Device) {
  // Some bad code. Rethink!
  std::vector<vk::ExtensionProperties> AvailableExtensions =
      Device.enumerateDeviceExtensionProperties();
  std::unordered_set<std::string_view> UniqieExt;
  std::for_each(AvailableExtensions.begin(), AvailableExtensions.end(),
                [&UniqieExt](auto &&Extension) {
                  UniqieExt.insert(Extension.extensionName);
                });
  return std::all_of(DeviceExtensions.begin(), DeviceExtensions.end(),
                     [&UniqieExt](char const *RequireExt) {
                       return UniqieExt.contains(RequireExt);
                     });
}

// Checks if device is suitable for our extensions and purposes.
bool PltManager::isDeviceSuitable(vk::PhysicalDevice const &Device) const {
  if (!Surface)
    throw std::runtime_error("Cannot pick physical device without Surface.");
  // name, type, supported Vulkan version can be quired via
  // GetPhysicalDeviceProperties.
  // vk::PhysicalDeviceProperties DeviceProps = Device.getProperties();

  // optional features like texture compressing, 64bit floating operations,
  // multiview-port and so one...
  vk::PhysicalDeviceFeatures DeviceFeat = Device.getFeatures();

  // Right now I have only one INTEGRATED GPU(on linux). But it will be more
  // suitable to calculate score and select preferred GPU with the highest
  // score. (eg. discrete GPU has +1000 score..)

  // Swap chain support is sufficient for this tutorial if there is at least
  // one supported image format and one supported presentation mode given
  // the window surface we have.
  auto swapchainSupport = [](SwapchainSupportDetails swapchainDetails) {
    return !swapchainDetails.formats.empty() &&
           !swapchainDetails.presentModes.empty();
  };

  // But we want to find out if GPU is graphicFamily. (?)
  return findQueueFamilies(Device).isComplete() &&
         checkDeviceExtensionSupport(Device) &&
         swapchainSupport(querySwapchainSupport(Device, Surface.value())) &&
         DeviceFeat.samplerAnisotropy;
  // All three ckecks are different. WTF!
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

  PhysDev = *FindIt;
  return PhysDev;
}

vk::Device PltManager::createDevice() {
  // TODO rethink about using findQueueFamilieses once.
  QueueFamilyIndices Indices = findQueueFamilies(PhysDev);

  // Each queue family has to have own VkDeviceQueueCreateInfo.
  std::vector<vk::DeviceQueueCreateInfo> QueueCreateInfos{};
  // This is the worst way of doing it. Rethink!
  std::set<uint32_t> UniqueIdc = {Indices.GraphicsFamily.value(),
                                  Indices.PresentFamily.value()};
  // TODO: what is this?
  std::array<float, 1> const QueuePriority = {1.f};
  for (uint32_t Family : UniqueIdc)
    QueueCreateInfos.push_back(
        vk::DeviceQueueCreateInfo{{}, Family, QueuePriority});

  vk::PhysicalDeviceFeatures DevFeat{};
  DevFeat.samplerAnisotropy = VK_TRUE;
  DevFeat.sampleRateShading = VK_TRUE;

  auto constexpr Layers = getValidationLayers<EnableDebug>();
  Device = PhysDev.createDevice(
      {{}, QueueCreateInfos, Layers, DeviceExtensions, &DevFeat});
  return Device.value();
}

PltManager::~PltManager() {
  if (Device)
    Device.value().destroy();

  if (Instance) {
    vk::Instance &Inst = Instance.value();
    DbgMsger.destroy(Inst);
    if (Surface)
      Inst.destroySurfaceKHR(Surface.value());
    // TODO: Do I need destroy this?
    Inst.destroy();
  }
}
