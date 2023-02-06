#include "Builder.hpp"

#include "../window/glfw_window.h"
#include "debug_callback.h"
#include "gEng/window.h"

#include <set>
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

// Checks which queue families are supported by the device and which one of
// these supports the commands that we want to use.
struct QueueFamilyIndices final {
  // optional just because we may be want to select GPU with some family, but
  // it is not strictly necessary.
  std::optional<uint32_t> GraphicsFamily{};
  // Not every device can support presentation of the image, so need to
  // check that divece has proper family queue.
  std::optional<uint32_t> PresentFamily{};

  bool isComplete() {
    return GraphicsFamily.has_value() && PresentFamily.has_value();
  }
};
// Swapchain requires more details to be checked.
// - basic surface capabilities.
// - surface format (pixel format, color space).
// - available presentation mode.
struct SwapchainSupportDetails final {
  vk::SurfaceCapabilitiesKHR capabilities;
  std::vector<vk::SurfaceFormatKHR> formats;
  std::vector<vk::PresentModeKHR> presentModes;
};

// TODO maybe everything regarding swapchain move to another module.
// This section covers how to query the structs that include this information.
static SwapchainSupportDetails
querySwapchainSupport(vk::SurfaceKHR Surface, vk::PhysicalDevice Device) {
  return SwapchainSupportDetails{
      .capabilities = Device.getSurfaceCapabilitiesKHR(Surface),
      .formats = Device.getSurfaceFormatsKHR(Surface),
      .presentModes = Device.getSurfacePresentModesKHR(Surface)};
}

static bool checkDeviceExtensionSupport(vk::PhysicalDevice Device) {
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

static QueueFamilyIndices findQueueFamilies(vk::SurfaceKHR Surface,
                                            vk::PhysicalDevice PhysDev) {
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
    if (PhysDev.getSurfaceSupportKHR(i, Surface))
      Indices.PresentFamily = i;

    // Not quite sure why the hell we need this early-break.
    if (Indices.isComplete())
      return Indices;
    ++i;
  }

  return QueueFamilyIndices{};
}

// Checks if device is suitable for our extensions and purposes.
static bool isDeviceSuitable(vk::SurfaceKHR Surface,
                             vk::PhysicalDevice Device) {
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
  return findQueueFamilies(Surface, Device).isComplete() &&
         checkDeviceExtensionSupport(Device) &&
         swapchainSupport(querySwapchainSupport(Surface, Device)) &&
         DeviceFeat.samplerAnisotropy;
  // All three ckecks are different. WTF!
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

template <>
vk::PhysicalDevice
PltBuilder::create<vk::PhysicalDevice>(vk::Instance &Inst,
                                       vk::SurfaceKHR &Surface) {
  std::vector<vk::PhysicalDevice> Devices = Inst.enumeratePhysicalDevices();

  // TODO stops here!
  auto FindIt =
      std::find_if(Devices.begin(), Devices.end(), [&Surface](auto &&Device) {
        return isDeviceSuitable(Surface, Device);
      });
  if (FindIt == Devices.end())
    throw std::runtime_error("failed to find a suitable GPU!");

  return *FindIt;
}

template <>
vk::Device PltBuilder::create<vk::Device>(vk::SurfaceKHR &Surface,
                                          vk::PhysicalDevice &PhysDev) {
  // TODO rethink about using findQueueFamilieses once.
  QueueFamilyIndices Indices = findQueueFamilies(Surface, PhysDev);

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
  return PhysDev.createDevice(
      {{}, QueueCreateInfos, Layers, DeviceExtensions, &DevFeat});
}
