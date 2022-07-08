#pragma once

namespace gEng {

class Platform;

// This class operates with devices and encapsulate vulkan operations.
class DeviceManager final {

  Platform *Plfm{nullptr};

public:
  DeviceManager();
  ~DeviceManager();
};

} // namespace gEng
