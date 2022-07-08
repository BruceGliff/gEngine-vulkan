#include "gEng/device_manager.h"

#include "platform.h"

using namespace gEng;

DeviceManager::DeviceManager() { Plfm = new Platform(); }

DeviceManager::~DeviceManager() { delete Plfm; }
