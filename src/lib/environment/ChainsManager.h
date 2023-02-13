#pragma once

#include "Builder.hpp"
#include "platform_handler.h"

namespace gEng {

class ChainsManager final {
  PlatformHandler const &PltMgr;
  ChainsBuilder B;

  ChainsManager(PlatformHandler const &PltIn) : PltMgr{PltIn} {}
};

} // namespace gEng
