#pragma once

#include <vulkan/vulkan.hpp>

#include "../environment/platform_handler.h"
#include "../uniform_buffer/UniformBuffer.hpp"
// FIXME This is should be combined with shader.h

namespace gEng {

struct DrawShader final {
  // TODO move it to config
  static constexpr auto FInF = 2u;
  DrawShader() {
#if 0
      auto& PltMgr = PlatformHandler::getInstance();
      for (auto &&UB : UBs)
        UB.emplace(PltMgr);
#endif
  }

private:
  std::array<std::optional<UniformBuffer>, FInF> UBs;
};

} // namespace gEng
