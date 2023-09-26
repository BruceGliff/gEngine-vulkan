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
    auto &PltMgr = PlatformHandler::getInstance();
    for (auto &&UB : UBs)
      UB.emplace(PltMgr);
  }

  auto &getUBs() { return UBs; }

  std::vector<vk::DescriptorSet> *DSL;

  std::array<std::optional<UniformBuffer>, FInF> UBs;

private:
};

} // namespace gEng
