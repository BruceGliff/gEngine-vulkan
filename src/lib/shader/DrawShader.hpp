#pragma once

#include <vulkan/vulkan.hpp>

#include "../environment/platform_handler.h"
#include "../uniform_buffer/UniformBuffer.hpp"
// FIXME This is should be combined with shader.h

namespace gEng {

struct DrawShader final {
  // TODO move it to config
  static constexpr auto FInF = 2u;
  DrawShader();

  auto &getUBs() { return UBs; }

  std::array<std::optional<UniformBuffer>, FInF> UBs;

  vk::DescriptorSetLayout DSL;
  vk::DescriptorPool DP;
  std::vector<vk::DescriptorSet> DSs;

private:
  static auto createDescriptorSetLayout(vk::Device);
  static auto createDescriptorPool(vk::Device);
  static auto createDescriptorSet(vk::Device, vk::DescriptorSetLayout,
                                  vk::DescriptorPool);
};

// FIXME move to .cc
inline auto DrawShader::createDescriptorSetLayout(vk::Device Dev) {
  // TODO samplers are null, but descriptorCount=1!.
  // Even though there are no pImmutableSamplers in both LB, descriptorCount
  // has to be at least 1. TODO: find out why.
  vk::DescriptorSetLayoutBinding LayoutBindingUBO{
      0, vk::DescriptorType::eUniformBuffer, 1,
      vk::ShaderStageFlagBits::eVertex};
  vk::DescriptorSetLayoutBinding LayoutBindingSampler{
      1, vk::DescriptorType::eCombinedImageSampler, 1,
      vk::ShaderStageFlagBits::eFragment};

  std::array<vk::DescriptorSetLayoutBinding, 2> Bindings = {
      LayoutBindingUBO, LayoutBindingSampler};

  return Dev.createDescriptorSetLayout({{}, Bindings});
}

inline auto DrawShader::createDescriptorPool(vk::Device Dev) {
  constexpr auto Frames = FInF;

  std::array<vk::DescriptorPoolSize, 2> PoolSizes = {
      vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, Frames},
      vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler,
                             Frames}};
  return Dev.createDescriptorPool({{}, Frames, PoolSizes});
}

inline auto DrawShader::createDescriptorSet(vk::Device Dev,
                                            vk::DescriptorSetLayout DSL,
                                            vk::DescriptorPool DP) {

  std::array<vk::DescriptorSetLayout, FInF> Layouts;
  Layouts.fill(DSL);

  return Dev.allocateDescriptorSets({DP, Layouts});
}

inline DrawShader::DrawShader() {
  auto &PltMgr = PlatformHandler::getInstance();
  auto Dev = PltMgr.get<vk::Device>();

  DSL = createDescriptorSetLayout(Dev);
  DP = createDescriptorPool(Dev);
  DSs = createDescriptorSet(Dev, DSL, DP);

  for (auto &&UB : UBs)
    UB.emplace(PltMgr);
}

} // namespace gEng
