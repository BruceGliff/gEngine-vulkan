#pragma once

#include <vulkan/vulkan.hpp>

#include "../environment/platform_handler.h"
#include "../image/Image.h"
#include "../model/Model.h"
// FIXME This is should be combined with shader.h

namespace gEng {

struct DrawShader final {
  // TODO move it to config
  static constexpr auto FInF = 2u;
  DrawShader();

  vk::Device Dev;
  vk::DescriptorSetLayout DSL;
  vk::DescriptorPool DP;
  std::vector<vk::DescriptorSet> DSs;
  vk::PipelineLayout PL;

  ~DrawShader() { cleanup(); }

  void connect(ModelVk const &, Image const &);

  void bind(vk::CommandBuffer CmdBuff, unsigned Frame) const {
    CmdBuff.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, PL, 0,
                               DSs[Frame], nullptr);
  }

private:
  static auto createDescriptorSetLayout(vk::Device);
  static auto createDescriptorPool(vk::Device);
  static auto createDescriptorSet(vk::Device, vk::DescriptorSetLayout,
                                  vk::DescriptorPool);
  static auto createPipelineLayout(vk::Device, vk::DescriptorSetLayout);

  void cleanup();
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

inline auto DrawShader::createPipelineLayout(vk::Device Dev,
                                             vk::DescriptorSetLayout DSL) {
  return Dev.createPipelineLayout({{}, DSL});
}

inline DrawShader::DrawShader() {
  auto &PltMgr = PlatformHandler::getInstance();
  Dev = PltMgr.get<vk::Device>();

  DSL = createDescriptorSetLayout(Dev);
  DP = createDescriptorPool(Dev);
  DSs = createDescriptorSet(Dev, DSL, DP);
  PL = createPipelineLayout(Dev, DSL);

  assert(FInF == DSs.size());
}

inline void DrawShader::cleanup() {
  Dev.destroyPipelineLayout(PL);
  Dev.destroyDescriptorPool(DP);
  Dev.destroyDescriptorSetLayout(DSL);
}

inline void DrawShader::connect(ModelVk const &Md, Image const &Img) {
  auto ImgInfo = Img.getDescriptorImgInfo();
  for (int i = 0; i != FInF; ++i) {
    auto BufInfo = Md.UBs[i]->getDescriptorBuffInfo();
    vk::WriteDescriptorSet BufWrite{
        DSs[i], 0, 0, vk::DescriptorType::eUniformBuffer, nullptr, BufInfo};
    vk::WriteDescriptorSet ImgWrite{
        DSs[i],  1,      0, vk::DescriptorType::eCombinedImageSampler,
        ImgInfo, nullptr};
    std::array<vk::WriteDescriptorSet, 2> DescriptorWrites{std::move(BufWrite),
                                                           std::move(ImgWrite)};
    // I think this update should occure each frame:
    // https://vulkan.lunarg.com/doc/view/latest/windows/tutorial/html/08-init_pipeline_layout.html
    Dev.updateDescriptorSets(DescriptorWrites, nullptr);
  }
}

} // namespace gEng
