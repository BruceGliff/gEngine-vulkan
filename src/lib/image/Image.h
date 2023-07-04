#pragma once

#include <string_view>
#include <vulkan/vulkan.hpp>

#include <iostream>

namespace gEng {

class Image final {
  // FIXME make it private
public:
  vk::Device Dev;
  vk::Image Img;
  vk::DeviceMemory Mem;

  vk::ImageView ImgView;
  vk::Sampler ImgSampler;
  uint32_t mipLevels;

public:
  Image() = default;

  void setImg(std::string_view Path);

  auto getDescriptorImgInfo() const {
    return vk::DescriptorImageInfo{ImgSampler, ImgView,
                                   vk::ImageLayout::eShaderReadOnlyOptimal};
  }

  ~Image() {
    Dev.destroySampler(ImgSampler);
    Dev.destroyImageView(ImgView);
    Dev.destroyImage(Img);
    Dev.freeMemory(Mem);
  }
};

} // namespace gEng
