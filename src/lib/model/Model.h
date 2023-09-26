#pragma once

#include <string_view>
#include <tuple>
#include <vector>

#include "../image/BufferBuilder.hpp"
#include "../shader/DrawShader.hpp"
#include <vertex.h>

namespace gEng {

class Image;

struct Model final {
  using Vertices = std::vector<Vertex>;
  using Indices = std::vector<uint32_t>;
  using DataTy = std::tuple<Vertices, Indices>;

  Model(std::string_view Path);
  Model() = default;
  Model(Model &&) = default;
  Model &operator=(Model const &) = delete;
  Model &operator=(Model &&) = default;

  template <typename T> auto &get() { return std::get<T>(Data); }

  template <typename T> auto &get() const { return std::get<T>(Data); }

private:
  DataTy Data;

  Model(DataTy &&Data) : Data{std::move(Data)} {}
};

struct ModelVk final {
  // FIXME unlock in Model.cc
  ModelVk(Model &&In) : M{std::move(In)} { initBuf(); }

  ModelVk &operator=(Model &&In) {
    std::swap(In, M);
    initBuf();
    return *this;
  }

  void updateDescriptorSets() const;
  void updateUniformBuffer(uint32_t CurrImg, float Ratio);

  // TODO temporary
  auto getVB() const { return VB.Buffer; }
  auto getIB() const { return IB.Buffer; }
  auto getIndicesSize() const { return M.get<Model::Indices>().size(); }
  auto &getShader() { return Shader; }

  Image *Img;

private:
  vk::Device Dev;
  Model M;
  DrawShader Shader;
  BufferBuilder::Type VB;
  BufferBuilder::Type IB;
  void initBuf();
};

} // namespace gEng
