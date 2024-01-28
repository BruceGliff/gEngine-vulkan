#pragma once

#include <string_view>
#include <tuple>
#include <vector>

#include "../image/BufferBuilder.hpp"
#include "../uniform_buffer/UniformBuffer.hpp"
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

  void updateUniformBuffer(uint32_t CurrImg, float Ratio);

  void bind(vk::CommandBuffer CmdBuff) const {
    vk::DeviceSize Offsets{0};
    CmdBuff.bindVertexBuffers(0, getVB(), Offsets);
    CmdBuff.bindIndexBuffer(getIB(), 0, vk::IndexType::eUint32);
  }

  void draw(vk::CommandBuffer CmdBuff) const {
    // vertexCount, instanceCount, fitstVertex, firstInstance
    CmdBuff.drawIndexed(getIndicesSize(), 1, 0, 0, 0);
  }

  // FIXME Config::FInF
  std::array<std::optional<UniformBuffer>, 2> UBs;

private:
  vk::Buffer getVB() const { return VB.Buffer; }
  vk::Buffer getIB() const { return IB.Buffer; }
  size_t getIndicesSize() const { return M.get<Model::Indices>().size(); }

  vk::Device Dev;
  Model M;
  BufferBuilder::Type VB;
  BufferBuilder::Type IB;

  void initBuf();
};

} // namespace gEng
