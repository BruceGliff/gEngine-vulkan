#pragma once

#include <string_view>
#include <tuple>
#include <vector>

#include "../image/BufferBuilder.hpp"
#include <vertex.h>

namespace gEng {

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
  ModelVk &operator=(Model &&In) {
    std::swap(In, M);
    initBuf();
    return *this;
  }
  ModelVk() = default;

  // TODO temporary
  auto getVB() const { return std::get<vk::Buffer>(VB); }
  auto getIB() const { return std::get<vk::Buffer>(IB); }
  auto getIndicesSize() const { return M.get<Model::Indices>().size(); }

  ~ModelVk() {
    Dev.destroyBuffer(std::get<vk::Buffer>(VB));
    Dev.freeMemory(std::get<vk::DeviceMemory>(VB));
    Dev.destroyBuffer(std::get<vk::Buffer>(IB));
    Dev.freeMemory(std::get<vk::DeviceMemory>(IB));
  }

private:
  Model M;
  vk::Device Dev;
  BufferBuilder::Type VB;
  BufferBuilder::Type IB;
  void initBuf();
};

} // namespace gEng
