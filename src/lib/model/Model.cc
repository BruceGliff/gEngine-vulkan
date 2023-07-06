#include "Model.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <vulkan/vulkan.hpp>

#include "../environment/platform_handler.h"

namespace gEng {

static Model::DataTy loadModel(std::string_view Path) {
  Model::Vertices V;
  Model::Indices I;

  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string warn, err;

  if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, Path.data()))
    throw std::runtime_error(warn + err);

  // FIXME delete this. This is does not work.
  // To avoid using Vertex duplications.
  std::unordered_map<Vertex, uint32_t> uniqueVertices{};
  for (auto const &shape : shapes)
    for (auto const &index : shape.mesh.indices) {
      Vertex vertex{};
      vertex.Position = {attrib.vertices[3 * index.vertex_index + 0],
                         attrib.vertices[3 * index.vertex_index + 1],
                         attrib.vertices[3 * index.vertex_index + 2]};
      vertex.TexCoord = {attrib.texcoords[2 * index.texcoord_index + 0],
                         1.0f - attrib.texcoords[2 * index.texcoord_index + 1]};
      vertex.Color = {1.0f, 1.0f, 1.0f};

      if (uniqueVertices.count(vertex) == 0) {
        uniqueVertices[vertex] = static_cast<uint32_t>(V.size());
        V.push_back(vertex);
      }
      I.push_back(uniqueVertices[vertex]);
    }

  return {V, I};
}

Model::Model(std::string_view Path) : Model(loadModel(Path)) {}

// Data - smth with size(), data().
static auto getDeviceBuffer(auto &&Data) {
  using ValueTy = typename std::remove_reference_t<decltype(Data)>::value_type;
  auto &PltMgr = PlatformHandler::getInstance();
  BufferBuilder B{PltMgr.get<vk::Device>(), PltMgr.get<vk::PhysicalDevice>()};
  auto Size = Data.size() * sizeof(ValueTy);
  return B.createViaStaging<BufferBuilder::Type>(
      Size, Data.data(),
      vk::BufferUsageFlagBits::eTransferDst |
          vk::BufferUsageFlagBits::eVertexBuffer,
      vk::MemoryPropertyFlagBits::eDeviceLocal);
}

void ModelVk::initBuf() {
  Dev = PlatformHandler::getInstance().get<vk::Device>();
  VB = getDeviceBuffer(M.get<Model::Vertices>());
  IB = getDeviceBuffer(M.get<Model::Indices>());
}

// For Initialization once.
#if 0
  ModelVk::ModelVk(Model &&M)
    : M{std::move(M)}
    , Dev{PlatformHandler::getInstance().get<vk::Device>()}
    , VB{getDeviceBuffer(M.get<Model::Vertices>())}
    , IB{getDeviceBuffer(M.get<Model::Indices>())}
         {}
#endif

} // namespace gEng
