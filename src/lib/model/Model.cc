#include "Model.h"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <vulkan/vulkan.hpp>

#include "../environment/platform_handler.h"
#include "../image/Image.h"
#include "../shader/DrawShader.hpp"

#include <chrono>

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
static auto getDeviceBuffer(auto &&Data, auto BufferUsage) {
  auto &PltMgr = PlatformHandler::getInstance();
  BufferBuilder B{PltMgr.get<vk::Device>(), PltMgr.get<vk::PhysicalDevice>()};
  return B.createViaStaging(Data,
                            vk::BufferUsageFlagBits::eTransferDst | BufferUsage,
                            vk::MemoryPropertyFlagBits::eDeviceLocal);
}

void ModelVk::initBuf() {
  auto &PltMgr = PlatformHandler::getInstance();
  Dev = PltMgr.get<vk::Device>();
  VB = getDeviceBuffer(M.get<Model::Vertices>(),
                       vk::BufferUsageFlagBits::eVertexBuffer);
  IB = getDeviceBuffer(M.get<Model::Indices>(),
                       vk::BufferUsageFlagBits::eIndexBuffer);
  for (auto &&UB : UBs)
    UB.emplace(PltMgr);
}

void ModelVk::updateUniformBuffer(uint32_t CurrImg, float Ratio) {
  static auto const StartTime = std::chrono::high_resolution_clock::now();

  auto const CurrTime = std::chrono::high_resolution_clock::now();
  float const Time = std::chrono::duration<float, std::chrono::seconds::period>(
                         CurrTime - StartTime)
                         .count();
  gEng::UniformBufferObject ubo{
      .Model = glm::rotate(glm::mat4(1.f), Time * glm::radians(90.f),
                           glm::vec3(0.f, 0.f, 1.f)),
      .View = glm::lookAt(glm::vec3(2.f, 2.f, 2.f), glm::vec3(0.f, 0.f, 0.f),
                          glm::vec3(0.f, 0.f, 1.f)),
      .Proj = glm::perspective(glm::radians(45.f), Ratio, 0.1f, 10.f)};
  ubo.Proj[1][1] *= -1; // because GLM designed for OpenGL.
  UBs[CurrImg]->store(ubo);
}

// For Initialization once.
#if 0
  ModelVk::ModelVk(Model &&M)
    : M{std::move(M)}
    , VB{getDeviceBuffer(M.get<Model::Vertices>())}
    , IB{getDeviceBuffer(M.get<Model::Indices>())}
       {}
#endif

} // namespace gEng
