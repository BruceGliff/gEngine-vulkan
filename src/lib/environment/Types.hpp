#pragma once

#include <vulkan/vulkan.hpp>

/// This aggregation of classes is to combine some vk::Handles in one boundle.
///

namespace gEng {
namespace detail {

struct GraphQ : public vk::Queue {
  GraphQ(vk::Queue const &Q) : vk::Queue{Q} {}
};
struct PresentQ : public vk::Queue {
  PresentQ(vk::Queue const &Q) : vk::Queue{Q} {}
};
using GraphPresentQ = std::pair<GraphQ, PresentQ>;
inline auto createGPQ(vk::Queue const &G, vk::Queue const &P) {
  return std::make_pair<GraphQ, PresentQ>(G, P);
}

} // namespace detail
} // namespace gEng
