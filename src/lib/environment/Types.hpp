#pragma once

#include <vulkan/vulkan.hpp>

/// This aggregation of classes is to combine some vk::Handles in one boundle.
///

namespace gEng {
namespace detail {

enum class QueueType { Graph, Present, None };

template <QueueType QT> struct Queue : vk::Queue {
  static constexpr QueueType Type = QT;
  Queue(vk::Queue const &Q) : vk::Queue{Q} {}
};
using GraphQ = Queue<QueueType::Graph>;
using PresentQ = Queue<QueueType::Present>;

using GraphPresentQ = std::pair<GraphQ, PresentQ>;
inline auto createGPQ(vk::Queue const &G, vk::Queue const &P) {
  return std::make_pair<GraphQ, PresentQ>(G, P);
}

} // namespace detail
} // namespace gEng
