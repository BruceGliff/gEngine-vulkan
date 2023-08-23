#pragma once

#include "Builder.hpp"
#include "debug_callback.h"
#include "detail/Types.hpp"

#include "gEng/utils/singleton.h"
#include "gEng/window.h"

#include <algorithm>
#include <iostream>
#include <optional>
#include <tuple>
#include <typeindex>
#include <unordered_map>

#include <vulkan/vulkan.hpp>

namespace gEng {

// This is designed for single filling all platform-dependent vk-handles and
// accessing to it from different part of code.
class PlatformHandler final : public singleton<PlatformHandler> {
  friend singleton;
  PlatformHandler() {}

  DebugMessenger<PltBuilder::EnableDebug> DbgMsger;

  // FIXME Why do we need each type as Optional and not whole collection as
  // optional?
  template <typename VKType> using Optional = std::optional<VKType>;

  template <typename... Tys> using PackedTys = std::tuple<Optional<Tys>...>;

  // Queue automatically created with logical device, but we need to create a
  // handles. And queues automatically destroyed within device.
  using Collection =
      PackedTys<vk::Instance, vk::SurfaceKHR, vk::PhysicalDevice, vk::Device,
                gEng::detail::GraphPresentQ, vk::CommandPool>;
  Collection HandledEntities;

  template <typename T, typename Cnt> static auto &get(Cnt &&C) {
    return std::get<std::optional<T>>(C);
  }

  // Class is vk::CommandBuffer but with default resource release on destructor.
  class ScopeSingleTimeCommandBufferImpl final : public vk::CommandBuffer {
    vk::Device Dev;
    vk::CommandPool CmdPool;
    // GraphicQueue
    vk::Queue Queue;

    static vk::CommandBuffer allocCmdBuf(vk::Device Dev,
                                         vk::CommandPool CmdPool) {
      return Dev
          .allocateCommandBuffers(
              {CmdPool, vk::CommandBufferLevel::ePrimary, 1})
          .front();
    }

  public:
    ScopeSingleTimeCommandBufferImpl(vk::Device Dev, vk::CommandPool CmdPool,
                                     vk::Queue Queue)
        : vk::CommandBuffer{allocCmdBuf(Dev, CmdPool)}, Dev{Dev},
          CmdPool{CmdPool}, Queue{Queue} {
      begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    }
    ScopeSingleTimeCommandBufferImpl(ScopeSingleTimeCommandBufferImpl const &) =
        delete;
    ~ScopeSingleTimeCommandBufferImpl() {
      end();
      vk::CommandBuffer CmdBuf{*this};
      Queue.submit(vk::SubmitInfo{}.setCommandBuffers(CmdBuf));
      Queue.waitIdle();
      Dev.freeCommandBuffers(CmdPool, CmdBuf);
    }
  };

public:
  void init(Window const &Window) {
    PltBuilder B;
    auto Inst = B.create<vk::Instance>(DbgMsger);
    auto Surface = B.create<vk::SurfaceKHR>(Inst, Window);
    auto PhysDev = B.create<vk::PhysicalDevice>(Inst, Surface);
    auto Dev = B.create<vk::Device>(Surface, PhysDev);
    auto GPQ = B.create<gEng::detail::GraphPresentQ>(Surface, PhysDev, Dev);
    auto CMDPool = B.create<vk::CommandPool>(Surface, PhysDev, Dev);

    HandledEntities =
        std::make_tuple(Inst, Surface, PhysDev, Dev, GPQ, CMDPool);
  }

  // Getting vk-handle from global access.
  // vk::HandleType Handle = gEng::PlatformHandler::get<vk::HandleType>();
  template <typename T> T get() const {
    auto &Ent = get<T>(HandledEntities);
    if (Ent)
      return Ent.value();
    // TODO normal error
    std::cerr << "Platform not initiated\n";
    throw std::runtime_error{"Platform not initiated\n"};
  }

  auto getSSTC() const {
    return ScopeSingleTimeCommandBufferImpl{
        get<vk::Device>(), get<vk::CommandPool>(),
        get<gEng::detail::GraphPresentQ>().first};
  }
};

// access to pltHandler -> vk::Device Dev = PlHdl::get<vk::Device>(); -> should
// generate warning if access to empty field.
//                      ->
//   Who should delete: PlHdl or Generator?
//                         vk::PhysicalDevice Dev =
//                         PlHdl::get<vk::PhysicalDevice>() and so one

} // namespace gEng
