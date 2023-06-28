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

  class SSTCImpl final {
    vk::Device Dev;
    vk::CommandPool CmdPool;
    // GraphicQueue
    vk::Queue Queue;

    bool ShouldDelete{true};

  public:
    vk::CommandBuffer CmdBuf;

    SSTCImpl(vk::Device Dev, vk::CommandPool CmdPool, vk::Queue Queue)
        : Dev{Dev}, CmdPool{CmdPool}, Queue{Queue} {
      CmdBuf = Dev.allocateCommandBuffers(
                      {CmdPool, vk::CommandBufferLevel::ePrimary, 1})
                   .front();
      CmdBuf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    }
    // These std::move do not do much, but only for observing let them stay.
    SSTCImpl(SSTCImpl &&Other)
        : Dev{std::move(Other.Dev)}, CmdPool{std::move(Other.CmdPool)},
          Queue{std::move(Other.Queue)}, CmdBuf{std::move(Other.CmdBuf)} {
      Other.ShouldDelete = false;
    }

    SSTCImpl(SSTCImpl const &) = delete;
    SSTCImpl &operator=(SSTCImpl const &) = delete;
    SSTCImpl &operator=(SSTCImpl &&) = delete;

    ~SSTCImpl() {
      if (!ShouldDelete)
        return;
      CmdBuf.end();
      Queue.submit(vk::SubmitInfo{}.setCommandBuffers(CmdBuf));
      Queue.waitIdle();
      Dev.freeCommandBuffers(CmdPool, CmdBuf);
    }
  };

  class ScopedSingleTimeCmd final {
    SSTCImpl Impl;

  public:
    ScopedSingleTimeCmd(SSTCImpl &&Impl) : Impl{std::move(Impl)} {}
    using LinkedTy = std::pair<ScopedSingleTimeCmd, vk::CommandBuffer>;
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

  // This Function returns Scoped class which calls Destructor to release
  // resources and returns vk::CommandBuffer, linked with Scoped class.
  ScopedSingleTimeCmd::LinkedTy getSSTC() const {
    SSTCImpl Impl{get<vk::Device>(), get<vk::CommandPool>(),
                  get<gEng::detail::GraphPresentQ>().first};
    auto CmdBuf = Impl.CmdBuf;
    return std::make_pair<ScopedSingleTimeCmd, vk::CommandBuffer>(
        std::move(Impl), std::move(CmdBuf));
  }
};

// access to pltHandler -> vk::Device Dev = PlHdl::get<vk::Device>(); -> should
// generate warning if access to empty field.
//                      ->
//   Who should delete: PlHdl or Generator?
//                         vk::PhysicalDevice Dev =
//                         PlHdl::get<vk::PhysicalDevice>() and so one

} // namespace gEng
