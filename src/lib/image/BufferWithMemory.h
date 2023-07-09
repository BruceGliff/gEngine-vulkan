#pragma once

#include <vulkan/vulkan.hpp>

namespace gEng {

// TODO incapsulating
struct BufferWithMemory final {
  vk::Buffer Buffer;
  vk::DeviceMemory Memory;
  vk::Device Dev;

  bool ToDelete = false;

  BufferWithMemory() = default;
  BufferWithMemory(vk::Buffer Buffer, vk::DeviceMemory Memory, vk::Device Dev)
      : Buffer{Buffer}, Memory{Memory}, Dev{Dev}, ToDelete{true} {}

  BufferWithMemory(BufferWithMemory &&Other)
      : Buffer{Other.Buffer}, Memory{Other.Memory}, Dev{Other.Dev},
        ToDelete{Other.ToDelete} {
    Other.ToDelete = false;
  }

  void swap(BufferWithMemory &&Other) {
    std::swap(Buffer, Other.Buffer);
    std::swap(Memory, Other.Memory);
    std::swap(Dev, Other.Dev);
    std::swap(ToDelete, Other.ToDelete);
  }

  void store(void const *Src, size_t Size) {
    void *Dst = Dev.mapMemory(Memory, 0, Size);
    std::memcpy(Dst, Src, Size);
    Dev.unmapMemory(Memory);
  }

  BufferWithMemory &operator=(BufferWithMemory &&Other) {
    swap(std::move(Other));
    return *this;
  }

  ~BufferWithMemory() {
    if (!ToDelete)
      return;
    Dev.destroyBuffer(Buffer);
    Dev.freeMemory(Memory);
  }
};

} // namespace gEng
