#pragma once

namespace gEng {

template <typename Impl> struct BuilderInterface {
  template <typename T, typename... Args> T create() {
    return impl()->create();
  }

private:
  Impl *impl() { return static_cast<Impl *>(this); }
};

} // namespace gEng
