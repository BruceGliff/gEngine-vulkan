#pragma once

namespace gEng {

template <typename Impl> struct BuilderInterface {
  template <typename T, typename... Args> T create() {
    return impl()->create();
  }

private:
  Impl *impl() { return static_cast<Impl *>(this); }
};

struct PltBuilder : BuilderInterface<PltBuilder> {
  friend BuilderInterface<PltBuilder>;
  template <typename T, typename... Args> T create(Args &&...args);

#ifndef NDEBUG
  static bool constexpr EnableDebug{true};
#else
  static bool constexpr EnableDebug{false};
#endif
};

// Forward declaration.
class PlatformHandler;
struct ChainsBuilder : BuilderInterface<ChainsBuilder> {
  friend BuilderInterface<ChainsBuilder>;

  template <typename T, typename... Args>
  T create(PlatformHandler const &Plt, Args &&...args);
};

} // namespace gEng
