
#pragma once

namespace gEng {

// Builder for platform specific handles.
// Used in PlatformManager.
// All function defined in PlatformBuilder.cc
struct PlatformBuilder final {
  template <typename T, typename... Args> static T create(Args &&...args);
};

// Deleter for platform specific handles.
struct PlatformDeleter final {
  template <typename T> static void destroy(T &&) {}
};

} // namespace gEng
