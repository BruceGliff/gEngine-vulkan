#pragma once

namespace gEng {

struct DefBuilder final {
  template <typename T, typename... Args> static T create(Args &&...args);
};

struct DefDeleter final {
  template <typename T> static void destroy(T &&) {}
};

} // namespace gEng
