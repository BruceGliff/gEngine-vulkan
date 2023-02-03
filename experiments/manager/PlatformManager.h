#pragma once

#include "builder.h"
#include <tuple>

// template <typename T>
// check for static function create
// concept Builder = requires { T::create; };

namespace gEng {

template <typename Builder = gEng::DefBuilder> class PlatformManager final {
  Builder B;
  using TypeReg = std::tuple<int, double, int *>;
  TypeReg Collection;

public:
  template <typename T, typename... Args> T &record(Args &&...args) {
    return record<T>(B, std::forward<Args>(args)...);
  }

  template <typename T, typename CustomBuilder, typename... Args>
  T &record(CustomBuilder &CB, Args &&...args) {
    auto &&Rcd = std::get<T>(Collection);
    return Rcd = CB.template create<T>(std::forward<Args>(args)...);
  }

  template <typename T> T &get() { return std::get<T>(Collection); }

  ~PlatformManager() {
    DefDeleter::destroy(std::get<1>(Collection));
    DefDeleter::destroy(std::get<2>(Collection));
  }
};

} // namespace gEng
