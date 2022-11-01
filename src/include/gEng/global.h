#pragma once

#include "gEng/utils/singleton.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <typeindex>
#include <unordered_map>

namespace gEng {

// Class represents singleton which is accessable from every point of the
// program.
class GlbManager : public singleton<GlbManager> {
  friend singleton;
  GlbManager() {}

  // Handle memory has another specific class.
  // TODO for now free memory by hands.
  using RegsTable = std::unordered_map<std::type_index, void *>;
  RegsTable Regs;

  template <typename T> T *castIt(auto const &It) const;

public:
  template <typename T, typename... Args> T &registerEntity(Args &&...args);

  template <typename T> T *getEntityIfPossible() const;

  template <typename T> T &getEntity() const;
};

} // namespace gEng

#include "global/global.hpp"
