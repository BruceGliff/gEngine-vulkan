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

  template <typename T> inline T *castIt(auto const &It) const;

public:
  template <typename T, typename... Args>
  inline T &registerEntity(Args &&...args);

  template <typename T> inline T *getEntityIfPossible() const;

  template <typename T> inline T &getEntity() const;
};

} // namespace gEng

#include "global/global.hpp"
