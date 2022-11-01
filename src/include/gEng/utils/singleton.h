#pragma once

#include <memory>

// Part of forward declarations.
namespace gEng {
class PltManager;
class GlbManager;
} // namespace gEng

// End of forward declarations.

namespace gEng {

// Every singleton in program has to be inherited from this singleton.
// All instantiations has to be in singleton.cc module.
// Instance can be accessed as:
//  1. SomeClass &S = singleton<SomeClass>::getInstance();
//  2. SomeClass &S = SomeClass::getInstance();
template <typename Parent> class singleton {
public:
  using InstanceTy = std::unique_ptr<Parent>;

private:
  static inline InstanceTy Instance{nullptr};

protected:
  singleton() {}

public:
  static Parent &getInstance() {
    Parent *Ptr = Instance.get();
    if (!Ptr)
      Instance = std::unique_ptr<Parent>(new Parent);
    return *Instance.get();
  }
};

} // namespace gEng

// Part of late instantiations.
#define LATE_INSTANCE(Class)                                                   \
  extern template gEng::singleton<Class>::InstanceTy                           \
      gEng::singleton<Class>::Instance;

LATE_INSTANCE(gEng::PltManager);
LATE_INSTANCE(gEng::GlbManager);

#undef LATE_INSTANCE
// End of late instantiations.
