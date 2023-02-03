#include "../environment/PlatformManager.h"
#include "gEng/global.h"

// Module for late instantiation of each singleton in the program.
// Usage: INSTANCE(SomeInheritedFromSingletonClass)

#define INSTANCE(Class)                                                        \
  template gEng::singleton<Class>::InstanceTy gEng::singleton<Class>::Instance;

INSTANCE(gEng::PlatformManager);
INSTANCE(gEng::GlbManager);

#undef INSTANCE
