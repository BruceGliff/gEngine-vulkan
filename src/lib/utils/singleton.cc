#include "../environment/platform_handler.h"
#include "gEng/global.h"

// Module for late instantiation of each singleton in the program.
// Usage: INSTANCE(SomeInheritedFromSingletonClass)

#define INSTANCE(Class)                                                        \
  template gEng::singleton<Class>::InstanceTy gEng::singleton<Class>::Instance;

INSTANCE(gEng::PlatformHandler);
INSTANCE(gEng::GlbManager);

#undef INSTANCE
