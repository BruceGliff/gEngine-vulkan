#include "Builder.hpp"

#include "platform_handler.h"

using namespace gEng;

template <> int ChainsBuilder::create(PlatformHandler const &Plt, int &i) {
  return i + 5;
}
