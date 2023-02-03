#include "builder.h"

using namespace gEng;

template <> int DefBuilder::create<int>() { return 43; }
template <> double DefBuilder::create<double>(double &&inc, int &&a) {
  return 5.0 + inc + a;
}
template <> int *DefBuilder::create<int *>() { return new int{1}; }

template <> void DefDeleter::destroy(int *&&v) { delete v; }
