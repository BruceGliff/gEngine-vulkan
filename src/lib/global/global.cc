#include "gEng/global.h"

using namespace gEng;

template <>
std::unique_ptr<SingletoneBase<GlbManager>> SingletoneBase<GlbManager>::Mgr{
    nullptr};
