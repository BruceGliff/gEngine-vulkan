#include "gEng/global.h"

using namespace gEng;

std::unique_ptr<GlbManager> GlbManager::Mgr{nullptr};
GlbManager &GlbManager::getInstance() {
  if (Mgr)
    return *Mgr.get();
  Mgr = std::unique_ptr<GlbManager>(new GlbManager{});
  return *Mgr.get();
}
