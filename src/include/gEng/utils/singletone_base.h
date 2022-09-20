#pragma once

#include <memory>

namespace gEng {

// Templace class for singleton logic realization.
// All instantiation have to be in lib/utils/singletone_insts.cc file.
// Simple access to instance throught getInstance() call.
template <typename T> class SingletoneBase {
  static std::unique_ptr<SingletoneBase<T>> Mgr;
  SingletoneBase() {}

public:
  static T &getInstance() {
    if (!Mgr)
      Mgr = std::unique_ptr<SingletoneBase<T>>(new SingletoneBase<T>{});
    return *static_cast<T *>(Mgr.get());
  }
  // TODO virtual destr?
};

// TODO remove this implementation. Do smth like this:
// class<T> Singleton {
//   static T& getInstance() {
//      static T inst;
//      return inst;
//    }
template <typename T>
std::unique_ptr<SingletoneBase<T>> SingletoneBase<T>::Mgr{nullptr};

} // namespace gEng
