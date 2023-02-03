#include "global.h"

#include <iostream>

int main() {

  TypeDesc Ty = TypeDesc::create<TypeDesc>();

  constexpr global g = {TypeDesc::create<int>(), TypeDesc::create<float>()};

  std::cout << Ty.Ty.name() << std::endl;
}
