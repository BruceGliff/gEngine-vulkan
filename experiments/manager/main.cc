#include "PlatformManager.h"

#include <iostream>

int main() {

  gEng::PlatformManager PM;
  PM.record<int>();
  int const i = 9;
  PM.record<double>(10.0, i);
  PM.record<int *>();

  std::cout << PM.get<int>() << '\n' << *PM.get<int *>();

  return 0;
}
