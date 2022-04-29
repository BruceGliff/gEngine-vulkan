#include "shader/shader.h"

#include <fstream>
#include <iostream>

#include <cassert>

std::vector<char> Shader::readFile(std::string_view PathToShader) {
  std::ifstream Kernel{PathToShader.data(), std::ios::binary};
  if (!Kernel.is_open()) {
    std::cerr << "Cannot open file: " << PathToShader << '\n';
    return {};
  }

  std::noskipws(Kernel);
  std::istreambuf_iterator<char> Begin{Kernel}, End;
  return std::vector<char>{Begin, End};
};

Shader::Shader(std::string_view PathToShader)
    : RawData{readFile(PathToShader)} {}

// Seems like defailt allocator already.
// SPIR-V declares that binary file has to be multiple by 4byte.
uint32_t const *Shader::getRawData() {
  return reinterpret_cast<uint32_t const *>(RawData.data());
}

uint32_t const *Shader::getRawData() const {
  return reinterpret_cast<uint32_t const *>(RawData.data());
}

uint32_t Shader::getSize() const {
  assert(RawData.size() % 4 == 0);
  return RawData.size();
}
