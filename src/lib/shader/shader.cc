#include <shader/shader.h>

#include <fstream>
#include <iostream>
#include <sstream>

#include <cassert>

std::string Shader::readFile(std::string_view PathToShader) {
  std::ifstream Kernel(PathToShader.data(), std::ios::binary);
  if (!Kernel.is_open()) {
    std::cerr << "Cannot open file: " << PathToShader << '\n';
    return {};
  }
  std::stringstream Buffer;
  Buffer << Kernel.rdbuf();

  return Buffer.str();
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
