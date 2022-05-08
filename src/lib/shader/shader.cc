#include "shader/shader.h"

#include <fstream>
#include <iostream>

#include <cassert>

std::vector<uint32_t> Shader::readFile(std::string_view PathToShader) {
  std::ifstream Kernel{PathToShader.data(), std::ios::binary};
  if (!Kernel.is_open()) {
    std::cerr << "Cannot open file: " << PathToShader << '\n';
    return {};
  }

  Kernel.seekg(0, std::ios_base::end);
  std::size_t const Size = Kernel.tellg();
  assert(Size % 4 == 0 && "SPIRV size has to be 4-multiple");
  Kernel.seekg(0, std::ios_base::beg);
  std::vector<uint32_t> SPIRV(Size / sizeof(uint32_t));
  Kernel.read((char *)&SPIRV[0], Size);
  return SPIRV;
};

Shader::Shader(std::string_view PathToShader)
    : RawData{readFile(PathToShader)} {}

// Seems like defailt allocator already.
// SPIR-V declares that binary file has to be multiple by 4byte.
std::vector<uint32_t> const &Shader::getSPIRV() const { return RawData; }

uint32_t Shader::getSize() const { return RawData.size() * 4; }
