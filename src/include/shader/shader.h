#pragma once

#include <string_view>
#include <vector>

class Shader {
  // TODO may be std::byte?
  std::vector<uint32_t> RawData{};

public:
  Shader(std::string_view PathToShader);

  Shader() = delete;
  Shader(Shader const &) = delete;
  Shader(Shader &&) = delete;
  Shader &operator=(Shader const &) = delete;
  Shader &operator=(Shader &&) = delete;

  static std::vector<uint32_t> readFile(std::string_view PathToShader);
  // Just for VK.
  std::vector<uint32_t> const &getSPIRV() const;
  uint32_t getSize() const;
};
