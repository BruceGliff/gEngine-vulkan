#pragma once

#include <string>
#include <string_view>

class Shader {
  // TODO may be std::byte?
  std::string RawData{};

public:
  Shader(std::string_view PathToShader);

  Shader() = delete;
  Shader(Shader const &) = delete;
  Shader(Shader &&) = delete;
  Shader &operator=(Shader const &) = delete;
  Shader &operator=(Shader &&) = delete;

  static std::string readFile(std::string_view PathToShader);
  // Just for VK.
  uint32_t const *getRawData();
  uint32_t const *getRawData() const;
  uint32_t getSize() const;
};
