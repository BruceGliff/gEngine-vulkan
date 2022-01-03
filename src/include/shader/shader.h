#pragma once

#include <string_view>
#include <vector>

class Shader {
  // TODO may be std::byte?
  std::vector<char> RawData{};

public:
  Shader(std::string_view PathToShader);

  Shader() = delete;
  Shader(Shader const &) = delete;
  Shader(Shader &&) = delete;
  Shader &operator=(Shader const &) = delete;
  Shader &operator=(Shader &&) = delete;

  static std::vector<char> readFile(std::string_view PathToShader);
  // Just for VK.
  uint32_t const *getRawData();
  uint32_t const *getRawData() const;
  uint32_t getSize() const;
};
