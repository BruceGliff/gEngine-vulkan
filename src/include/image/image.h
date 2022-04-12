#pragma once

#include <cstdint>
#include <string_view>

class image {
  void *RawData;
  uint32_t Size;

public:
  image(std::string_view Pass);
  ~image();

  void *getRawData() const;
  uint32_t getSize() const;
};
