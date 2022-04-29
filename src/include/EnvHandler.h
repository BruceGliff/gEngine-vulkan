#pragma once

#include <filesystem>
#include <string>

namespace fs {
using path = std::filesystem::path;
} // namespace fs

struct EnvHandler {
  EnvHandler(char const *);

  std::string getPathStr() const;
  std::string getFilenameStr() const;
  operator fs::path const &() const;
  operator fs::path &();

private:
  fs::path BinPath;
  fs::path FileName;
};
