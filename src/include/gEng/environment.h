#pragma once

#include <boost/program_options.hpp>

#include "gEng/utils/singleton.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace fs {
using path = std::filesystem::path;
} // namespace fs

namespace gEng {

struct SysEnv : public singleton<SysEnv> {
  friend singleton;

  void init(int argc, char *argv[]);

  std::string getPathStr() const;
  std::string getFilenameStr() const;
  operator fs::path const &() const;
  operator fs::path &();

  bool getHelp() const;
  std::optional<uint32_t> getFramesLimit() const;

private:
  SysEnv() {}
  boost::program_options::options_description Desc{"Allowed options"};
  boost::program_options::variables_map Opts;
  fs::path BinPath;
  fs::path FileName;
};

} // namespace gEng
