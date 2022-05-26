#include "gEng/environment.h"

using namespace gEng;

SysEnv::SysEnv(char const *PathToBin)
    : BinPath{fs::path{PathToBin}.remove_filename()},
      FileName{fs::path{PathToBin}.filename()} {}

std::string SysEnv::getPathStr() const {
  return BinPath.lexically_normal().string();
}

std::string SysEnv::getFilenameStr() const {
  return FileName.lexically_normal().string();
}

SysEnv::operator fs::path const &() const { return BinPath; }

SysEnv::operator fs::path &() { return BinPath; }
