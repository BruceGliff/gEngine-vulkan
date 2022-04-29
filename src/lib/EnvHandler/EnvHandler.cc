#include "EnvHandler.h"

EnvHandler::EnvHandler(char const *PathToBin)
    : BinPath{fs::path{PathToBin}.remove_filename()},
      FileName{fs::path{PathToBin}.filename()} {}

std::string EnvHandler::getPathStr() const {
  return BinPath.lexically_normal().string();
}

std::string EnvHandler::getFilenameStr() const {
  return FileName.lexically_normal().string();
}

EnvHandler::operator fs::path const &() const { return BinPath; }

EnvHandler::operator fs::path &() { return BinPath; }
