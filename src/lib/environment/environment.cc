#include "gEng/environment.h"

#include <iostream>

using namespace gEng;
namespace opt = boost::program_options;

SysEnv::SysEnv(int argc, char *argv[])
    : BinPath{fs::path{argv[0]}.remove_filename()},
      FileName{fs::path{argv[0]}.filename()} {

  Desc.add_options()("help", "produce help message")(
      "frames-limit", opt::value<uint32_t>(),
      "set number of frames to render.");

  opt::store(opt::parse_command_line(argc, argv, Desc), Opts);
  opt::notify(Opts);
}

bool SysEnv::getHelp() const {
  if (Opts.count("help")) {
    std::cout << Desc << "\n";
    return true;
  }
  return false;
}

std::optional<uint32_t> SysEnv::getFramesLimit() const {
  if (Opts.count("frames-limit"))
    return Opts["frames-limit"].as<uint32_t>();
  return {};
}

std::string SysEnv::getPathStr() const {
  return BinPath.lexically_normal().string();
}

std::string SysEnv::getFilenameStr() const {
  return FileName.lexically_normal().string();
}

SysEnv::operator fs::path const &() const { return BinPath; }

SysEnv::operator fs::path &() { return BinPath; }
