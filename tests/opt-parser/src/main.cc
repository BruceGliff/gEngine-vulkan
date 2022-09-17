#include <boost/program_options.hpp>
#include <iostream>

namespace po = boost::program_options;
using namespace std;

int main(int argc, char *argv[]) {

  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()("help", "produce help message")(
      "random_number", po::value<int>(), "set random number");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    cout << desc << "\n";
    return 1;
  }

  if (vm.count("random_number")) {
    cout << "Compression level was set to " << vm["random_number"].as<int>()
         << ".\n";
  } else {
    cout << "Random number was not set.\n";
  }
  return 0;
}
