#ifndef OPTIONS_H
#define OPTIONS_H

#include "configuration_util.hpp"

#include <boost/program_options.hpp>
#include <string>
#include <iostream>

class Options {

public:

  explicit Options( int _argc, char** _argv ) {
    namespace po = boost::program_options;
    po::options_description desc = po::options_description("fibercrack options and flags");
    desc.add_options()
      ("help,h", "Print help messages")
      ("file,f", po::value<std::string>(&configuration_file_), "simulation configuration filename (will be created on demand)")
      ("output,o", po::value<std::string>(&output_file_)->default_value("result.csv"), "output csv file, will be overwritten!")
      ("device,i", po::value<int>(&device_id_)->default_value(0), "CUDA device id")
      ("double-precision,d", "use double-precision instead of single-precision")
      ("dump,D", "dump intermediates for verification")
      ("verbose,v", "for console output")
      ;

    po::variables_map vm;
    try {
      po::parsed_options parsed
        = po::command_line_parser( _argc, _argv ).options(desc).allow_unregistered().run();
      po::store(parsed, vm);

      if( vm.count("dump")  ) {
        dump_intermediates_ = true;
      }

      if( vm.count("double-precision")  ) {
        double_precision_ = 1;
      }

      if( vm.count("verbose")  ) {
        verbose_ = 1;
      }

      if( vm.count("help")  ) {
        help_requested_ = true;
      }

      if( !vm.count("file") ) {
        throw po::error("Please provide a configuration file name.");
      }
      po::notify(vm);
    } catch(po::error& e) {
      std::cerr << e.what() << "\n";
      help_requested_ = true;
    }
    if(help_requested_) {
      std::cout << desc << "\n";
    }
  }

  std::string configuration_file() const {
    return configuration_file_;
  }

  template<typename T>
  Configuration<T> configuration() const {
    return ConfigurationReader<T>(verbose())(configuration_file_);
  }

  std::string output_file() const {
    return output_file_;
  }

  bool double_precision() const {
    return double_precision_;
  }

  bool dump_intermediates() const {
    return dump_intermediates_;
  }

  bool help_requested() const {
    return help_requested_;
  }

  int device_id() const {
    return device_id_;
  }

  int verbose() const {
    return verbose_;
  }

private:

  std::string configuration_file_;
  std::string output_file_;
  int device_id_ = 0;
  int verbose_ = 0;
  bool double_precision_ = false;
  bool help_requested_   = false;
  bool dump_intermediates_ = false;
};

#endif /* OPTIONS_H */
