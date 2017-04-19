#ifndef CONFIGURATION_UTIL_H
#define CONFIGURATION_UTIL_H

#include "configuration.hpp"

#include <boost/program_options.hpp>

#include <iostream>
#include <fstream>
#include <string>


template<typename T>
struct ConfigurationMapper {

  Configuration<T> config_;
  boost::program_options::options_description desc_;
  double version_ = 0.0;

  ConfigurationMapper() {
    namespace po = boost::program_options;

    // regexp from config to this: ^ *\([A-Za-z]+\) \([A-Za-z_0-9]+\) *=.* -> ("\2", po::value<\1>(&config_.\2), "\2")

    desc_.add_options()
      ("version", po::value<double>(&version_), "version")
      ("lambda", po::value<T>(&config_.lambda), "lambda")
      ("nr", po::value<uint>(&config_.nr), "nr")
      ("rand_seed", po::value<int>(&config_.rand_seed), "rand_seed")
      ("rand_offset", po::value<int>(&config_.rand_offset), "rand_offset")
      ("xmax", po::value<T>(&config_.xmax), "xmax")
      ("ymax", po::value<T>(&config_.ymax), "ymax")
      ("zmax", po::value<T>(&config_.zmax), "zmax")
      ("zlevel", po::value<T>(&config_.zlevel), "zlevel")
      ("NV", po::value<T>(&config_.NV), "NV")
      ("mu", po::value<T>(&config_.mu), "mu")
      ("fdiam", po::value<T>(&config_.fdiam), "fdiam")
      ("sigmafu", po::value<T>(&config_.sigmafu), "sigmafu")
      ("d_end", po::value<T>(&config_.d_end), "d_end")
      ("n_d", po::value<uint>(&config_.n_d), "n_d")
      ("weib_m", po::value<T>(&config_.weib_m), "weib_m")
      ("weib_xc", po::value<T>(&config_.weib_xc), "weib_xc")
      ("fis_k", po::value<T>(&config_.fis_k), "fis_k")
      ("fis_b", po::value<T>(&config_.fis_b), "fis_b")
      ("fis_x0", po::value<T>(&config_.fis_x0), "fis_x0")
      ("fis_c", po::value<T>(&config_.fis_c), "fis_c")
      ;
  }
};

template<typename T>
struct ConfigurationWriter
{
  Configuration<T> config_;

  ConfigurationWriter(Configuration<T> _config)
    : config_(_config) {}

  void operator()(const std::string& fname) {
    std::ofstream fsconf(fname);

    fsconf
      << "version = 0.1" << "\n"
      << "lambda = " << config_.lambda << "\n"
      << "nr = " << config_.nr << "\n"
      << "rand_seed = " << config_.rand_seed << "\n"
      << "rand_offset = " << config_.rand_offset << "\n"
      << "xmax = " << config_.xmax << "\n"
      << "ymax = " << config_.ymax << "\n"
      << "zmax = " << config_.zmax << "\n"
      << "zlevel = " << config_.zlevel << "\n"
      << "NV = " << config_.NV << "\n"
      << "mu = " << config_.mu << "\n"
      << "fdiam = " << config_.fdiam << "\n"
      << "sigmafu = " << config_.sigmafu << "\n"
      << "d_end = " << config_.d_end << "\n"
      << "n_d = " << config_.n_d << "\n"
      << "weib_m = " << config_.weib_m << "\n"
      << "weib_xc = " << config_.weib_xc << "\n"
      << "fis_k = " << config_.fis_k << "\n"
      << "fis_b = " << config_.fis_b << "\n"
      << "fis_x0 = " << config_.fis_x0 << "\n"
      << "fis_c = " << config_.fis_c << "\n"
      ;
  }

};

template<typename T>
struct ConfigurationReader
{

  Configuration<T> operator()(const std::string& fname) {
    namespace po = boost::program_options;

    ConfigurationMapper<T> config_mapper;
    std::ifstream fsconf(fname);

    if(fsconf.good()==false) {
      ConfigurationWriter<T>(config_mapper.config_)(fname);
    }

    auto vm = po::variables_map();
    po::store(po::parse_config_file(fsconf, config_mapper.desc_), vm);
    po::notify(vm);

    return config_mapper.config_;
  }

};


#endif /* CONFIGURATION_UTIL_H */
