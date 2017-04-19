#ifndef CONFIGURATION_UTIL_H
#define CONFIGURATION_UTIL_H

#include "configuration.hpp"

#include <boost/program_options.hpp>

#include <stdexcept>
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
std::ostream& operator<<(std::ostream& _fs, const Configuration<T>& _config) {
  return _fs
      << "version = 0.1" << "\n"
      << "lambda = " << _config.lambda << "\n"
      << "nr = " << _config.nr << "\n"
      << "rand_seed = " << _config.rand_seed << "\n"
      << "rand_offset = " << _config.rand_offset << "\n"
      << "xmax = " << _config.xmax << "\n"
      << "ymax = " << _config.ymax << "\n"
      << "zmax = " << _config.zmax << "\n"
      << "zlevel = " << _config.zlevel << "\n"
      << "NV = " << _config.NV << "\n"
      << "mu = " << _config.mu << "\n"
      << "fdiam = " << _config.fdiam << "\n"
      << "sigmafu = " << _config.sigmafu << "\n"
      << "d_end = " << _config.d_end << "\n"
      << "n_d = " << _config.n_d << "\n"
      << "weib_m = " << _config.weib_m << "\n"
      << "weib_xc = " << _config.weib_xc << "\n"
      << "fis_k = " << _config.fis_k << "\n"
      << "fis_b = " << _config.fis_b << "\n"
      << "fis_x0 = " << _config.fis_x0 << "\n"
      << "fis_c = " << _config.fis_c << "\n"
      ;
}

template<typename T>
struct ConfigurationReader
{
  int verbose_ = 0;
  ConfigurationReader(int _verbose) : verbose_(_verbose) {}

  Configuration<T> operator()(const std::string& _fname) {
    namespace po = boost::program_options;

    ConfigurationMapper<T> config_mapper;

    std::ifstream fsconf(_fname);
    if(fsconf.good()==false) {
      fsconf.close();

      std::ofstream fsconf_new(_fname);
      if(fsconf_new.good()) {
        fsconf_new << config_mapper.config_;
        if(verbose_)
          std::cout << "Initial configuration file '"<<_fname<<"' created.\n";
      }else
        throw std::runtime_error("Could not create configuration file.");
    }

    auto vm = po::variables_map();
    po::store(po::parse_config_file(fsconf, config_mapper.desc_), vm);
    po::notify(vm);

    if(verbose_)
      std::cout << "Configuration file '"<<_fname<<"' loaded.\n";

    return config_mapper.config_;
  }

};


#endif /* CONFIGURATION_UTIL_H */
