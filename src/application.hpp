#ifndef APPLICATION_H
#define APPLICATION_H

#include "cuda_globals.hpp"
#include "options.hpp"
#include "configuration.hpp"
#include <boost/core/noncopyable.hpp>

template<typename T>
class Application : private boost::noncopyable {
public:

  explicit Application(const Options& _options)
    : Application( _options.configuration<T>(),
                   _options.output_file(),
                   _options.device_id(),
                   _options.dump_intermediates(),
                   _options.verbose()
      ) { }
  
  Application(Configuration<T> _conf,
              std::string _output_file,
              int _device_id,
              bool _dump_intermediates,
              int _verbose
    ) : output_file_(_output_file),
        device_id_(_device_id),
        dump_intermediates_(_dump_intermediates),
        verbose_(_verbose)
    {
    _conf.fis_b = 2.0/(2.0*_conf.fis_k
                                + sqrt(4.0*_conf.fis_k*_conf.fis_k+4.0));
    _conf.fis_x0= (1.0-_conf.fis_b)/(1.0+_conf.fis_b);
    _conf.fis_c = _conf.fis_k*_conf.fis_x0
      + 2.0*log(1.0-_conf.fis_x0*_conf.fis_x0);
    _conf.Af = 0.25 * _conf.fdiam * _conf.diam * PI;
    _conf.NV = _conf.Vf/(_conf.Af*_conf.EL);
    _conf.lambda = _conf.xmax * _conf.ymax * _conf.zmax * _conf.NV;
    _conf.zlevel_abs = _conf.zlevel_rel * _conf.zmax;

    configuration_ = _conf;


    CHECK_CUDA( cudaDeviceGetAttribute(&number_sm_, cudaDevAttrMultiProcessorCount, device_id()) );

  }

  Configuration<T> configuration() const {
    return configuration_;
  }

  std::string output_file() const {
    return output_file_;
  }

  int number_sm() const {
    return number_sm_;
  }

  int device_id() const {
    return device_id_;
  }

  int verbose() const {
    return verbose_;
  }
  
  bool dump_intermediates() const {
    return dump_intermediates_;
  }

private:
  Configuration<T> configuration_;
  std::string output_file_;
  int verbose_ = 0;
  int device_id_ = 0;
  int number_sm_ = 0; // number of streaming multiprocessors
  bool dump_intermediates_ = false;
};

#endif /* APPLICATION_H */
