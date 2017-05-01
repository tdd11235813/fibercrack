
#include "cuda_globals.hpp"
#include "application.hpp"
#include "kernel.cuh"
#include "configuration.hpp"

#include <iomanip>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <chrono>


template<typename T>
void cuda_initialize( const Application<T>& _app ) {

  CHECK_CUDA(cudaSetDevice(_app.device_id()));

  if(_app.dump_intermediates()) {
    std::ofstream fs_intermediates;
    fs_intermediates.open("_intermediate_forces", std::ofstream::out);
    fs_intermediates << "\"iteration n_r\", \"iteration n_d\", \"d\", \"iteration ninters\", \"force\"\n";
    fs_intermediates.close();

    fs_intermediates.open("_intermediate_pos_length_orientation", std::ofstream::out);
    fs_intermediates << "\"iteration n_r\", \"iteration n_p\", \"pos.x\",  \"pos.y\", \"pos.z\", \"length\", \"orientation x\", \"orientation y\", \"orientation z\"\n";
    fs_intermediates.close();
  }
}

template<typename T>
void cuda_allocate_data(Data<T>& _data,
                        int _nr_fibers) {
  cuda_malloc(&_data.values, 2*_nr_fibers);
  cuda_malloc(&_data.fibers_find, _nr_fibers);
  cuda_memset(_data.fibers_find, 0, _nr_fibers);
}

template<typename T>
void cuda_free_data(Data<T>& _data) {
  CHECK_CUDA( cudaFree(_data.values_compact) );
  CHECK_CUDA( cudaFree(_data.forces) );
  CHECK_CUDA( cudaFree(_data.values) );
  CHECK_CUDA( cudaFree(_data.fibers_find) );
}



template<typename T>
void run( const Options& _options )
{
  Application<T> app( _options );

  Configuration<T> config = app.configuration();
  std::vector<uint> numbers_fibers(config.nr);

  std::ofstream fs;
  fs.open(app.output_file(), std::ofstream::out);

  if(fs.good()==false)
    throw std::runtime_error("Could not open result file.");

  // csv header
  fs << listCudaDevices().str() << "\n"
     << config
     << std::string(3, '\n')  // 3x '\n'
     << std::string(8, '-') << "\n"
     << "\"iteration n_r\", \"iteration n_d\", \"d\", \"sum force\""
     << "\n";
  fs.close();

  if(app.verbose()) {
    std::cout << "> cuda_initialize\n";
  }
  cuda_initialize( app );

  if(app.verbose()) {
    std::cout << "> cuda_compute_number_of_fibers\n";
  }
  cuda_compute_number_of_fibers(app, numbers_fibers.data());

  fs.open(app.output_file(), std::ofstream::out | std::ofstream::app);
  fs.setf(std::ios::scientific);
  fs.precision(9);

  // we have our poisson numbers in *_h, so loop over them for the main algorithm
  // possible to parallelize over devices/nodes
  for(int j=0; j<config.nr; ++j) {

    uint nr_fibers = numbers_fibers[j];

    if(app.verbose()) {
      std::cout << "> iteration "<<j<<" with "<<nr_fibers << " fiber objects.\n";
    }

    Data<T> data;
    cuda_allocate_data(data, nr_fibers);

    int nr_intersections = cuda_create_and_intersect_fibers(app, data, nr_fibers);
    if(app.verbose()) {
      std::cout << "> ... "<< nr_intersections << " intersections found.\n";
    }


    if(app.dump_intermediates()) {
      std::ofstream fs_intermediates;
      auto host = new typename TVec4<T>::type[2*nr_fibers];
      cuda_memcpy(host, data.values, 2*nr_fibers);
      fs_intermediates.open("_intermediate_pos_length_orientation", std::ofstream::out | std::ofstream::app);
      for(int i=0; i<nr_fibers; ++i)
        fs_intermediates
          << j
          <<", "<< i
          <<", "<< host[i].x
          <<", "<< host[i].y
          <<", "<< host[i].z
          <<", "<< host[i].w
          <<", "<< host[i+nr_fibers].x
          <<", "<< host[i+nr_fibers].y
          <<", "<< host[i+nr_fibers].z
          <<"\n"
          ;
      fs_intermediates.close();
      delete[] host;
    }


    // compactify scattered data to obtain coalesced data
    cuda_compact_data(data, nr_intersections, nr_fibers);


    if(app.verbose()) {
      std::cout << "> enter crack computations\n";
    }
    // n_d times summation of forces with different d
    for( uint k=0; k<config.n_d; ++k ) {

      T dstep = k * config.d_end/(config.n_d-1.0);

      T force = cuda_compute_force(data, config, nr_intersections, nr_fibers, dstep);


      if(app.dump_intermediates()) {
        std::ofstream fs_intermediates;
        T* host = new T[nr_intersections];
        cuda_memcpy(host, data.forces, nr_intersections);
        fs_intermediates.open("_intermediate_forces", std::ofstream::out | std::ofstream::app);
        for(int i=0; i<nr_intersections; ++i)
          fs_intermediates << j << ", " << k << ", " << dstep << ", " << i <<", "<< host[i] << "\n";
        fs_intermediates.close();
        delete[] host;
      }

      fs << std::setw(3) << j
         << ", " << std::setw(3) << k
         << ", " << std::setw(11) << dstep
         << ", " << std::setw(11) << force
         << "\n";

    } // for n_d

    cuda_free_data(data);

  } // for config.nr

  fs.close();
  if(app.verbose()) {
    std::cout << "Result dumped to " << app.output_file() << "\n";
  }
}

int main(int _argc, char** _argv)
{
  Options options(_argc, _argv);

  if(options.help_requested())
    return 0;

  if(options.verbose()) {
    std::cout
      << "\nfibercrack - 05/2017\n\n"
      << listCudaDevices().str() << "\n";
  }

  if(options.double_precision()) {
    run<double>(options);
  } else {
    run<float>(options);
  }

  return 0;
}
