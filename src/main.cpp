
#include "cuda_globals.hpp"
#include "application.hpp"
#include "kernel.cuh"
#include "configuration.hpp"

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
void run( const Options& options )
{
  Application<T> app( options );

  Configuration<T> config = app.configuration();
  std::vector<uint> numbers_fibers(config.nr);

  std::ofstream fs;
  fs.open(app.output_file(), std::ofstream::out);
  fs << "; \"number repetitions\", " << config.nr << "\n";
  fs << "\"iteration n_r\", \"iteration n_d\", \"d\", \"sum force\"\n";
  fs.close();

  cuda_initialize( app );

  cuda_compute_number_of_fibers(app, numbers_fibers.data());

  fs.open(app.output_file(), std::ofstream::out | std::ofstream::app);

  // we have our poisson numbers in *_h, so loop over them for the main algorithm
  // possible to parallelize over devices/nodes
  for(int j=0; j<config.nr; ++j) {

    uint nr_fibers = numbers_fibers[j];
    std::cerr << nr_fibers << " fiber objects.\n";

    Data<T> data;
    cuda_allocate_data(data, nr_fibers);

    int nr_intersections = cuda_create_and_intersect_fibers(app, data, nr_fibers);
    std::cerr << nr_intersections << " intersections found.\n";


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



      fs << j << ", " << k << ", " << dstep << ", " << force << "\n";

    } // for n_d

    cuda_free_data(data);

  } // for config.nr

  fs.close();
  std::cout << "Result dumped to " << app.output_file() << "\n";
}

int main(int argc, char** argv)
{

  std::cout
    << "\nfibercrack - 04/2017\n\n"
    << listCudaDevices().str() << "\n";

  Options options(argc, argv);

  if(options.help_requested())
    return 0;

  if(options.double_precision()) {
    run<double>(options);
  } else {
    run<float>(options);
  }

  return 0;
}
