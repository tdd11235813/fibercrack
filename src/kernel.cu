#include "cuda_globals.hpp"
#include "application.hpp"
#include "kernel.cuh"
#include "configuration.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>

#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include <iostream>
#include <fstream>
#include <chrono>

#ifndef PI
#define PI 3.14159265359
#endif

// -------------
//  Kernels
// -------------

// for curand_uniform (this uses XORWOW and is faster than Philox4)
__global__ void d_setup_kernel(int _n,
                               int _rand_seed,
                               int _rand_offset,
                               curandState* _state) {
  int i;
  for (i = blockIdx.x * blockDim.x + threadIdx.x;
       i < _n;
       i += blockDim.x * gridDim.x) {
    curand_init(_rand_seed, i, _rand_offset, &_state[i]);
  }
}

// n4 = n/4
__global__ void d_setup_kernel(int _n4,
                               int _rand_seed,
                               int _rand_offset,
                               curandStatePhilox4_32_10_t* _state) {
  int i;
  for (i = blockIdx.x * blockDim.x + threadIdx.x;
       i < _n4;
       i += blockDim.x * gridDim.x) {
    curand_init(_rand_seed, i, _rand_offset, &_state[i]);
  }
}

template<typename T>
__global__ void d_generate_poisson_numbers(int _n4,
                                           uint4* _poisson_numbers,
                                           T _lambda,
                                           curandStatePhilox4_32_10_t* _state) {
  int i;
  for (i = blockIdx.x * blockDim.x + threadIdx.x;
       i < _n4;
       i += blockDim.x * gridDim.x) {
    curandStatePhilox4_32_10_t local_state = _state[i];
    _poisson_numbers[i] = curand_poisson4(&local_state,
                                          _lambda);
    _state[i] = local_state;
  }
}

template<typename T>
__device__ T genexp(T _lambda, curandState* _state) {
  return -1.0/_lambda * log(1.0-curand_uniform(_state));
}


template<typename T>
__device__
void d_gen_fischer(curandState& _local_state,
                   const Configuration<T>& _params,
                   typename TVec3<T>::type& _orientation
                   // T& beta,
                   // T& lambda
  ) {
  /// gen_fischer
  T waux,a1,a2;
//    int safe = 10000;
  do
  {
    a1=curand_uniform(&_local_state);
    a2=curand_uniform(&_local_state);
    waux=(1.0-(1.0+_params.fis_b)*a1) / (1.0-(1.0-_params.fis_b)*a1);
    //    if(--safe<0) break;
  }while(_params.fis_k*waux+2.0*log(1.0-_params.fis_x0*waux)-_params.fis_c >= log(a2));

  T u1 = curand_uniform(&_local_state);
  T u2 = curand_uniform(&_local_state);
  T v1=sqrt(-2*log(u1))*cos(2*PI*u2);
  T v2=sqrt(-2*log(u1))*sin(2*PI*u2);
  v1=v1/(sqrt(v1*v1+v2*v2));
  v2=v2/(sqrt(v1*v1+v2*v2));
  T x = sqrt(1.0-waux*waux)*v1;
  T y = sqrt(1.0-waux*waux)*v2;
  T z = waux;
  if(z<0.0) {
    x = -x; // _orientation
    y = -y;
    z = -z;
  }
  _orientation.x = x;
  _orientation.y = y;
  _orientation.z = z;
  // beta = acos(z); // calculate later
  // lambda = PI+atan2(y, x);
}

template<bool DumpForIntermediates, typename T, typename U>
__global__
void d_compute_fibers(int _nf,
                      typename TVec4<T>::type* _values,
                      U* _fibers_find,
                      const Configuration<T> _config,
                      curandState* _states) {

  int i;
  for (i = blockIdx.x * blockDim.x + threadIdx.x;
       i < _nf;
       i += blockDim.x * gridDim.x) {
    curandState local_state = _states[i];

    /// position
    T x = _config.xmax * curand_uniform(&local_state);
    T y = _config.ymax * curand_uniform(&local_state);
    T z = _config.zmax * curand_uniform(&local_state);


    /// fiber length
    /// rand_weib(NP,l,m,xc)
    T l = pow(genexp(pow(_config.weib_xc, _config.weib_m), &local_state),
              1.0/_config.weib_m);

    /// fiber orientation
    typename TVec3<T>::type orientation;
    d_gen_fischer(local_state, _config, orientation);

    // if(i==2)
    //   printf("%f %f %f %f\n", z, l, orientation.z, _config.zlevel);

    if(DumpForIntermediates) {
      _values[i].x = x;
      _values[i].y = y;
      _values[i].z = z;
      _values[i].w = l;
      _values[i+_nf].x = orientation.x;
      _values[i+_nf].y = orientation.y;
      _values[i+_nf].z = orientation.z;
      if((z>_config.zlevel) && (z-l*orientation.z)<_config.zlevel) {
        _fibers_find[i] = 1;
      }
    } else {
      if((z>_config.zlevel) && (z-l*orientation.z)<_config.zlevel) {
//      rmin=fmin((z-zlevel)/orientation.z, l-(z-zlevel)/orientation.z); // ??? compute later
//      rmax = l-rmin;
        _values[i].x = x;
        _values[i].y = y;
        _values[i].z = z;
        _values[i].w = l;
        _values[i+_nf].x = orientation.x;
        _values[i+_nf].y = orientation.y;
        _values[i+_nf].z = orientation.z;
        _fibers_find[i] = 1;
        // store rmin, rmax, beta, l, lambda ..?
      }
    }
    _states[i] = local_state;
  }
}

template<typename T>
__device__ inline
T tau(T d) { return 1.02; }

// try to put loop for _d in here
template<typename T, typename TVec4>
__device__ T d_Psingle(const TVec4& _pos,
                       const TVec4& _orientation,
                       T _d,
                       T _ninters,
                       const Configuration<T> _config) {
  T sum = 0;
  T z = _pos.z;
  T l = _pos.w; // fiber length
  T beta = acos(_orientation.z);
//  T lambda = PI + atan2(_orientation.y, _orientation.x);
  T rmin = fmin((z-_config.zlevel)/_orientation.z, l-(z-_config.zlevel)/_orientation.z);
  // rmax = l-rmin; ?
  T Ef, Em, n, u0, sigma0, rhof,aux;
  // @todo use compile-constant expressions?

  Ef= 720.0;
  Em= 500.0;
  n = Ef/Em;
  rhof = _ninters/(4.0/(PI*_config.fdiam*_config.fdiam)+_ninters);
  sigma0 = 4.0*tau(_d)*(rmin)/(_config.fdiam);
  u0  = sigma0*(rmin)/Ef;
  aux = (1.0+n*rhof)/(2.0-n*n*rhof*rhof);

  if (rmin>0 && rmin>_d)
  {
    if(_d<=u0*(1.0+n*rhof))
      sum=PI*_config.fdiam*_config.fdiam/4.0*sigma0*sqrt(_d/u0*(1.0+n*rhof));
    else
      sum=PI*_config.fdiam*_config.fdiam/4.0*sigma0*(rmin/u0*aux-sqrt((rmin/u0*aux)*(rmin/u0*aux)-2.0/u0*(rmin-_d)*aux));
    if( (_d<=u0*(1.0+n*rhof)
       && _d/u0*(1.0+n*rhof) < _config.sigmafu*_config.sigmafu*exp(-2.0*_config.mu*beta))
      || (_d>u0*(1.0+n*rhof) && (sigma0*(1+n*rhof)<_config.sigmafu*exp(-2.0*_config.mu*beta))))
    sum *= exp( _config.mu * beta );
  }
  return sum;
}



// for thrust::copy_if
struct is_flagged
{
  __device__
  bool operator()(const int x) {
    return x==1;
  }
};


template<typename T>
struct ComputeForce {
  using DataVec4T = typename Data<T>::TVec4T;
  const T ninters_;
  const Configuration<T> config_;
  const T dstep_;

  ComputeForce(const T _dstep,
               const int _ninters,
               const Configuration<T> _config)
    : dstep_(_dstep), ninters_(_ninters), config_(_config) {}

  __device__
  T operator()(const DataVec4T& pos, const DataVec4T& orientation) const {
    return d_Psingle(pos, orientation, dstep_, ninters_, config_);
  }
};



// ----------------
//  Host functions
// ----------------


template<typename T>
void cuda_compute_number_of_fibers(const Application<T>& _app,
                                   uint* _output ) {

  Configuration<T> config = _app.configuration();
  dim3 threads(128);
  dim3 blocks( 8*_app.number_sm() );

  int n4 = (config.nr+3) / 4;
  uint4* poisson_numbers_d = nullptr;
  curandStatePhilox4_32_10_t* devStatesPhilox = nullptr;
  cuda_malloc(&devStatesPhilox, n4);
  cuda_malloc(&poisson_numbers_d, n4);

  d_setup_kernel<<<blocks, threads>>>(n4,
                                      config.rand_seed,
                                      config.rand_offset,
                                      devStatesPhilox);

  d_generate_poisson_numbers<<<blocks, threads>>>(n4,
                                                  poisson_numbers_d,
                                                  config.lambda,
                                                  devStatesPhilox);
  CHECK_LAST( "Kernel failure.");
  cuda_memcpy(_output, reinterpret_cast<uint*>(poisson_numbers_d), config.nr);
  CHECK_CUDA( cudaFree(devStatesPhilox) );
  CHECK_CUDA( cudaFree(poisson_numbers_d) );
  poisson_numbers_d = nullptr;


  if(_app.dump_intermediates()) {
    std::ofstream fs_intermediates;
    uint* ptr = _output;
    fs_intermediates.open("_intermediate_numbers_for_np", std::ofstream::out);
    for(int i=0; i<config.nr; ++i) {
      fs_intermediates << i <<", "<< *ptr++ << "\n";
    }
    fs_intermediates.close();
  }

}

template<typename T>
int cuda_create_and_intersect_fibers(const Application<T>& _app,
                                     Data<T>& _data,
                                     int _nr_fibers) {

  Configuration<T> config = _app.configuration();
  dim3 threads(128);
  dim3 blocks( 32*_app.number_sm() );

  curandState* devStates = nullptr;
  cuda_malloc(&devStates, _nr_fibers);

  d_setup_kernel<<<blocks, threads>>>(_nr_fibers,
                                      config.rand_seed,
                                      config.rand_offset,
                                      devStates);
  CHECK_LAST( "Kernel failure" );

  // main computation, filters intersecting fibers
  if(_app.dump_intermediates())
    d_compute_fibers<true><<<blocks, threads>>>(_nr_fibers, _data.values, _data.fibers_find, config, devStates);
  else
    d_compute_fibers<false><<<blocks, threads>>>(_nr_fibers, _data.values, _data.fibers_find, config, devStates);
  CHECK_LAST( "Kernel failure" );

  CHECK_CUDA(cudaStreamSynchronize(0));

  CHECK_CUDA( cudaFree(devStates) );

  int ninters = thrust::reduce(thrust::device,
                               _data.fibers_find,
                               _data.fibers_find + _nr_fibers,
                               0,
                               thrust::plus<int>());
  return ninters;
}


template<typename T>
void cuda_compact_data(Data<T>& _data,
                        int _nr_intersections,
                        int _nr_fibers) {
  cuda_malloc(&_data.values_compact, 2*_nr_intersections);
  cuda_malloc(&_data.forces, _nr_intersections);
  // stream compaction (store only values of intersecting fibers)
  thrust::copy_if(thrust::device,
                  _data.values,
                  _data.values + _nr_fibers,
                  _data.fibers_find,
                  _data.values_compact,
                  is_flagged());
  // second part (orientation, ..)
  thrust::copy_if(thrust::device,
                  _data.values + _nr_fibers,
                  _data.values + 2*_nr_fibers,
                  _data.fibers_find,
                  _data.values_compact + _nr_intersections,
                  is_flagged());
}

template<typename T>
T cuda_compute_force(Data<T>& _data,
                     const Configuration<T>& _config,
                     int _nr_intersections,
                     int _nr_fibers,
                     T _dstep) {
  // inplace transform
  thrust::transform(thrust::device,
                    _data.values_compact,         //first1
                    _data.values_compact + _nr_intersections, //last1
                    _data.values_compact + _nr_intersections, //first2
                    _data.forces,
                    ComputeForce<T>(_dstep, _nr_intersections, _config));

  T value_init = 0;
  T result = thrust::reduce(thrust::device,
                            _data.forces,
                            _data.forces+_nr_intersections,
                            value_init);
  return result;
}



template
void cuda_compute_number_of_fibers<float>(const Application<float>& _app,
                                   uint* _output );

template
int cuda_create_and_intersect_fibers<float>(const Application<float>& _app,
                                     Data<float>& _data,
                                     int _nr_fibers);

template
void cuda_compact_data<float>(Data<float>& _data,
                       int _nr_intersections,
                       int _nr_fibers);


template
float cuda_compute_force<float>(Data<float>& _data,
                     const Configuration<float>& _config,
                     int _nr_intersections,
                     int _nr_fibers,
                     float _dstep);


template
void cuda_compute_number_of_fibers<double>(const Application<double>& _app,
                                   uint* _output );

template
int cuda_create_and_intersect_fibers<double>(const Application<double>& _app,
                                     Data<double>& _data,
                                     int _nr_fibers);

template
void cuda_compact_data<double>(Data<double>& _data,
                       int _nr_intersections,
                       int _nr_fibers);


template
double cuda_compute_force<double>(Data<double>& _data,
                     const Configuration<double>& _config,
                     int _nr_intersections,
                     int _nr_fibers,
                     double _dstep);
