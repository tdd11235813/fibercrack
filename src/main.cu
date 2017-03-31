#include "cuda_globals.hpp"

#include <iostream>
#include <fstream>
#include <chrono>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#ifndef PI
#define PI 3.14159265359
#endif


// for thrust::copy_if
struct is_flagged
{
  __host__ __device__
  bool operator()(const int x) {
    return x==1;
  }
};

// for curand_uniform (this uses XORWOW and is faster than Philox4)
template<typename T>
__global__ void d_setup_kernel(int _n,
                               const Parameters<T> _params,
                               curandState* _state) {
  int i;
  for (i = blockIdx.x * blockDim.x + threadIdx.x;
       i < _n;
       i += blockDim.x * gridDim.x) {
    curand_init(_params.rand_seed, i, _params.rand_offset, &_state[i]);
  }
}

// n4 = n/4
template<typename T>
__global__ void d_setup_kernel(int _n4,
                               const Parameters<T> _params    ,
                               curandStatePhilox4_32_10_t* _state) {
  int i;
  for (i = blockIdx.x * blockDim.x + threadIdx.x;
       i < _n4;
       i += blockDim.x * gridDim.x) {
    curand_init(_params.rand_seed, i, _params.rand_offset, &_state[i]);
  }
}

template<typename T>
__global__ void d_generate_poisson_numbers(int _n4,
                                           uint4* _poisson_numbers,
                                           const Parameters<T> _params,
                                           curandStatePhilox4_32_10_t* _state) {
  int i;
  for (i = blockIdx.x * blockDim.x + threadIdx.x;
       i < _n4;
       i += blockDim.x * gridDim.x) {
    curandStatePhilox4_32_10_t localState = _state[i];
    _poisson_numbers[i] = curand_poisson4(&localState,
                                          _params.lambda);
    _state[i] = localState;
  }
}

template<typename T>
__device__ T genexp(T _lambda, curandState* _state) {
  return -1.0/_lambda * log(1.0-curand_uniform(_state));
}


template<typename T>
__device__
void d_gen_fischer(curandState& localState,
                   const Parameters<T>& _params,
                   typename TVec3<T>::type& orientation
                   // T& beta,
                   // T& lambda
  ) {
  /// gen_fischer
  T waux,a1,a2;
//    int safe = 10000;
  do
  {
    a1=curand_uniform(&localState);
    a2=curand_uniform(&localState);
    waux=(1.0-(1.0+_params.fis_b)*a1) / (1.0-(1.0-_params.fis_b)*a1);
    //    if(--safe<0) break;
  }while(_params.fis_k*waux+2.0*log(1.0-_params.fis_x0*waux)-_params.fis_c >= log(a2));

  T u1 = curand_uniform(&localState);
  T u2 = curand_uniform(&localState);
  T v1=sqrt(-2*log(u1))*cos(2*PI*u2);
  T v2=sqrt(-2*log(u1))*sin(2*PI*u2);
  v1=v1/(sqrt(v1*v1+v2*v2));
  v2=v2/(sqrt(v1*v1+v2*v2));
  T x = sqrt(1.0-waux*waux)*v1;
  T y = sqrt(1.0-waux*waux)*v2;
  T z = waux;
  if(z<0.0) {
    x = -x; // orientation
    y = -y;
    z = -z;
  }
  orientation.x = x;
  orientation.y = y;
  orientation.z = z;
  // beta = acos(z); // calculate later
  // lambda = PI+atan2(y, x);
}

template<bool DumpForIntermediates, typename T>
__global__
void d_compute_fibers(Data<T> _data,
                      const Parameters<T> _params,
                      curandState* _states) {

  int i;
  for (i = blockIdx.x * blockDim.x + threadIdx.x;
       i < _params.np;
       i += blockDim.x * gridDim.x) {
    curandState localState = _states[i];

    /// position
    T x = _params.xmax * curand_uniform(&localState);
    T y = _params.ymax * curand_uniform(&localState);
    T z = _params.zmax * curand_uniform(&localState);


    /// fiber length
    /// rand_weib(NP,l,m,xc)
    T l = pow(genexp(pow(_params.weib_xc, _params.weib_m), &localState),
              1.0/_params.weib_m);

    /// fiber orientation
    typename TVec3<T>::type orientation;
    d_gen_fischer(localState, _params, orientation);

    // if(i==2)
    //   printf("%f %f %f %f\n", z, l, orientation.z, _params.zlevel);

    if(DumpForIntermediates) {
      _data.values[i].x = x;
      _data.values[i].y = y;
      _data.values[i].z = z;
      _data.values[i].w = l;
      _data.values[i+_params.np].x = orientation.x;
      _data.values[i+_params.np].y = orientation.y;
      _data.values[i+_params.np].z = orientation.z;
      if((z>_params.zlevel) && (z-l*orientation.z)<_params.zlevel) {
        _data.fibers_find[i] = 1;
      }
    } else {
      if((z>_params.zlevel) && (z-l*orientation.z)<_params.zlevel) {
//      rmin=fmin((z-zlevel)/orientation.z, l-(z-zlevel)/orientation.z); // ??? compute later
//      rmax = l-rmin;
        _data.values[i].x = x;
        _data.values[i].y = y;
        _data.values[i].z = z;
        _data.values[i].w = l;
        _data.values[i+_params.np].x = orientation.x;
        _data.values[i+_params.np].y = orientation.y;
        _data.values[i+_params.np].z = orientation.z;
        _data.fibers_find[i] = 1;
        // store rmin, rmax, beta, l, lambda ..?
      }
    }
    _states[i] = localState;
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
                       const Parameters<T> _params) {
  T sum = 0;
  T z = _pos.z;
  T l = _pos.w; // fiber length
  T beta = acos(_orientation.z);
//  T lambda = PI + atan2(_orientation.y, _orientation.x);
  T rmin = fmin((z-_params.zlevel)/_orientation.z, l-(z-_params.zlevel)/_orientation.z);
  // rmax = l-rmin; ?
  T Ef, Em, n, u0, sigma0, rhof,aux;
  // @todo use compile-constant expressions?

  Ef= 720.0;
  Em= 500.0;
  n = Ef/Em;
  rhof = _ninters/(4.0/(PI*_params.fdiam*_params.fdiam)+_ninters);
  sigma0 = 4.0*tau(_d)*(rmin)/(_params.fdiam);
  u0  = sigma0*(rmin)/Ef;
  aux = (1.0+n*rhof)/(2.0-n*n*rhof*rhof);

  if (rmin>0 && rmin>_d)
  {
    if(_d<=u0*(1.0+n*rhof))
      sum=PI*_params.fdiam*_params.fdiam/4.0*sigma0*sqrt(_d/u0*(1.0+n*rhof));
    else
      sum=PI*_params.fdiam*_params.fdiam/4.0*sigma0*(rmin/u0*aux-sqrt((rmin/u0*aux)*(rmin/u0*aux)-2.0/u0*(rmin-_d)*aux));
    if( (_d<=u0*(1.0+n*rhof)
       && _d/u0*(1.0+n*rhof) < _params.sigmafu*_params.sigmafu*exp(-2.0*_params.mu*beta))
      || (_d>u0*(1.0+n*rhof) && (sigma0*(1+n*rhof)<_params.sigmafu*exp(-2.0*_params.mu*beta))))
    sum *= exp( _params.mu * beta );
  }
  return sum;
}

template<typename T>
struct ComputeForce {
  using DataVec4T = typename Data<T>::TVec4T;
  const T ninters_;
  const Parameters<T> params_;
  const T dstep_;

  ComputeForce(const T _dstep,
               const int _ninters,
               const Parameters<T> _params)
    : dstep_(_dstep), ninters_(_ninters), params_(_params) {}

  __device__ __host__
  T operator()(const DataVec4T& pos, const DataVec4T& orientation) const {
    return d_Psingle(pos, orientation, dstep_, ninters_, params_);
  }
};

template<typename T>
void run( const Parameters<T>& _params,
          const std::string& _filename,
          int _devId,
          bool _dump_intermediates)
{
  using DataVec4T = typename Data<T>::TVec4T;
  // parameters not used on device
  std::ofstream fs;
  std::ofstream fs_intermediates;
  fs.open(_filename, std::ofstream::out);
  fs << "; \"number repetitions\", " << _params.nr << "\n";
  fs << "\"iteration n_r\", \"iteration n_d\", \"d\", \"sum force\"\n";
  fs.close();


  if(_dump_intermediates) {
    fs_intermediates.open("_intermediate_forces", std::ofstream::out);
    fs_intermediates << "\"iteration n_r\", \"iteration n_d\", \"d\", \"iteration ninters\", \"force\"\n";
    fs_intermediates.close();

    fs_intermediates.open("_intermediate_pos_length_orientation", std::ofstream::out);
    fs_intermediates << "\"iteration n_r\", \"iteration n_p\", \"pos.x\",  \"pos.y\", \"pos.z\", \"length\", \"orientation x\", \"orientation y\", \"orientation z\"\n";
    fs_intermediates.close();
  }

  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, _devId);
  dim3 threads(128);
  dim3 blocks( 8*numSMs );
  CHECK_CUDA(cudaSetDevice(_devId));

  int n4 = (_params.nr+3) / 4;
  uint4* poisson_numbers_d = nullptr;
  uint4* poisson_numbers_h = new uint4[n4];
  curandStatePhilox4_32_10_t* devStatesPhilox = nullptr;
  CHECK_CUDA(cudaMalloc(&devStatesPhilox, n4 * sizeof(curandStatePhilox4_32_10_t)));
  /* Allocate n unsigned ints on device */
  CHECK_CUDA(cudaMalloc(&poisson_numbers_d, n4 * sizeof(uint4)));

  d_setup_kernel<<<blocks, threads>>>(n4, _params, devStatesPhilox);
  d_generate_poisson_numbers<<<blocks, threads>>>(n4,
                                                  poisson_numbers_d,
                                                  _params,
                                                  devStatesPhilox);
  CHECK_LAST( "Kernel failure.");
  CHECK_CUDA( cudaMemcpy(poisson_numbers_h, poisson_numbers_d, n4*sizeof(uint4), cudaMemcpyDeviceToHost) );
  CHECK_CUDA( cudaFree(devStatesPhilox) );
  CHECK_CUDA( cudaFree(poisson_numbers_d) );
  poisson_numbers_d = nullptr;

  // --

  if(_dump_intermediates) {
    uint* ptr = reinterpret_cast<uint*>(poisson_numbers_h);
    fs_intermediates.open("_intermediate_numbers_for_np", std::ofstream::out);
    for(int i=0; i<_params.nr; ++i)
      fs_intermediates << i <<", "<< *ptr++ << "\n";
    fs_intermediates.close();
  }

  // ---

  fs.open(_filename, std::ofstream::out | std::ofstream::app);

  // we have our poisson numbers in *_h, so loop over them for the main algorithm
  // possible to parallelize over devices/nodes
  for(int j=0; j<_params.nr; ++j) {

    Parameters<T> params = _params; // local params with np value computed before
    params.np = reinterpret_cast<uint*>(poisson_numbers_h)[j];

    std::cerr << params.np << " fiber objects (params.np).\n";

    Data<T> data;
    dim3 threads(128);
    dim3 blocks( 32*numSMs );
    curandState* devStates = nullptr;

    CHECK_CUDA(cudaMalloc(&devStates, params.np * sizeof(curandState)));
    CHECK_CUDA(cudaMalloc(&data.values, 2*params.np * sizeof(DataVec4T)));
    CHECK_CUDA(cudaMalloc(&data.fibers_find, params.np * sizeof(int)));
    CHECK_CUDA(cudaMemset(data.fibers_find, 0, params.np * sizeof(int)));

    d_setup_kernel<<<blocks, threads>>>(params.np, params, devStates);
    CHECK_LAST( "Kernel failure" );

    // main computation, filters intersecting fibers
    if(_dump_intermediates)
      d_compute_fibers<true><<<blocks, threads>>>(data, params, devStates);
    else
      d_compute_fibers<false><<<blocks, threads>>>(data, params, devStates);
    CHECK_LAST( "Kernel failure" );

    CHECK_CUDA(cudaStreamSynchronize(0));

    // --

    if(_dump_intermediates) {
      DataVec4T* host = new DataVec4T[2*params.np];
      CHECK_CUDA(cudaMemcpy(host, data.values, 2*params.np*sizeof(DataVec4T), cudaMemcpyDeviceToHost));
      fs_intermediates.open("_intermediate_pos_length_orientation", std::ofstream::out | std::ofstream::app);
      for(int i=0; i<params.np; ++i)
        fs_intermediates
          << j
          <<", "<< i
          <<", "<< host[i].x
          <<", "<< host[i].y
          <<", "<< host[i].z
          <<", "<< host[i].w
          <<", "<< host[i+params.np].x
          <<", "<< host[i+params.np].y
          <<", "<< host[i+params.np].z
          <<"\n"
          ;
      fs_intermediates.close();
      delete[] host;
    }

    // ---

    int ninters = thrust::reduce(thrust::device,
                                 data.fibers_find,
                                 data.fibers_find + params.np,
                                 0,
                                 thrust::plus<int>());

    std::cerr << ninters << " intersections found.\n";

    DataVec4T* values;
    T* forces;
    CHECK_CUDA( cudaMalloc(&values, 2*ninters * sizeof(DataVec4T)) );
    CHECK_CUDA( cudaMalloc(&forces, ninters * sizeof(T)) );
    // stream compaction (store only values of intersecting fibers)
    thrust::copy_if(thrust::device,
                    data.values,
                    data.values + params.np,
                    data.fibers_find,
                    values,
                    is_flagged());
    // second part (orientation, ..)
    thrust::copy_if(thrust::device,
                    data.values + params.np,
                    data.values + 2*params.np,
                    data.fibers_find,
                    values + ninters,
                    is_flagged());

    // n_d times summation of forces with different d
    for( uint k=0; k<params.n_d; ++k ) {
      T dstep = k * params.d_end/(params.n_d-1.0);
      //
      thrust::transform(thrust::device,
                        values,         //first1
                        values+ninters, //last1
                        values+ninters, //first2
                        forces,
                        ComputeForce<T>(dstep, ninters, params));

      // --
      if(_dump_intermediates) {
        T* host = new T[ninters];
        CHECK_CUDA(cudaMemcpy(host, forces, ninters*sizeof(T), cudaMemcpyDeviceToHost));
        fs_intermediates.open("_intermediate_forces", std::ofstream::out | std::ofstream::app);
        for(int i=0; i<ninters; ++i)
          fs_intermediates << j << ", " << k << ", " << dstep << ", " << i <<", "<< host[i] << "\n";
        fs_intermediates.close();
        delete[] host;
      }
      // ---

      T value_init = 0;
      T result = thrust::reduce(thrust::device,
                                forces,
                                forces+ninters,
                                value_init);

      fs << j << ", " << k << ", " << dstep << ", " << result << "\n";
      //std::cerr << "Force: " << result << "\n";
    } // for n_d
    CHECK_CUDA( cudaFree(values) );
    CHECK_CUDA( cudaFree(forces) );
    CHECK_CUDA( cudaFree(data.values) );
    CHECK_CUDA( cudaFree(data.fibers_find) );
  } // for _params.nr
  fs.close();
  std::cout << "Result dumped to " << _filename << "\n";
  delete[] poisson_numbers_h;
}

int main(int argc, char** argv)
{
  Parameters<float> params;
  bool dump_intermediates = false;
  params.fis_b = 2.0/(2.0*params.fis_k
                      +sqrt(4.0*params.fis_k*params.fis_k+4.0));
  params.fis_x0= (1.0-params.fis_b)/(1.0+params.fis_b);
  params.fis_c = params.fis_k*params.fis_x0
    +2.0*log(1.0-params.fis_x0*params.fis_x0);
  params.lambda = params.xmax * params.ymax * params.zmax * params.NV;

  std::string filename = "dump.csv";
  if(argc>=2)
    params.nr = atoi(argv[1]);
  if(argc>=3)
    filename = argv[2];
  if(argc>=4)
    params.zlevel = atof(argv[3]);
  if(argc>=5)
    dump_intermediates = atoi(argv[4]);

  printf("Program start with n=%i and file=%s\n", params.nr, filename.c_str());

  std::cout << listCudaDevices().str();
  int devId = 0;

  run(params, filename, devId, dump_intermediates);

  return 0;
}

