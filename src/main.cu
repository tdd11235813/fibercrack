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
  bool operator()(const bool x) {
    return x;
  }
};

// for curand_uniform (this uses XORWOW and is faster than Philox4)
template<typename T>
__global__ void d_setup_kernel(int _n,
                               const Parameters<T> _params,
                               curandState* _state)
{
  int i;
  for (i = blockIdx.x * blockDim.x + threadIdx.x;
       i < _n;
       i += blockDim.x * gridDim.x)
  {
    curand_init(_params.rand_seed, i, _params.rand_offset, &_state[i]);
  }
}

// n4 = n/4
template<typename T>
__global__ void d_setup_kernel(int _n4,
                               const Parameters<T> _params    ,
                               curandStatePhilox4_32_10_t* _state)
{
  int i;
  for (i = blockIdx.x * blockDim.x + threadIdx.x;
       i < _n4;
       i += blockDim.x * gridDim.x)
  {
    curand_init(_params.rand_seed, i, _params.rand_offset, &_state[i]);
  }
}

template<typename T>
__global__ void d_generate_poisson_numbers(int _n4,
                                           uint4* _poisson_numbers,
                                           const Parameters<T> _params,
                                           curandStatePhilox4_32_10_t* _state)
{
  int i;
  for (i = blockIdx.x * blockDim.x + threadIdx.x;
       i < _n4;
       i += blockDim.x * gridDim.x)
  {
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
__global__
void d_compute_fibers(Data<T> _data,
                      const Parameters<T> _params,
                      curandState* _states)
{
  int i;
  for (i = blockIdx.x * blockDim.x + threadIdx.x;
       i < _params.np;
       i += blockDim.x * gridDim.x)
  {
    curandState localState = _states[i];

    /// gen_fischer
    T waux,a1,a2;
//    int safe = 10000;
    do
    {
      a1=curand_uniform(&localState);
      a2=curand_uniform(&localState);
      waux=(1.0-(1.0+_params.fis_b)*a1)
        /(1.0-(1.0-_params.fis_b)*a1);
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
      x = -x;
      y = -y;
      z = -z;
    }

    /// rand_weib(NP,l,m,xc)
    T l = pow(genexp(
                pow(_params.weib_xc, _params.weib_m), &localState
                ),
              1.0/_params.weib_m);

//    T beta = acos(z);
    // cos( acos(z) ) = z, z \in [-1,1]
//    if((z>zlevel) && (z-l*cos(beta))<zlevel) {
//      rmin=fmin((z[i]-zlevel)/cos(beta[i]),l[i]-(z[i]-zlevel)/cosx(beta[i]));
//      rmax = l-rmin;
      // store rmin, rmax, beta, l, lambda ..?
    if(z>_params.zlevel && (1.0-l)*z<_params.zlevel) {
      _data.fibers_find[i] = 1;
      _data.values[i].z = z;
      _data.values[i].w = l;
    }
    _states[i] = localState;
  }
}

template<typename T>
__device__ inline
T tau(T d) { return 1.02; }

// try to put loop for _d in here
template<typename T, typename TVec4>
__device__ T d_Psingle(TVec4 _values, T _d, T _ninters, Parameters<T> _params) {
  T sum = 0;
  T z = _values.z;
  T l = _values.w;
  T beta = acos(z);
  T rmin = min( (z-_params.zlevel)/z, l-(z-_params.zlevel)/z);
//  T lambda = PI + atan2(y,x); // if required then x,y buffer needed
  // @todo compute on host where possible
  T Ef, Em, n, u0, sigma0, rhof,aux;
  Ef= 720;
  Em= 500;
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
struct SumValuePair {
  using DataVec4T = typename Data<T>::TVec4T;
  const T ninters_;
  const Parameters<T> params_;
  const DataVec4T dstep_;
  SumValuePair(const DataVec4T _dstep,
               const int _ninters,
               const Parameters<T> _params) :
    dstep_(_dstep), ninters_(_ninters), params_(_params)
    {}
  __device__ __host__
  DataVec4T operator()(const DataVec4T& op1, const DataVec4T& op2) const {
/*    DataVec4T result = {op1.x+op2.x+1,
                        op1.y+op2.y+2,
                        op1.z+op2.z+3,
                        op1.w+op2.w+4};
*/
    DataVec4T result;
    result.x = d_Psingle(op1, dstep_.x, ninters_, params_)
      + d_Psingle(op2, dstep_.x, ninters_, params_);
    result.y = d_Psingle(op1, dstep_.y, ninters_, params_)
      + d_Psingle(op2, dstep_.y, ninters_, params_);
    result.z = d_Psingle(op1, dstep_.z, ninters_, params_)
      + d_Psingle(op2, dstep_.z, ninters_, params_);
    result.w = d_Psingle(op1, dstep_.w, ninters_, params_)
      + d_Psingle(op2, dstep_.w, ninters_, params_);
    return result;
  }
};

template<typename T>
void run( const Parameters<T>& _params,
          const std::string& _filename,
          int _devId )
{
  using DataVec4T = typename Data<T>::TVec4T;
  // parameters not used on device
  std::ofstream fs;
  fs.open(_filename, std::ofstream::out);
  fs << _params.nr << std::endl;
  fs.close();

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

  // ---

  // we have our poisson numbers in *_h, so loop over them for the main algorithm
  // possible to parallelize over devices/nodes
  for(int j=0; j<_params.nr; ++j) {

    Parameters<T> params = _params;
    params.np = reinterpret_cast<uint*>(poisson_numbers_h)[j];

    Data<T> data;
    dim3 threads(128);
    dim3 blocks( 32*numSMs );
    curandState* devStates = nullptr;
    CHECK_CUDA(cudaMalloc(&devStates, params.np * sizeof(curandState)));

    CHECK_CUDA(cudaMalloc(&data.fibers_find, params.np * sizeof(int)));
    CHECK_CUDA(cudaMemset(data.fibers_find, 0, params.np * sizeof(int)));

    CHECK_CUDA(cudaMalloc(&data.values, params.np * sizeof(DataVec4T)));

    d_setup_kernel<<<blocks, threads>>>(params.np, params, devStates);
    CHECK_LAST( "Kernel failure" );
    // main computation, filters intersecting fibers
    d_compute_fibers<<<blocks, threads>>>(data, params, devStates);
    CHECK_LAST( "Kernel failure" );
    CHECK_CUDA(cudaStreamSynchronize(0));
    int ninters = thrust::reduce(thrust::device,
                                 data.fibers_find,
                                 data.fibers_find + params.np,
                                 0,
                                 thrust::plus<int>());
    std::cerr << ninters << " intersections found.\n";
    DataVec4T* values;
    CHECK_CUDA( cudaMalloc(&values, ninters * sizeof(DataVec4T)) );
    // stream compaction (store only values of intersecting fibers)
    thrust::copy_if(thrust::device,
                    data.values,
                    data.values + params.np,
                    data.fibers_find,
                    values,
                    is_flagged());

    fs.open(_filename, std::ofstream::out | std::ofstream::app);
    // n_d times summation of forces
    for( uint k=0; k<params.n_d; k+=4 ) {
      DataVec4T d = { 0.0 };
      if(params.n_d>1) {
        d.x = params.d_end/(params.n_d-1.0)*(k);
        d.y = params.d_end/(params.n_d-1.0)*(k+1);
        d.z = params.d_end/(params.n_d-1.0)*(k+2);
        d.w = params.d_end/(params.n_d-1.0)*(k+3);
      }
      // using 4dim vector we can do 4x times summation
      DataVec4T value_init = {0};
      DataVec4T result = thrust::reduce(thrust::device,
                                        values,
                                        values+ninters,
                                        value_init,
                                        SumValuePair<T>(d,ninters,params));

      fs << k   << "," << result.x << std::endl
         << k+1 << "," << result.y << std::endl
         << k+2 << "," << result.z << std::endl
         << k+3 << "," << result.w << std::endl;
    }
    fs.close();
    std::cout << "Result dumped to " << _filename << std::endl;
    CHECK_CUDA( cudaFree(values) );
    CHECK_CUDA( cudaFree(data.values) );
    CHECK_CUDA( cudaFree(data.fibers_find) );
  } // for _params.nr
  delete[] poisson_numbers_h;
}

int main(int argc, char** argv)
{
  Parameters<float> params;

  params.fis_b = 2.0/(2.0*params.fis_k
                      +sqrt(4.0*params.fis_k*params.fis_k+4.0));
  params.fis_x0= (1.0-params.fis_b)/(1.0+params.fis_b);
  params.fis_c = params.fis_k*params.fis_x0
    +2.0*log(1.0-params.fis_x0*params.fis_x0);
  params.lambda = params.xmax * params.ymax * params.zmax * params.NV;
  params.zlevel = 0.5; //params.zmax * 0.5;

  std::string filename = "dump.csv";
  if(argc>=2)
    params.nr = atoi(argv[1]);
  if(argc>=3)
    filename = argv[2];

  printf("Program start with n=%i and file=%s\n", params.nr, filename.c_str());

  std::cout << listCudaDevices().str();
  int devId = 0;

  run(params, filename, devId);

  return 0;
}



/*
/// @todo we need different random number streams
template<typename T>
__global__
void d_compute_fibers(Data<T> _data,
                      const Parameters<T> _params,
                      curandState* _states)
{
  int i;
  for (i = blockIdx.x * blockDim.x + threadIdx.x;
       i < _params.np;
       i += blockDim.x * gridDim.x)
  {
    curandState localState = _states[i];
    T beta;
    /// init_PPP(NP,x,y,z,xmax,ymax,zmax,&seedinit);
    // - this contained x|y|z * runif(seed)

    /// rand_weib(NP,l,m,xc)
    T l = pow(genexp(pow(_params.xc, _params.m)), 1.0/_params.m);
    /// rand_beta(NP,beta,c,&seedinit);
    if(_params.c==1.0) {
      beta = acos( curand_uniform(&localState) );
    }else if(_params.c<1.0) {
      T c2 = _params.c*_params.c;
      T c6 = c2*c2*c2;
      do {
        T u1 = curand_uniform(&localState);
        T u2 = curand_uniform(&localState);
      } while(u1*u1*(1.0+(c2-1.0)*u2*u2)*
              (1.0+(c2-1.0)*u2*u2)*
              (1.0+(c2-1.0)*u2*u2)>c6);
      beta = acos(u2);
    }else if(_params.c>1.0){
      T c2 = _params.c*_params.c;
      do {
        T u1 = curand_uniform(&localState);
        T u2 = curand_uniform(&localState);
      } while(u1*u1*(1.0+(c2-1.0)*u2*u2)*
              (1.0+(c2-1.0)*u2*u2)*
              (1.0+(c2-1.0)*u2*u2)>1.0);
      beta = acos(u2);
    }
    /// rand_lambda(NP,lambda,&seedinit);
    T lambda = curand_state(&localState) * 2.0 * PI;
    /// NA=((float)calc_Ninters(NP, z, l, beta, zmax/2.0))/(xmax*ymax);
    // - calcs number of intersection, but we just put our flags out here
//    T x = curand_uniform(&localState) * _params.xmax;
//    T y = curand_uniform(&localState) * _params.ymax;
    T z = curand_uniform(&localState) * _params.zmax;
    if((z>zlevel) && (z-l*cos(beta))<zlevel)
      // ...
    _states[i] = localState;
  }
}
*/
