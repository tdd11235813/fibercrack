#ifndef __CUDA_GLOBALS_H
#define __CUDA_GLOBALS_H

#include <sstream>
#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <curand.h>

// ----------------------------------------------------------------------------
template<typename T=float>
struct TVec3 {
  using type = float3;
};
template<>
struct TVec3<double> {
  using type = double3;
};

template<typename T=float>
struct TVec4 {
  using type = float4;
};
template<>
struct TVec4<double> {
  using type = double4;
};

template<typename T>
struct Data
{
  using TVec4T = typename TVec4<T>::type;
  int* fibers_find = nullptr;
  TVec4T* values; // .., .., zpos, len
};

/// Parameters used for kernels
template<typename T>
struct Parameters
{
  double lambda = 11e+7; // will be xmax*ymax*zmax*NV
  uint np = 0; // number from poisson process
  uint nr = 4; // number repetitions
  int rand_seed = 1337;
  int rand_offset = 0;

  T xmax=100.0;         //in mm
  T ymax=100.0;         //  ""
  T zmax=100.0;         //  ""
  T zlevel = 50.0;       // eg zmax/2.0
  T NV=0.0333;          // in 1/mm³
  T mu=0.7;           //in rad^(-1)
  T fdiam=0.2;       // in mm
  T sigmafu=2500.0;      // in MPA
  T d_end=0.3;                //in mm (changed from 3 to 0.3)
  uint n_d=200; //Anzahl der Schrittzahl bei der Spaltöffnung

//  T beta_c = 2.0;
  T weib_m = 10.0;
  T weib_xc = 10.0;

  T fis_k = 20.0;
  T fis_b;// = 2.0/(2.0*k+sqrt(4.0*_params.k*_params.k+4.0));
  T fis_x0;//= (1.0-b)/(1.0+b);
  T fis_c;// = _params.k*x0+2.0*log(1.0-x0*x0);
};

// ----------------------------------------------------------------------------


#ifndef CUDA_DISABLE_ERROR_CHECKING
#define CHECK_CUDA(ans) check_cuda((ans), "", #ans, __FILE__, __LINE__)
#define CHECK_LAST(msg) check_cuda(cudaGetLastError(), msg, "CHECK_LAST", __FILE__, __LINE__)
#else
#define CHECK_CUDA(ans) {}
#define CHECK_LAST(msg) {}
#endif


inline
void throw_error(int code,
                 const char* error_string,
                 const char* msg,
                 const char* func,
                 const char* file,
                 int line) {
  throw std::runtime_error("CUDA error "
                           +std::string(msg)
                           +" "+std::string(error_string)
                           +" ["+std::to_string(code)+"]"
                           +" "+std::string(file)
                           +":"+std::to_string(line)
                           +" "+std::string(func)
    );
}

inline
void check_cuda(cudaError_t code, const char* msg, const char *func, const char *file, int line) {
  if (code != cudaSuccess) {
    throw_error(static_cast<int>(code),
                cudaGetErrorString(code), msg, func, file, line);
  }
}

#ifdef CURAND_H_
// cuRAND API errors
static const char *getCuRandErrorString(curandStatus_t error)
{
  switch (error)
  {
  case CURAND_STATUS_SUCCESS:
    return "CURAND_STATUS_SUCCESS";

  case CURAND_STATUS_VERSION_MISMATCH:
    return "CURAND_STATUS_VERSION_MISMATCH";

  case CURAND_STATUS_NOT_INITIALIZED:
    return "CURAND_STATUS_NOT_INITIALIZED";

  case CURAND_STATUS_ALLOCATION_FAILED:
    return "CURAND_STATUS_ALLOCATION_FAILED";

  case CURAND_STATUS_TYPE_ERROR:
    return "CURAND_STATUS_TYPE_ERROR";

  case CURAND_STATUS_OUT_OF_RANGE:
    return "CURAND_STATUS_OUT_OF_RANGE";

  case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
    return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

  case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
    return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

  case CURAND_STATUS_LAUNCH_FAILURE:
    return "CURAND_STATUS_LAUNCH_FAILURE";

  case CURAND_STATUS_PREEXISTING_FAILURE:
    return "CURAND_STATUS_PREEXISTING_FAILURE";

  case CURAND_STATUS_INITIALIZATION_FAILED:
    return "CURAND_STATUS_INITIALIZATION_FAILED";

  case CURAND_STATUS_ARCH_MISMATCH:
    return "CURAND_STATUS_ARCH_MISMATCH";

  case CURAND_STATUS_INTERNAL_ERROR:
    return "CURAND_STATUS_INTERNAL_ERROR";
  }

  return "<unknown>";
}

inline
void check_cuda(curandStatus_t code, const char* msg, const char *func, const char *file, int line) {
  if (code != CURAND_STATUS_SUCCESS) {
    throw_error(static_cast<int>(code),
                getCuRandErrorString(code),
                msg, func, file, line);
  }
}

#endif

inline
std::stringstream getCUDADeviceInformations(int dev) {
  std::stringstream info;
  cudaDeviceProp prop;
  int runtimeVersion = 0;
  size_t f=0, t=0;
  CHECK_CUDA( cudaRuntimeGetVersion(&runtimeVersion) );
  CHECK_CUDA( cudaGetDeviceProperties(&prop, dev) );
  CHECK_CUDA( cudaMemGetInfo(&f, &t) );
  info << '"' << prop.name << '"'
       << ", \"CC\", " << prop.major << '.' << prop.minor
       << ", \"Multiprocessors\", "<< prop.multiProcessorCount
       << ", \"Memory [MiB]\", "<< t/1048576
       << ", \"MemoryFree [MiB]\", " << f/1048576
       << ", \"MemClock [MHz]\", " << prop.memoryClockRate/1000
       << ", \"GPUClock [MHz]\", " << prop.clockRate/1000
       << ", \"CUDA Runtime\", " << runtimeVersion
    ;
  return info;
}

inline
std::stringstream listCudaDevices() {
  std::stringstream info;
  int nrdev = 0;
  CHECK_CUDA( cudaGetDeviceCount( &nrdev ) );
  if(nrdev==0)
    throw std::runtime_error("No CUDA capable device found");
  for(int i=0; i<nrdev; ++i)
    info << "\"ID\"," << i << "," << getCUDADeviceInformations(i).str() << std::endl;
  return info;
}

#endif
