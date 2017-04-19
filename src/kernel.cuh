#ifndef KERNEL_CUH
#define KERNEL_CUH

#include "application.hpp"
#include "cuda_globals.hpp"

template<typename T>
inline void cuda_malloc(T** _ptr, size_t _count) {
  CHECK_CUDA(cudaMalloc(_ptr, _count*sizeof(T)));
}

template<typename T>
inline void cuda_memset(T* _ptr, T _value, size_t _count) {
  CHECK_CUDA(cudaMemset(_ptr, _value, _count*sizeof(T)));
}

template<typename T>
inline void cuda_memcpy(T* _dst, T* _src, size_t _count) {
  CHECK_CUDA( cudaMemcpy(_dst, _src, _count*sizeof(T), cudaMemcpyDefault) );
}



template<typename T>
void cuda_compute_number_of_fibers(const Application<T>& _app,
                                   uint* _output );

template<typename T>
int cuda_create_and_intersect_fibers(const Application<T>& _app,
                                     Data<T>& _data,
                                     int _nr_fibers);

template<typename T>
void cuda_compact_data(Data<T>& _data,
                       int _nr_intersections,
                       int _nr_fibers);


template<typename T>
T cuda_compute_force(Data<T>& _data,
                     const Configuration<T>& _config,
                     int _nr_intersections,
                     int _nr_fibers,
                     T _dstep);




#endif /* KERNEL_CUH */