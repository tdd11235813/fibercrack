cmake_minimum_required(VERSION 2.8)
project(fibercrack)

find_package(CUDA REQUIRED)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  if(CUDA_VERSION_STRING VERSION_EQUAL 8.0 OR CUDA_VERSION_STRING VERSION_GREATER 8.0)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 6.0
        OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 6.0)
      message(FATAL_ERROR "CUDA 8.0 does not support gcc6+. Please choose an older gcc version like:\n
cmake -DCMAKE_C_COMPILER=`which gcc-5` -DCMAKE_CXX_COMPILER=`which g++-5` ..")
    endif()
  elseif(CUDA_VERSION_STRING VERSION_EQUAL 7.5 OR CUDA_VERSION_STRING VERSION_GREATER 7.5)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0
        OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0)
      message(FATAL_ERROR "CUDA 7.5 does not support gcc5+. Please choose an older gcc version like:\n
cmake -DCMAKE_C_COMPILER=`which gcc-4` -DCMAKE_CXX_COMPILER=`which g++-4` ..")
    endif()
  endif()
endif()

include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)

set(CMAKE_CXX_STANDARD 14)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  list(APPEND CMAKE_CXX_FLAGS "-std=gnu++14")
endif ()


include(FindCUDA)
include_directories(${CUDA_INCLUDE_DIRS})

list(APPEND CUDA_NVCC_FLAGS "-std=c++11") #--ftz=false") # || --use_fast_math
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_20,code=compute_20")
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_30,code=compute_30")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)


if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND CUDA_NVCC_FLAGS "-g -lineinfo")
endif()

cuda_add_executable(${PROJECT_NAME} main.cu)
target_link_libraries(${PROJECT_NAME}
  ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY}
)
