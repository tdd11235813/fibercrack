cmake_minimum_required(VERSION 2.8)
project(fibercrack)


#------------------------------------------------------------------------------
# Boost
#------------------------------------------------------------------------------

find_package(Boost 1.55 COMPONENTS program_options REQUIRED)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  link_directories(${Boost_LIBRARY_DIRS})
endif()

#------------------------------------------------------------------------------
# CUDA + GCC
#------------------------------------------------------------------------------

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

set(GENCODE_SM20 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20)
set(GENCODE_SM30 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_30,code=compute_30)
set(GENCODE_SM35 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_35,code=compute_35)
set(GENCODE_SM37 -gencode=arch=compute_37,code=sm_37 -gencode=arch=compute_37,code=compute_37)
set(GENCODE_SM50 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_50,code=compute_50)
set(GENCODE_SM60 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60)
set(GENCODE_SM61 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61)

option(CUDA_VERBOSE_PTXAS
  "On to enable verbose output from the PTXAS assembler."
  OFF
  )

option(FIBERCRACK_GENCODE_SM20
  "ON to generate code for Compute Capability 2.0 devices (e.g. GTX 580)"
  ON)

option(FIBERCRACK_GENCODE_SM30
  "ON to generate code for Compute Capability 3.0 devices (e.g. Tesla K10)"
  OFF)

option(FIBERCRACK_GENCODE_SM35
  "ON to generate code for Compute Capability 3.5 devices (e.g. Tesla K20)"
  ON)

option(FIBERCRACK_GENCODE_SM37
  "ON to generate code for Compute Capability 3.7 devices (e.g. Tesla K80)"
  OFF)

option(FIBERCRACK_GENCODE_SM50
  "ON to generate code for Compute Capability 5.0 devices (e.g. GeForce GTX 750)"
  OFF)

option(FIBERCRACK_GENCODE_SM52
  "ON to generate code for Compute Capability 5.2 devices (e.g. GeForce GTX 980)"
  OFF)

option(FIBERCRACK_GENCODE_SM60
  "ON to generate code for Compute Capability 6.0 devices (e.g. P100)"
  OFF)

option(FIBERCRACK_GENCODE_SM61
  "ON to generate code for Compute Capability 6.1 devices (e.g. GTX 1080)"
  OFF)

if (FIBERCRACK_GENCODE_SM20)
  set(GENCODE ${GENCODE} ${GENCODE_SM20})
endif(FIBERCRACK_GENCODE_SM20)

if (FIBERCRACK_GENCODE_SM30)
  set(GENCODE ${GENCODE} ${GENCODE_SM30})
endif(FIBERCRACK_GENCODE_SM30)

if (FIBERCRACK_GENCODE_SM35)
  set(GENCODE ${GENCODE} ${GENCODE_SM35})
endif(FIBERCRACK_GENCODE_SM35)

if (FIBERCRACK_GENCODE_SM37)
  set(GENCODE ${GENCODE} ${GENCODE_SM37})
endif(FIBERCRACK_GENCODE_SM37)

if (FIBERCRACK_GENCODE_SM50)
  set(GENCODE ${GENCODE} ${GENCODE_SM50})
endif(FIBERCRACK_GENCODE_SM50)

if (FIBERCRACK_GENCODE_SM52)
  set(GENCODE ${GENCODE} ${GENCODE_SM52})
endif(FIBERCRACK_GENCODE_SM52)

if (FIBERCRACK_GENCODE_SM60)
  set(GENCODE ${GENCODE} ${GENCODE_SM60})
endif(FIBERCRACK_GENCODE_SM60)

if (FIBERCRACK_GENCODE_SM61)
  set(GENCODE ${GENCODE} ${GENCODE_SM61})
endif(FIBERCRACK_GENCODE_SM61)

if (CUDA_VERBOSE_PTXAS)
  set(VERBOSE_PTXAS --ptxas-options=-v)
endif (CUDA_VERBOSE_PTXAS)




if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND CUDA_NVCC_FLAGS "-g -lineinfo")
endif()

set(CUDA_PROPAGATE_HOST_FLAGS OFF)
cuda_add_executable(${PROJECT_NAME} main.cpp kernel.cu OPTIONS ${GENCODE} ${VERBOSE_PTXAS})

# @todo check if this can be solved via cmake and without to include cuda
target_link_libraries(${PROJECT_NAME}
  ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY} ${Boost_LIBRARIES}
)
