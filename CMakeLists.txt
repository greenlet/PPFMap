cmake_minimum_required (VERSION 2.8.12 FATAL_ERROR)

project (PPFMap CXX)
message(STATUS "CMAKE_VERSION: ${CMAKE_VERSION}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")


##############################################################################
#   Check for C++11 compiler support                                         #
##############################################################################

# include(CheckCXXCompilerFlag)
# check_cxx_compiler_flag(-std=c++11 SUPPORTS_STD_CXX11)
# check_cxx_compiler_flag(-std=c++0x SUPPORTS_STD_CXX01)
# if(SUPPORTS_STD_CXX11)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#     set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
#     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c++11")
# elseif(SUPPORTS_STD_CXX01)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
#     set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++0x")
#     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c++0x")
# else()
#      message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
# endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c++14")


##############################################################################
#   Dependencies                                                             #
##############################################################################

set (CUDA_ARCH "sm_35" CACHE STRING "Cuda Architechture") 

set (DEMO_FILES src/demo.cpp)
set (DEMO_DKP_FILES src/demo_dkp.cpp)
set (PCL_VIS_FILES src/pcl_visualizer_demo.cpp)

set (CPP_FILES "")

set (HPP_FILES
    "include/PPFMap/common.h"
    "include/PPFMap/cuda_map.h"
    "include/PPFMap/CudaPPFMatch.h"
    )


set (CU_SRC_FILES
    src/Map.cu
    src/ppf_cuda_calls.cu)
    
list(APPEND LIBRARIES "uuid")

# CUDA
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-10.1")
set(CUDA_cublas_LIBRARY "/usr/lib/x86_64-linux-gnu/libcublas.so")
FIND_PACKAGE(CUDA REQUIRED)
message(STATUS "CUDA Version: " ${CUDA_VERSION})
message(STATUS "CUDA Libararies: " ${CUDA_LIBRARIES})
message(STATUS "CUDA Cublas Libararies: " ${CUDA_CUBLAS_LIBRARIES})
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
list(APPEND LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY})
list(APPEND CUDA_NVCC_FLAGS "-arch=${CUDA_ARCH};-O2;-Xcompiler;-fPIC;-DVERBOSE")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
add_definitions(-DGPU)
cuda_include_directories (${CMAKE_SOURCE_DIR}/include)


set(CUDA_ARCH_BIN "50 52" CACHE STRING "Specify 'real' GPU arch to build binaries for, BIN(PTX) format is supported. Example: 1.3 2.1(1.3) or 13 21(13)")
set(CUDA_ARCH_PTX "" CACHE STRING "Specify 'virtual' PTX arch to build PTX intermediate code for. Example: 1.0 1.2 or 10 12")              

include(CudaComputeTargetFlags)
APPEND_TARGET_ARCH_FLAGS()


include_directories(${CMAKE_SOURCE_DIR}/include)

CUDA_COMPILE(cuda_objs ${CU_SRC_FILES})


# Point Cloud Library (PCL)
#set(Boost_DEBUG ON)
# set(Boost_USE_STATIC_LIBS ON)
# set(Boost_USE_STATIC ON)
set(Qt5Network_DIR "/usr/lib/x86_64-linux-gnu/cmake/Qt5Network")
set(Qt5WebKit_DIR "/usr/lib/x86_64-linux-gnu/cmake/Qt5WebKit")
set(PCL_DIR "/usr/lib/x86_64-linux-gnu/cmake/pcl")
# set(BOOST_ROOT "/usr/lib/x86_64-linux-gnu")
set(Boost_NO_SYSTEM_PATHS ON)
set(BOOST_INCLUDEDIR "/usr/include")
set(BOOST_LIBRARYDIR "/usr/lib/x86_64-linux-gnu")
# find_package(Boost 1.65.0 REQUIRED COMPONENTS system filesystem thread date_time iostreams serialization chrono)
find_package (PCL 1.8 REQUIRED)
message(STATUS "---- PCL_INCLUDE_DIRS: ${PCL_INCLUDE_DIRS}")
include_directories (${PCL_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
find_package(Poco REQUIRED COMPONENTS JSON)
find_package(Eigen3 REQUIRED)

# get_cmake_property(var_names VARIABLES)
# list (SORT var_names)
# set(vars_str "")
# foreach (var_name ${var_names})
#     set(vars_str "${vars_str}\n${var_name} = ${${var_name}}")
# endforeach()
# file(WRITE "cmake_variables.txt" ${vars_str})

cuda_add_library(ppfmap SHARED ${CPP_FILES} ${HPP_FILES} ${cuda_objs})
target_precompile_headers(ppfmap PUBLIC "include/PPFMap/common.h")

cuda_add_executable(demo ${DEMO_FILES})
target_link_libraries (demo ppfmap ${PCL_LIBRARIES} ${LIBRARIES})
target_precompile_headers(demo PUBLIC "src/common.h")

cuda_add_executable(demo_dkp ${DEMO_DKP_FILES})
target_link_libraries (demo_dkp ppfmap ${PCL_LIBRARIES} ${LIBRARIES} ${OpenCV_LIBS} ${Poco_LIBRARIES})
target_precompile_headers(demo_dkp PUBLIC "src/common.h")

cuda_add_executable(pcl_visualizer_demo ${PCL_VIS_FILES})
target_link_libraries (pcl_visualizer_demo ${PCL_LIBRARIES} ${LIBRARIES})
target_precompile_headers(pcl_visualizer_demo PUBLIC "src/common.h")

##############################################################################
#   Installation rules                                                       #
##############################################################################

install (TARGETS ppfmap DESTINATION lib)
install (DIRECTORY include/PPFMap DESTINATION include)

##############################################################################
#   Doxygen documentation                                                    #
##############################################################################

# adds custom command 'make doc' to generate the documentation.
find_package(Doxygen)
if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in
        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(doc
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif(DOXYGEN_FOUND)

