project(benchmarks)
cmake_minimum_required(VERSION 3.4.1)

set(CMAKE_CXX_STANDARD 20)
add_compile_options(-Wall -Wextra -Wpedantic -Werror)

find_package(PkgConfig)
pkg_check_modules(benchmark REQUIRED)
find_package(OpenMP REQUIRED)

set(KOMPUTE_OPT_LOG_LEVEL Error)
#KOMPUTE_OPT_BUILD_SHADER=ON
add_subdirectory(subprojects/kompute)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

find_package(CUDAToolkit)

add_executable(vector-addition vector-addition.cc)
target_link_libraries(vector-addition benchmark kompute OpenMP::OpenMP_CXX)

add_executable(matrix-multiplication matrix-multiplication.cc)
target_link_libraries(matrix-multiplication benchmark Eigen3::Eigen OpenMP::OpenMP_CXX CUDA::cudart CUDA::cublas CUDA::curand)
