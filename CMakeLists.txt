project(main)
cmake_minimum_required(VERSION 3.4.1)

set(CMAKE_CXX_STANDARD 20)
add_compile_options(-Wall -Wextra -Wpedantic -Werror)

find_package(PkgConfig)
pkg_check_modules(benchmark REQUIRED)
find_package(OpenMP REQUIRED)

set(KOMPUTE_OPT_LOG_LEVEL Error)
#KOMPUTE_OPT_BUILD_SHADER=ON
add_subdirectory(subprojects/kompute)

add_executable(main main.cc)
target_link_libraries(main benchmark kompute OpenMP::OpenMP_CXX)
