# FLOPS & Latency Benchmarks

This repo tries to answer the question "for a specific input size, and (simple) computation, which framework/accelerator should I choose?".

By implementing the same computation on all available frameworks/accelerators (CPU FPU baseline, SIMD, threads, GPU, FPGA, multi-machine, ...) and running with input sizes from 1 float to 1 billion (or more) floats, we'll see which framework is optimal for which input size.

Frameworks:
- [ ] single-threaded scalar/FPU-only
- SIMD
  - [x] SIMD compiler auto-vectorisation
  - [ ] SIMD explicit ([std::experimental::simd](https://en.cppreference.com/w/cpp/experimental/simd/simd)) (equivalent to [gcc/clang vector intrinsics](https://gcc.gnu.org/onlinedocs/gcc/Vector-Extensions.html))
  - [ ] [`-fopenmp-simd` OpenMP SIMD](https://github.com/simd-everywhere/simde#openmp-4-simd)
- threading
  - [x] `-fopenmp` OpenMP threading
  - [x] [C++11 threads](https://en.cppreference.com/w/cpp/thread/thread)
  - [ ] [`std::execution::par_unseq`](https://en.cppreference.com/w/cpp/algorithm/execution_policy_tag) [clang parallel stl (pstl)](https://libcxx.llvm.org/Status/PSTL.html)
- GPU
  - [ ] [`-fopenmp -fopenmp-targets=nvptx64` OpenMP GPU offloading](https://enccs.github.io/openmp-gpu/target/)
  - [ ] [CUDA/HIP](https://llvm.org/docs/CompileCudaWithLLVM.html), HIP is a subset of CUDA that normal clang can compile to both nvidia and amd GPUs, see https://github.com/ROCm/HIP#what-is-this-repository-for and https://rocm.docs.amd.com/projects/HIP/en/latest/user_guide/faq.html. [automatic CUDA to HIP translation](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/hipify-clang.html)
  - [x] GPU compute shaders via vulkan via [kompute](https://kompute.cc/)
  - [ ] [Halide](https://halide-lang.org/)?
  - [ ] [Kokkos](https://kokkos.github.io/kokkos-core-wiki/#)?
  - [ ] [SYCL](https://en.wikipedia.org/wiki/SYCL#Implementations)
- multi-machine
  - [ ] [OpenMP remote offloading](https://openmp.llvm.org/design/Runtimes.html#remote-offloading-plugin)
  - [ ] manual messaging
  - [ ] MPI

TODO:
- [ ] flops/byte estimation for the two different calculations
- [ ] flops/byte (maximum FLOPS throughput / maximum memory bandwidth) estimation for all the different hardware (CPU, SIMD, threaded, GPU, multi-machine)
- [ ] is it possible to create a flops/byte measurement tool, and a set of stress tests to measure

Similar projects
- https://github.com/ashvardanian/ParallelReductionsBenchmark

# Parallel Programming Ecosystem Comparison

a parallel programming ecosystem needs:
- language/language-extensions/compiler for describing parallelism, tasks, async, dependencies, etc
- backends for SIMD, multi-core, GPU, and multi-machine
- standard algorithms (blas, sort, reduce, etc)
- tools for debugging and profiling

| ecosystem | compiler | SIMD | Multi-core | GPU | Multi-machine | sort | reduce | blas |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C++ STL | any (plain c++) | `std::ex::simd` | `std::thread` | `std::executors` (future) | :grey_question: asio? (future) | `std::sort` `par_unseq` | `std::ex::parallel::reduce` | `stdblas` (future) |
| OpenMP | gcc, clang, icc (pragma extended c++) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | :heavy_check_mark: | :grey_question: OpenBLAS? Eigen? |
| sandia Kokkos | any (plain c++) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: MPI | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: `stdblas` |
| intel oneAPI | intel dpc++ (sycl) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: MPI | :heavy_check_mark: TBB | :heavy_check_mark: TBB | :heavy_check_mark: MKL |
| nvidia CUDA | clang, nvc++ (extended c++) | :x: | :x: | :heavy_check_mark: | :heavy_check_mark: NCCL | :heavy_check_mark: thrust / libcu++ | :heavy_check_mark: thrust / libcu++ | :heavy_check_mark: cutlass / cuBLAS |

## Dependencies

```
sudo pacman -S \
    cmake clang \
    benchmark \
    python-matplotlib python-pandas \
    openmp \
    vulkan-tools vulkan-driver vulkan-headers glslang \
    eigen
```

## Running

```
./test.sh
```

## Example

For O(N) vector addition on specific hardware, this graph answers the question in the following way:
- for input size <= 2^8 floats, use CPU SIMD
- for input size between 2^12 and 2^20 floats, use OpenMP
- for input size >= 2^24 floats (and negligible device/host memory transfer cost) use the GPU

![](output.png)

For O(N^3) matrix multiplication

![](matmul.png)
