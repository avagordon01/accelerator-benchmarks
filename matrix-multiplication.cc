#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <curand.h>
#include <cublas_v2.h>

size_t range_start = 1<<4;
size_t range_end = 1<<24;
size_t range_multiplier = 1<<4;

static void eigen_cpu_matmul(benchmark::State &state) {
    size_t s = sqrt(state.range(0));
    auto a = Eigen::MatrixXf::Random(s, s);
    auto b = Eigen::MatrixXf::Random(s, s);
    auto c = Eigen::MatrixXf(s, s);

    omp_set_num_threads(0);
    Eigen::setNbThreads(0);

    for (auto _: state) {
        c = a * b;
    }
}
BENCHMARK(eigen_cpu_matmul)->RangeMultiplier(range_multiplier)->Range(range_start, range_end);

static void eigen_threaded_matmul(benchmark::State &state) {
    size_t s = sqrt(state.range(0));
    auto a = Eigen::MatrixXf::Random(s, s);
    auto b = Eigen::MatrixXf::Random(s, s);
    auto c = Eigen::MatrixXf(s, s);

    omp_set_num_threads(8);
    Eigen::setNbThreads(8);

    for (auto _: state) {
        c = a * b;
    }
}
BENCHMARK(eigen_threaded_matmul)->RangeMultiplier(range_multiplier)->Range(range_start, range_end);

void GPU_fill_rand(float *x, size_t s) {
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
    curandGenerateUniform(prng, x, s * s);
}
static void cublas_matmul(benchmark::State &state) {
    size_t s = sqrt(state.range(0));

    float *a, *b, *c;
    cudaMalloc(&a, s * s * sizeof(float));
    cudaMalloc(&b, s * s * sizeof(float));
    cudaMalloc(&c, s * s * sizeof(float));

    GPU_fill_rand(a, s);
    GPU_fill_rand(b, s);

    cublasHandle_t handle;
    cublasCreate(&handle);

    for (auto _: state) {
        float one = 1.0f;
        cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            s, s, s,
            &one, a, s,
            b, s, &one,
            c, s
        );
    }

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}
BENCHMARK(cublas_matmul)->RangeMultiplier(range_multiplier)->Range(range_start, range_end);

BENCHMARK_MAIN();
