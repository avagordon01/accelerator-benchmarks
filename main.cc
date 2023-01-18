#include <benchmark/benchmark.h>
#include <vector>
#include <random>
#include <thread>
#include <memory>

size_t range_start = 1<<4;
size_t range_end = 1<<28;
size_t range_multiplier = 1<<4;

template<typename C>
static void randomise_container(C container) {
    using T = typename C::value_type;

    std::random_device rd;
    std::mt19937 gen(rd());
    if constexpr (std::numeric_limits<T>::is_integer) {
        std::uniform_int_distribution<T> distrib{};
        for (auto &x: container) {
            x = distrib(gen);
        }
    } else {
        std::uniform_real_distribution<T> distrib{};
        for (auto &x: container) {
            x = distrib(gen);
        }
    }
}

template<typename T>
static void no_simd_addition(benchmark::State &state) {
    auto a = std::vector<T>(state.range(0));
    auto b = std::vector<T>(state.range(0));
    auto c = std::vector<T>(state.range(0));
    randomise_container(a);
    randomise_container(b);
    for (auto _: state) {
#pragma clang loop vectorize(disable)
#pragma clang loop vectorize_predicate(disable)
#pragma clang loop interleave(disable)
#pragma clang loop unroll(disable)
        for (int i = 0; i < state.range(0); i++) {
            c[i] = a[i] + b[i];
        }
    }
}
BENCHMARK(no_simd_addition<float>)->RangeMultiplier(range_multiplier)->Range(range_start, range_end);

template<typename T>
static void simd_addition(benchmark::State &state) {
    auto a = std::vector<T>(state.range(0));
    auto b = std::vector<T>(state.range(0));
    auto c = std::vector<T>(state.range(0));
    randomise_container(a);
    randomise_container(b);
    for (auto _: state) {
#pragma clang loop vectorize(enable)
#pragma clang loop vectorize_predicate(enable)
#pragma clang loop interleave(enable)
#pragma clang loop unroll(enable)
        for (int i = 0; i < state.range(0); i++) {
            c[i] = a[i] + b[i];
        }
    }
}
BENCHMARK(simd_addition<float>)->RangeMultiplier(range_multiplier)->Range(range_start, range_end);

template<typename T>
static void openmp_addition(benchmark::State &state) {
    auto a = std::vector<T>(state.range(0));
    auto b = std::vector<T>(state.range(0));
    auto c = std::vector<T>(state.range(0));
    randomise_container(a);
    randomise_container(b);
    for (auto _: state) {
#pragma omp parallel for
        for (int i = 0; i < state.range(0); i++) {
            c[i] = a[i] + b[i];
        }
    }
}
BENCHMARK(openmp_addition<float>)->RangeMultiplier(range_multiplier)->Range(range_start, range_end);

template<typename T>
static void openmp_simd_addition(benchmark::State &state) {
    auto a = std::vector<T>(state.range(0));
    auto b = std::vector<T>(state.range(0));
    auto c = std::vector<T>(state.range(0));
    randomise_container(a);
    randomise_container(b);
    for (auto _: state) {
#pragma omp parallel for simd
        for (int i = 0; i < state.range(0); i++) {
            c[i] = a[i] + b[i];
        }
    }
}
BENCHMARK(openmp_simd_addition<float>)->RangeMultiplier(range_multiplier)->Range(range_start, range_end);

template<typename T>
static void threaded_addition(benchmark::State &state) {
    auto a = std::vector<T>(state.range(0));
    auto b = std::vector<T>(state.range(0));
    auto c = std::vector<T>(state.range(0));
    randomise_container(a);
    randomise_container(b);

    auto inner = [&](size_t begin, size_t end){
        for (size_t i = begin; i < end; i++) {
            c[i] = a[i] + b[i];
        }
    };

    for (auto _: state) {
        //XXX this is really slow because it doesn't pre-spawn threads in a pool
        std::vector<std::thread> threads;
        size_t n = std::thread::hardware_concurrency();
        for (size_t t = 0; t < n; t++) {
            threads.emplace_back(
                inner,
                t * (state.range(0) / n),
                (t + 1) * (state.range(0) / n)
            );
        }
        for (auto& thread: threads) {
            thread.join();
        }
    }
}
BENCHMARK(threaded_addition<float>)->RangeMultiplier(range_multiplier)->Range(range_start, range_end);

#include <fstream>
static std::vector<uint32_t> compileSource(const std::string& source) {
    std::ofstream fileOut("tmp_kp_shader.comp");
    fileOut << source;
    fileOut.close();
    if (system(std::string("glslangValidator -V tmp_kp_shader.comp -o tmp_kp_shader.comp.spv").c_str()))
        throw std::runtime_error("Error running glslangValidator command");
    std::ifstream fileStream("tmp_kp_shader.comp.spv", std::ios::binary);
    std::vector<char> buffer;
    buffer.insert(buffer.begin(), std::istreambuf_iterator<char>(fileStream), {});
    return {(uint32_t*)buffer.data(), (uint32_t*)(buffer.data() + buffer.size())};
}

#include <kompute/Kompute.hpp>
template<typename T>
static void kompute_addition(benchmark::State &state) {
    auto a = std::vector<T>(state.range(0));
    auto b = std::vector<T>(state.range(0));
    auto c = std::vector<T>(state.range(0));
    randomise_container(a);
    randomise_container(b);

    std::string shader = (R"(
        #version 450

        layout(
            local_size_x = 1024,
            local_size_y = 1,
            local_size_z = 1
        ) in;

        layout(set = 0, binding = 0) buffer buf_a { float a[]; };
        layout(set = 0, binding = 1) buffer buf_b { float b[]; };
        layout(set = 0, binding = 2) buffer buf_c { float c[]; };

        void main() {
            uint index = gl_GlobalInvocationID.x;
            c[index] = a[index] + b[index];
        }
    )");

    kp::Manager mgr; 

    std::vector<std::shared_ptr<kp::Tensor>> params = {
        mgr.tensorT<float>(a),
        mgr.tensorT<float>(b),
        mgr.tensorT<float>(c)
    };
    auto algorithm = mgr.algorithm(
        params,
        compileSource(shader),
        kp::Workgroup{static_cast<unsigned>(state.range(0)) / 1024, 1, 1}
    );

    for (auto _: state) {
        mgr.sequence()
            ->record<kp::OpTensorSyncDevice>(params)
            ->record<kp::OpAlgoDispatch>(algorithm)
            ->eval<kp::OpTensorSyncLocal>(params);
    }
}
BENCHMARK(kompute_addition<float>)->RangeMultiplier(range_multiplier)->Range(range_start, range_end);

BENCHMARK_MAIN();
