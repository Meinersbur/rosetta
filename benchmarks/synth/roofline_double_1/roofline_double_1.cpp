#include "roofline_double_1.h"
#include <benchmark/benchmark.h>
#include <cstdlib>
#include <string>

static void kernel(int n, double *A) {
    for (int i = 0; i < n; i += 1) {
        A[i] = 42 + A[i];
    }
}


static void roofline_double_1_serial(benchmark::State& state, int n) {
    double *A = new double[n];

    for (auto &&_ : state) {
        kernel(n, A);
        benchmark::ClobberMemory();
    }

    delete[] A;
}



// BENCHMARK(pointwise_seq)->Unit(benchmark::kMicrosecond);


int main(int argc, char* argv[]) {
    ::benchmark::Initialize(&argc, argv);

    int n = N;
    if (argc > 1) {
       n = std::atoi(argv[1]);
       argc -= 1;
       argv += 1;
    }

    benchmark::RegisterBenchmark(("roofline_double_1.serial" + std::string("/") +std:: to_string(n)).c_str(), roofline_double_1_serial, n)->Unit(benchmark::kMillisecond);

    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return EXIT_SUCCESS;
}

