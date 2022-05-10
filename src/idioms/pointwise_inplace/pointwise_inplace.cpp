#include "pointwise_inplace.h"
#include <benchmark/benchmark.h>
#include <cstdlib>
#include <string>

static void kernel(int n, double *A) {
    for (int i = 0; i < n; i += 1) {
        A[i] += 42;
    }
}


static void pointwise_inplace_serial(benchmark::State& state, int n) {
    double *A = new double[n];

    for (auto &&_ : state) {
        kernel(n, A);
        benchmark::ClobberMemory();
    }

    delete[] A;
}





int main(int argc, char* argv[]) {
    ::benchmark::Initialize(&argc, argv);

    int n = N;
    if (argc > 1) {
       n = std::atoi(argv[1]);
       argc -= 1;
       argv += 1;
    }

    benchmark::RegisterBenchmark(("pointwise_inplace_serial" + std::string("/") +std:: to_string(n)).c_str(), pointwise_inplace_serial, n)->Unit(benchmark::kMillisecond);

    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return EXIT_SUCCESS;
}
