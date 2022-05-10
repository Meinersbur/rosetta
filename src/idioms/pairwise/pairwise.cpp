#include "pairwise.h"
#include <benchmark/benchmark.h>
#include <cstdlib>
#include <string>

static void kernel(int n, double *A, double *B, double *C) {
    for (int i = 0; i < n; i += 1) {
        C[i] = B[i] + A[i];
    }
}


static void pairwise_serial(benchmark::State& state, int n) {
    double *A = new double[n];
    double *B = new double[n];
    double *C = new double[n];

    for (auto &&_ : state) {
        kernel(n, B, A, C);
        benchmark::ClobberMemory();
    }

    delete[] A;
    delete[] B;
    delete[] C;
}


int main(int argc, char* argv[]) {
    ::benchmark::Initialize(&argc, argv);

    int n = N;
    if (argc > 1) {
       n = std::atoi(argv[1]);
       argc -= 1;
       argv += 1;
    }

    benchmark::RegisterBenchmark(("pairwise.serial" + std::string("/") +std:: to_string(n)).c_str(), pairwise_serial, n)->Unit(benchmark::kMillisecond);

    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return EXIT_SUCCESS;
}
