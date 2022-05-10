#include "pointwise.h"
#include <benchmark/benchmark.h>
#include <cstdlib>
#include <string>

static void kernel(int n, double *B, double *A) {
    #pragma omp target teams distribute parallel for map(from:B) map(to:A) 
    for (int i = 0; i < n; i += 1) {
        B[i] = 42 * A[i];
    }
    #pragma omp taskwait
}


static void pointwise_seq(benchmark::State& state, int n) {
    double *A = new double[n];
    double *B = new double[n];

    for (auto &&_ : state) {
        kernel(n, B, A);
        benchmark::ClobberMemory();
    }

    delete[] A;
    delete[] B;
}


int main(int argc, char* argv[]) {
    ::benchmark::Initialize(&argc, argv);

    int n = N;
    if (argc > 1) {
       n = std::atoi(argv[1]);
       argc -= 1;
       argv += 1;
    }


    benchmark::RegisterBenchmark(("pointwise.omp_target" + std::string("/") +std:: to_string(n)).c_str(), pointwise_seq, n)->Unit(benchmark::kMillisecond);


    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return EXIT_SUCCESS;
}

