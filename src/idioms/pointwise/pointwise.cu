#include "pointwise.h"
#include <benchmark/benchmark.h>
#include <cstdlib>
#include <string>

// Loosely based on CUDA Toolkit sample: vectorAdd

__device__  void kernel(int n, double *B, double *A) {
     int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        B[i] = 42* A[i];
}


static void pointwise_cuda(benchmark::State& state, int n) {
    double *A = new double[n];
    double *B = new double[n];

    double *dev_A, *dev_B;
    cudaMalloc((void**)&dev_A, n * sizeof(double));
    cudaMalloc((void**)&dev_B, n * sizeof(double));

    cudaMemcpy(dev_A, A, N * sizeof(double), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid =(n + threadsPerBlock - 1) / threadsPerBlock;

    for (auto &&_ : state) {
        kernel<<blocksPerGrid, threadsPerBlock>>(n, dev_B, dev_A);
        
        // TODO: Is the invocation already blocking?
         cudaMemcpy( B, dev_B, n * sizeof(double), cudaMemcpyDeviceToHost );
    }

    cudaFree(dev_A);
    cudaFree(dev_B);

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

    benchmark::RegisterBenchmark(("pointwise.cuda" + std::string("/") +std:: to_string(n)).c_str(), pointwise_cuda, n)->Unit(benchmark::kMillisecond);

    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return EXIT_SUCCESS;
}

