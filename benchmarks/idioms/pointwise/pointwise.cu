#include "pointwise.h"
#include "rosetta.h"
//#include <benchmark/benchmark.h>
//#include <cstdlib>
//#include <string>
//#include "synchronization.hpp" 

// Loosely based on CUDA Toolkit sample: vectorAdd

__global__ void kernel(int n, double *B, double *A) {
     int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        B[i] = 42* A[i];
}


static void pointwise_cuda(benchmark::State& state, int n) {
    double *A = new double[n];
    double *B = new double[n];

    double *dev_A, *dev_B;
    BENCH_CUDA_TRY(cudaMalloc((void**)&dev_A, n * sizeof(double)));
    BENCH_CUDA_TRY(cudaMalloc((void**)&dev_B, n * sizeof(double)));

    cudaMemcpy(dev_A, A, N * sizeof(double), cudaMemcpyHostToDevice);

// TODO: dim3 dimBlock(16, 16, 1);
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

//state.PauseTiming();
    for (auto &&_ : state) {
state.PauseTiming();
       cudaStream_t stream = 0;
    //    cuda_event_timer raii(state, true, stream); 
state.ResumeTiming();
        kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(n, dev_B, dev_A);


<<<<<<< HEAD:src/idioms/pointwise/pointwise.cu
        //cudaMemcpy(B, dev_B, n * sizeof(double), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }
||||||| 17c7cd3:src/idioms/pointwise/pointwise.cu
        // TODO: Is the invocation already blocking?
         cudaMemcpy( B, dev_B, n * sizeof(double), cudaMemcpyDeviceToHost );
    }
=======
        // TODO: Is the invocation already blocking?
         cudaMemcpy( B, dev_B, n * sizeof(double), cudaMemcpyDeviceToHost );
         benchmark::ClobberMemory(); 
>>>>>>> meinersbur/master:benchmarks/idioms/pointwise/pointwise.cu

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

    benchmark::RegisterBenchmark(("pointwise.cuda" + std::string("/") +std:: to_string(n) + "/gpu").c_str(), &pointwise_cuda, n)->MeasureProcessCPUTime()->UseRealTime()->Unit(benchmark::kMillisecond)->UseManualTime();
    benchmark::RegisterBenchmark(("pointwise.cuda" + std::string("/") +std:: to_string(n) + "/cpu").c_str() , &pointwise_cuda, n)->MeasureProcessCPUTime()->UseRealTime()->Unit(benchmark::kMillisecond);

    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return EXIT_SUCCESS;
}
