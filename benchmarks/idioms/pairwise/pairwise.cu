#include "pairwise.h"
#include "rosetta.h"





__global__ void kernel(int n, double *C, double *B, double *A) {
    int ij = blockDim.x * blockIdx.x + threadIdx.x;
    if (ij >= n*n) return;

    int i = ij / n;
    int j = ij % n;
    C[i * n + j]  =  B[i] * A[i];
}


static void pointwise_cuda(benchmark::State& state, int n) {
    double *A = new double[n];
    double *B = new double[n];
    double *C = new double[n*n];

    double *dev_A, *dev_B, *dev_C;
    BENCH_CUDA_TRY(cudaMalloc((void**)&dev_A, n * sizeof(double)));
    BENCH_CUDA_TRY(cudaMalloc((void**)&dev_B, n * sizeof(double)));
    BENCH_CUDA_TRY(cudaMalloc((void**)&dev_C, n * n * sizeof(double)));

    cudaMemcpy(dev_A, A, N * sizeof(double), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = ((n*n) + threadsPerBlock - 1) / threadsPerBlock;

//state.PauseTiming();
    for (auto &&_ : state) {
//state.PauseTiming();
     //  cudaStream_t stream = 0;
   //     cuda_event_timer raii(state, true, stream); 
    //    state.ResumeTiming();
        kernel<<<blocksPerGrid, threadsPerBlock>>>(n,dev_C,  dev_B, dev_A);


        // TODO: Is the invocation already blocking?
         BENCH_CUDA_TRY(cudaMemcpy(C, dev_C, n * n * sizeof(double), cudaMemcpyDeviceToHost));
         benchmark::ClobberMemory(); 
    }
 
    BENCH_CUDA_TRY(cudaFree(dev_A));
    BENCH_CUDA_TRY(cudaFree(dev_B));
    BENCH_CUDA_TRY(cudaFree(dev_C));

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

    benchmark::RegisterBenchmark(("pointwise.cuda" + std::string("/") +std:: to_string(n) + "/gpu").c_str(), &pointwise_cuda, n)->MeasureProcessCPUTime()->UseRealTime()->Unit(benchmark::kMillisecond)->UseManualTime();
    benchmark::RegisterBenchmark(("pointwise.cuda" + std::string("/") +std:: to_string(n) + "/cpu").c_str() , &pointwise_cuda, n)->MeasureProcessCPUTime()->UseRealTime()->Unit(benchmark::kMillisecond);

    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return EXIT_SUCCESS;
}

