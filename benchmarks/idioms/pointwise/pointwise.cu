//#include "pointwise.h"
#include "rosetta.h"
//#include <benchmark/benchmark.h>
//#include <cstdlib>
//#include <string>
//#include "synchronization.hpp" 

// Loosely based on CUDA Toolkit sample: vectorAdd

__global__ void kernel(int n, double *A) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        A[i] += 42;
}


 void run(State& state, int n) {
    double *A = state.malloc<double>(n);   


    double *dev_A;
    BENCH_CUDA_TRY(cudaMalloc((void**)&dev_A, n * sizeof(double)));


    cudaMemcpy(dev_A, A, n * sizeof(double), cudaMemcpyHostToDevice);



// TODO: dim3 dimBlock(16, 16, 1);
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

//state.PauseTiming();
    for (auto &&_ : state.manual()) {
//state.PauseTiming();
       cudaStream_t stream = 0;
    //    cuda_event_timer raii(state, true, stream); 
//state.ResumeTiming();

{
        auto &&scope = _.scope();
        kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(n, dev_A);
        }

        // TODO: Is the invocation already blocking?
        cudaMemcpy( A, dev_A, n * sizeof(double), cudaMemcpyDeviceToHost ); cudaDeviceSynchronize();

    }

    cudaFree(dev_A);


    state.verifydata(A, n);
    state.free(A);
}

