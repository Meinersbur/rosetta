#ifndef ROSETTA_CUDA_H_
#define ROSETTA_CUDA_H_

#define BENCH_CUDA_TRY(call)                                 \
  do {                                                       \
    auto const status = (call);                              \
    if (cudaSuccess != status) {                             \
      printf("CUDA call '" #call "' returned %d\n", status); \
      abort();                                               \
    }                                                        \
  } while (0);

#endif
