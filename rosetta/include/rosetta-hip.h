#ifndef ROSETTA_HIP_H_
#define ROSETTA_HIP_H_

#define BENCH_HIP_TRY(call)                                 \
  do {                                                      \
    auto const status = (call);                             \
    if (hipSuccess != status) {                             \
      printf("HIP call '" #call "' returned %d\n", status); \
      abort();                                              \
    }                                                       \
  } while (0);

#endif
