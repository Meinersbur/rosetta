# Adapted from Polybench/C 4.2.1 beta
# https://web.cse.ohio-state.edu/~pouchet.2/software/polybench/
# Only LARGE_DATASET (default) has the same dataset size as --pbsize=large

# datamining
add_subdirectory(correlation)
add_subdirectory(covariance)

# linear-algebra/blas
add_subdirectory(gemm)
add_subdirectory(gemver)
add_subdirectory(gesummv)
add_subdirectory(symm)
add_subdirectory(syr2k)
add_subdirectory(syr)
add_subdirectory(trmm)

# linear-algebra/kernels
add_subdirectory(2mm)
add_subdirectory(3mm)
add_subdirectory(atax)
add_subdirectory(bicg)
add_subdirectory(doitgen)
add_subdirectory(mvt)

# linear-algebra/solvers
add_subdirectory(cholesky)
add_subdirectory(durbin)
add_subdirectory(gramschmidt)
add_subdirectory(lu)
add_subdirectory(ludcmp)
add_subdirectory(trisolv)

# medley
add_subdirectory(deriche)
add_subdirectory(floyd-warshall)
add_subdirectory(nussinov)

# stencils
add_subdirectory(adi)
add_subdirectory(fdtd-2d)
add_subdirectory(heat-3d)
add_subdirectory(jacobi-1d)
add_subdirectory(jacobi-2d)
add_subdirectory(seidel-2d)
