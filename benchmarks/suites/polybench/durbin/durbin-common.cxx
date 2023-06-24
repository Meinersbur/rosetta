#include "durbin-common.h"

void initialize_input_vector(int n, real r[]) {
  for (int i = 0; i < n; i++)
  {
     r[i] = n+1-i;
  }
}
