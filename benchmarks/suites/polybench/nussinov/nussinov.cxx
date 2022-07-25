#include "rosetta.h"



#define match(b1, b2) (((b1)+(b2)) == 3 ? 1 : 0)
#define max_score(s1, s2) ((s1 >= s2) ? s1 : s2)

static void kernel(int n, real seq[], multarray<real,2> table) {
#pragma scop
 for (int i = n-1; i >= 0; i--) {
  for (int j=i+1; j<n; j++) {

   if (j-1>=0)
      table[i][j] = max_score(table[i][j], table[i][j-1]);
   if (i+1<n )
      table[i][j] = max_score(table[i][j], table[i+1][j]);

   if (j-1>=0 && i+1<n) {
     /* don't allow adjacent elements to bond */
     if (i<j-1)
        table[i][j] = max_score(table[i][j], table[i+1][j-1]+match(seq[i], seq[j]));
     else
        table[i][j] = max_score(table[i][j], table[i+1][j-1]);
   }

   for (int k=i+1; k<j; k++) {
      table[i][j] = max_score(table[i][j], table[i][k] + table[k+1][j]);
   }
  }
 }
#pragma endscop
}



void run(State &state, int pbsize) {
  size_t n = pbsize; // 2500




  auto seq  = state.allocate_array<double>({n}, /*fakedata*/ true , /*verify*/ true);
  auto table  = state.allocate_array<double>({n, n}, /*fakedata*/ true , /*verify*/ false);


  for (auto &&_ : state)
    kernel( n, seq, table );
}
