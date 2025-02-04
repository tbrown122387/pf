#include <stdio.h>

#include "forward_sim.h"
#include "hilb_sort_pf.h"
#include "resamp_comparison.h"
#include "svol_comparison.h"

int main(int argc, char **argv) {

  if (argc != 3) {
    printf(
        "please enter: \n"
        "1. 1, 2,  3 or 4 (svol comparison, forward sim example, resamp "
        "comparison, pf wth hilb. sort)\n"
        "2. the path to the data file (ignored if you previously chose 2)\n");

  } else { // entered three inputs

    if (std::atoi(argv[1]) == 1) {
      run_svol_comparison(argv[2]);
    } else if (std::atoi(argv[1]) == 2) {
      forward_sim();
    } else if (std::atoi(argv[1]) == 3) {
      run_resamp_comparison(argv[2]);
    } else if (std::atoi(argv[1]) == 4) {
      run_hilb_pf_example(argv[2]);
    } else {
      printf("invalid input\n");
    }
  }

  return 0;
}
