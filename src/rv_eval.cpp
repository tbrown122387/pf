#include "rv_eval.h"

#include <cmath> // tgamma, pow, exp, otherstuff
#include <stdexcept> // std::invalid_argument


unsigned int rveval::nChooseK( unsigned int n, unsigned int k )
{
    if (k > n) return 0;
    if (k * 2 > n) /*return*/ k = n-k;  //remove the commented section
    if (k == 0) return 1;

    int result = n;
    for( int i = 2; i <= k; ++i ) {
        result *= (n-i+1);
        result /= i;
    }
    return result;
}


