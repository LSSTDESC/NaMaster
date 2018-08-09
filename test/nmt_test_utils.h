#ifndef _NMT_TEST_UTILS_
#define _NMT_TEST_UTILS_

#define NSIDE_TESTS 128
#define NNO_RNG 10000

#include "utils.h"

double **test_make_map_analytic_flat(nmt_flatsky_info *fsk,int pol,int i0_x,int i0_y);

double **test_make_map_analytic(long nside,int pol);

int *test_get_sequence(int n0,int nf);

void test_compare_arrays(int n,double *y,int narr,int iarr,char *fname,double rtol);

#endif //_NMT_TEST_UTILS_
