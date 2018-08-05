#ifndef _NMT_TEST_UTILS_
#define _NMT_TEST_UTILS_

#define NSIDE_TESTS 128
#define NNO_RNG 10000

int *test_get_sequence(int n0,int nf);

void test_compare_arrays(int n,double *y,int narr,int iarr,char *fname);

#endif //_NMT_TEST_UTILS_
