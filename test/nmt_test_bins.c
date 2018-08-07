#include "namaster.h"
#include "ctest.h"
#include "utils.h"
#include "nmt_test_utils.h"

CTEST_SKIP(nmt,bins_constant) {
  nmt_binning_scheme *bin=nmt_bins_constant(4,2000);
  ASSERT_EQUAL(bin->n_bands,499);
  ASSERT_EQUAL(bin->ell_list[5][2],2+4*5+2);
  nmt_bins_free(bin);
}

CTEST_SKIP(nmt,bins_variable) {
  int i,j;
  nmt_binning_scheme *bin=nmt_bins_constant(4,2000);
  int *ells=test_get_sequence(2,1998);
  int *bpws=malloc(1996*sizeof(int));
  double *weights=malloc(1996*sizeof(double));
  for(i=0;i<1996;i++) {
    bpws[i]=i/4;
    weights[i]=0.25;
  }
  nmt_binning_scheme *bin2=nmt_bins_create(1996,bpws,ells,weights,2000);

  ASSERT_EQUAL(bin->n_bands,499);
  ASSERT_EQUAL(bin->n_bands,bin2->n_bands);
  ASSERT_EQUAL(bin->ell_list[5][2],2+4*5+2);
  ASSERT_EQUAL(bin2->ell_list[5][2],2+4*5+2);
  free(bpws);
  free(ells);
  free(weights);
  nmt_bins_free(bin);
  nmt_bins_free(bin2);
}
