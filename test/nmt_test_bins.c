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
  for(i=0;i<499;i++) {
    for(j=0;j<4;j++) {
      ASSERT_EQUAL(bin->ell_list[i][j],2+4*i+j);
      ASSERT_EQUAL(bin2->ell_list[i][j],2+4*i+j);
    }
  }

  set_error_policy(THROW_ON_ERROR);

  printf("\nError messages expected: \n");
  nmt_bins_free(bin2); bin2=NULL;
  for(i=0;i<4;i++)
    weights[16+i]=0;
  try { bin2=nmt_bins_create(1996,bpws,ells,weights,2000); }
  catch(1) {}
  ASSERT_EQUAL(1,exception_status);
  ASSERT_NULL(bin2);
  set_error_policy(EXIT_ON_ERROR);

  free(bpws);
  free(ells);
  free(weights);
  nmt_bins_free(bin);
}

CTEST_SKIP(nmt,bins_read) {
  set_error_policy(THROW_ON_ERROR);
  printf("\nError messages expected: \n");

  nmt_binning_scheme *b=NULL;
  try { b=nmt_bins_read("test/cls.txt",2000); }
  catch(1) {}
  ASSERT_EQUAL(1,exception_status);
  ASSERT_NULL(b);
  set_error_policy(EXIT_ON_ERROR);

  b=nmt_bins_read("test/bins.txt",100);
  nmt_bins_free(b);
}

CTEST_SKIP(nmt,bins_binning) {
  int i,nside=256;
  int lmax=3*nside-1;
  double *cls=my_malloc((lmax+1)*sizeof(double));
  double *cls_back=my_malloc((lmax+1)*sizeof(double));
  nmt_binning_scheme *bin=nmt_bins_constant(4,lmax);
  double *cls_binned=my_malloc(bin->n_bands*sizeof(double));
  double *leff=my_malloc(bin->n_bands*sizeof(double));

  for(i=0;i<=lmax;i++)
    cls[i]=i+0.;

  nmt_bin_cls(bin,&cls,&cls_binned,1);
  nmt_unbin_cls(bin,&cls_binned,&cls_back,1);
  nmt_ell_eff(bin,leff);

  for(i=0;i<bin->n_bands;i++) {
    int j;
    ASSERT_DBL_NEAR_TOL(2+4*i+1.5,cls_binned[i],1E-10);
    ASSERT_DBL_NEAR_TOL(leff[i],cls_binned[i],1E-10);
    for(j=0;j<4;j++)
      ASSERT_DBL_NEAR_TOL(2+4*i+1.5,cls_back[2+4*i+j],1E-10);
  }

  nmt_bins_free(bin);
  free(leff);
  free(cls_binned);
  free(cls);
  free(cls_back);
}
