#include "namaster.h"
#include "ctest.h"
#include "utils.h"
#include "nmt_test_utils.h"

CTEST(nmt,bins_flat_search) {
  int ii;
  int lmax=2000,nlb=5;
  nmt_binning_scheme_flat *bin=nmt_bins_flat_constant(nlb,lmax);
  
  int ib=0;
  for(ii=0;ii<=lmax;ii++) {
    int ib_test,ib2;
    double l=ii;

    //Check efficient searching
    ib=nmt_bins_flat_search_fast(bin,l,ib);
    //Check inefficient searching
    ib2=nmt_bins_flat_search_fast(bin,l,-15);
    if((l<2) || (l>=bin->ell_f_list[bin->n_bands-1]))
      ib_test=-1;
    else
      ib_test=(int)((l-2)/5);
    ASSERT_EQUAL(ib_test,ib);
    ASSERT_EQUAL(ib_test,ib2);
  }
  nmt_bins_flat_free(bin);
}

CTEST(nmt,bins_flat_alloc) {
  int ii;
  nmt_binning_scheme_flat *b_a;
  nmt_binning_scheme_flat *b_b=nmt_bins_flat_constant(5,2000);
  double *l0=my_malloc(b_b->n_bands*sizeof(double));
  double *lf=my_malloc(b_b->n_bands*sizeof(double));

  for(ii=0;ii<b_b->n_bands;ii++) {
    l0[ii]=2+5*ii;
    lf[ii]=2+5*(ii+1);
  }
  b_a=nmt_bins_flat_create(b_b->n_bands,l0,lf);

  for(ii=0;ii<b_b->n_bands;ii++) {
    ASSERT_DBL_NEAR_TOL(l0[ii],b_b->ell_0_list[ii],1E-10);
    ASSERT_DBL_NEAR_TOL(lf[ii],b_b->ell_f_list[ii],1E-10);
    ASSERT_DBL_NEAR_TOL(l0[ii],b_a->ell_0_list[ii],1E-10);
    ASSERT_DBL_NEAR_TOL(lf[ii],b_a->ell_f_list[ii],1E-10);
  }
  
  free(l0);
  free(lf);
  nmt_bins_flat_free(b_b);
  nmt_bins_flat_free(b_a);
}

CTEST(nmt,bins_flat_binning) {
  int ii;
  int lmax=2000;
  double *cl=my_malloc((lmax+1)*sizeof(double));
  nmt_binning_scheme_flat *bin=nmt_bins_flat_constant(5,2000);
  double *cl_binned=my_malloc(bin->n_bands*sizeof(double));
  double *leff=my_malloc(bin->n_bands*sizeof(double));

  for(ii=0;ii<=lmax;ii++)
    cl[ii]=ii;

  nmt_bin_cls_flat(bin,lmax+1,cl,&cl,&cl_binned,1);
  nmt_ell_eff_flat(bin,leff);
  nmt_unbin_cls_flat(bin,&cl_binned,lmax+1,cl,&cl,1);

  for(ii=0;ii<bin->n_bands;ii++) {
    ASSERT_DBL_NEAR_TOL(0.5*(bin->ell_0_list[ii]+bin->ell_f_list[ii]),leff[ii],1E-10);
    ASSERT_DBL_NEAR_TOL(leff[ii],cl_binned[ii],1E-5);
  }
  int ib=0;
  for(ii=0;ii<=lmax;ii++) {
    ib=nmt_bins_flat_search_fast(bin,(double)ii,ib);
    if(ib>=0)
      ASSERT_DBL_NEAR_TOL(cl_binned[ib],cl[ii],1E-5);
  }
  
  nmt_bins_flat_free(bin);
  free(cl);
  free(cl_binned);
  free(leff);
}
