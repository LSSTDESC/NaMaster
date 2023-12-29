#include "namaster.h"
#include "ctest.h"
#include "utils.h"
#include "nmt_test_utils.h"

CTEST(nmt,bins_constant) {
  nmt_binning_scheme *bin=nmt_bins_constant(4,2000,0);
  ASSERT_EQUAL(bin->n_bands,499);
  ASSERT_EQUAL(bin->ell_list[5][2],2+4*5+2);
  nmt_bins_free(bin);
}

CTEST(nmt,bins_f_ell) {
  int i,j;
  nmt_binning_scheme *bin_nc,*bin_c=nmt_bins_constant(4,2000,1);
  int *ells=test_get_sequence(2,1998);
  int *bpws=malloc(1996*sizeof(int));
  double *weights=malloc(1996*sizeof(double));
  double *f_ell=malloc(1996*sizeof(double));
  for(i=0;i<1996;i++) {
    bpws[i]=i/4;
    weights[i]=0.25;
    f_ell[i]=ells[i]*(ells[i]+1)/(2*M_PI);
  }
  bin_nc=nmt_bins_create(1996,bpws,ells,weights,f_ell,2000);

  for(i=0;i<499;i++) {
    for(j=0;j<4;j++) {
      int ell=2+4*i+j;
      ASSERT_DBL_NEAR_TOL(bin_c->f_ell[i][j],ell*(ell+1)/(2*M_PI),1E-5);
      ASSERT_DBL_NEAR_TOL(bin_c->f_ell[i][j],bin_nc->f_ell[i][j],1E-5);
    }
  }

  free(bpws);
  free(ells);
  free(weights);
  free(f_ell);
  nmt_bins_free(bin_c);
  nmt_bins_free(bin_nc);
}

CTEST(nmt,bins_variable) {
  int i,j;
  nmt_binning_scheme *bin=nmt_bins_constant(4,2000,0);
  int *ells=test_get_sequence(2,1998);
  int *bpws=malloc(1996*sizeof(int));
  double *weights=malloc(1996*sizeof(double));
  for(i=0;i<1996;i++) {
    bpws[i]=i/4;
    weights[i]=0.25;
  }
  nmt_binning_scheme *bin2=nmt_bins_create(1996,bpws,ells,weights,NULL,2000);

  ASSERT_EQUAL(bin->n_bands,499);
  ASSERT_EQUAL(bin->n_bands,bin2->n_bands);
  for(i=0;i<499;i++) {
    for(j=0;j<4;j++) {
      ASSERT_EQUAL(bin->ell_list[i][j],2+4*i+j);
      ASSERT_EQUAL(bin2->ell_list[i][j],2+4*i+j);
    }
  }

  set_error_policy(THROW_ON_ERROR);

  nmt_bins_free(bin2); bin2=NULL;
  for(i=0;i<4;i++)
    weights[16+i]=0;
  try { bin2=nmt_bins_create(1996,bpws,ells,weights,NULL,2000); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(bin2);
  set_error_policy(EXIT_ON_ERROR);

  free(bpws);
  free(ells);
  free(weights);
  nmt_bins_free(bin);
}

CTEST(nmt,bins_binning) {
  int i,nside=256;
  int lmax=3*nside-1;
  double *cls=my_malloc((lmax+1)*sizeof(double));
  double *cls_back=my_malloc((lmax+1)*sizeof(double));
  nmt_binning_scheme *bin=nmt_bins_constant(4,lmax,0);
  double *cls_binned=my_malloc(bin->n_bands*sizeof(double));
  double *leff=my_malloc(bin->n_bands*sizeof(double));

  for(i=0;i<=lmax;i++)
    cls[i]=i+0.;

  nmt_bin_cls(bin,&cls,&cls_binned,1);
  nmt_unbin_cls(bin,&cls_binned,&cls_back,1);
  nmt_ell_eff(bin,leff);

  for(i=0;i<bin->n_bands;i++) {
    int j;
    ASSERT_DBL_NEAR_TOL(2+4*i+1.5,cls_binned[i],1E-5);
    ASSERT_DBL_NEAR_TOL(leff[i],cls_binned[i],1E-5);
    for(j=0;j<4;j++)
      ASSERT_DBL_NEAR_TOL(2+4*i+1.5,cls_back[2+4*i+j],1E-5);
  }

  nmt_bins_free(bin);
  free(leff);
  free(cls_binned);
  free(cls);
  free(cls_back);
}

CTEST(nmt,bins_binning_f_ell) {
  int i,nside=256;
  int lmax=3*nside-1;
  double *cls=my_malloc((lmax+1)*sizeof(double));
  double *cls_back=my_malloc((lmax+1)*sizeof(double));
  nmt_binning_scheme *bin=nmt_bins_constant(4,lmax,2);
  double *cls_binned=my_malloc(bin->n_bands*sizeof(double));
  double *leff=my_malloc(bin->n_bands*sizeof(double));

  for(i=0;i<=lmax;i++)
    cls[i]=i+0.;

  nmt_bin_cls(bin,&cls,&cls_binned,1);
  nmt_unbin_cls(bin,&cls_binned,&cls_back,1);
  nmt_ell_eff(bin,leff);

  for(i=0;i<bin->n_bands;i++) {
    int j;
    int l0=2+4*i;
    ASSERT_DBL_NEAR_TOL((25+l0*(27+l0*(11+2*l0)))/(4*M_PI),cls_binned[i],1E-5);
    for(j=0;j<4;j++) {
      int ell=l0+j;
      ASSERT_DBL_NEAR_TOL((25+l0*(27+l0*(11+2*l0)))/(2.*ell*(ell+1.)),cls_back[l0+j],1E-5);
    }
  }

  nmt_bins_free(bin);
  free(leff);
  free(cls_binned);
  free(cls);
  free(cls_back);
}
