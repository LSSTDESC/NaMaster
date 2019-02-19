#include "namaster.h"
#include "ctest.h"
#include "utils.h"
#include "nmt_test_utils.h"

CTEST(nmt,covar_f_ell) {
  int ii;
  nmt_curvedsky_info *cs=nmt_curvedsky_info_alloc(1,1,-1,-1,-1,-1,-1,-1);
  double *msk=he_read_map("test/benchmarks/msk.fits",cs,0);
  double *map=he_read_map("test/benchmarks/mps.fits",cs,0);
  nmt_workspace *w=nmt_workspace_read("test/benchmarks/bm_nc_np_w00.dat");
  nmt_field *f0=nmt_field_alloc_sph(cs,msk,0,&map,0,NULL,NULL,0,0,3,1E-10,HE_NITER_DEFAULT);
  nmt_binning_scheme *bin=nmt_bins_constant(16,191,2);
  nmt_covar_workspace *cw=nmt_covar_workspace_init(f0,f0,bin,f0,f0,bin,HE_NITER_DEFAULT);
  nmt_field_free(f0);
  nmt_bins_free(bin);
  free(msk); free(map);
  free(cs);

  //Init power spectra
  int ncls=1;
  long lmax=he_get_lmax(cw->cs);
  double *cell=my_malloc((lmax+1)*sizeof(double));
  //Read signal and noise power spectrum
  FILE *fi=my_fopen("test/benchmarks/cls_lss.txt","r");
  for(ii=0;ii<=lmax;ii++) {
    double dum,cl,nl;
    int stat=fscanf(fi,"%lf %lf %lf %lf %lf %lf %lf %lf %lf",&dum,
		    &cl,&dum,&dum,&dum,&nl,&dum,&dum,&dum);
    ASSERT_EQUAL(stat,9);
    cell[ii]=cl+nl;
    cell[ii]*=2*M_PI/(ii*(ii+1.));
  }
  fclose(fi);

  double *covar=my_malloc(cw->bin_a->n_bands*cw->bin_b->n_bands*sizeof(double));
  nmt_compute_gaussian_covariance(cw,w,w,cell,cell,cell,cell,covar);

  fi=my_fopen("test/benchmarks/bm_nc_np_cov.txt","r");
  for(ii=0;ii<cw->bin_a->n_bands*cw->bin_b->n_bands;ii++) {
    double cov;
    int stat=fscanf(fi,"%lf",&cov);
    ASSERT_DBL_NEAR_TOL(cov,covar[ii],fabs(fmax(cov,covar[ii]))*1E-3);
  }
  fclose(fi);

  free(covar);
  free(cell);
  nmt_covar_workspace_free(cw);
  nmt_workspace_free(w);
}

CTEST(nmt,covar) {
  int ii;
  nmt_curvedsky_info *cs=nmt_curvedsky_info_alloc(1,1,-1,-1,-1,-1,-1,-1);
  double *msk=he_read_map("test/benchmarks/msk.fits",cs,0);
  double *map=he_read_map("test/benchmarks/mps.fits",cs,0);
  nmt_workspace *w=nmt_workspace_read("test/benchmarks/bm_nc_np_w00.dat");
  nmt_field *f0=nmt_field_alloc_sph(cs,msk,0,&map,0,NULL,NULL,0,0,3,1E-10,HE_NITER_DEFAULT);
  nmt_binning_scheme *bin=w->bin;
  nmt_covar_workspace *cw=nmt_covar_workspace_init(f0,f0,bin,f0,f0,bin,HE_NITER_DEFAULT);
  nmt_covar_workspace *cwr=nmt_covar_workspace_read("test/benchmarks/bm_nc_np_cw00.dat");
  nmt_field_free(f0);
  free(msk); free(map);
  free(cs);

  ASSERT_EQUAL(cwr->lmax_a,cw->lmax_a);
  ASSERT_EQUAL(cwr->lmax_b,cw->lmax_b);
  ASSERT_EQUAL(cwr->ncls_a,cw->ncls_a);
  ASSERT_EQUAL(cwr->ncls_b,cw->ncls_b);
  ASSERT_TRUE(nmt_diff_curvedsky_info(cwr->cs,cw->cs));
  for(ii=0;ii<cw->ncls_a*(cw->lmax_a+1);ii++) {
    int jj;
    for(jj=0;jj<cw->ncls_b*(cw->lmax_b+1);jj++) {
      ASSERT_EQUAL(cwr->xi_1122[ii][jj],cw->xi_1122[ii][jj]);
      ASSERT_EQUAL(cwr->xi_1221[ii][jj],cw->xi_1221[ii][jj]);
    }
  }

  nmt_covar_workspace_free(cwr);

  //Init power spectra
  int ncls=1;
  long lmax=he_get_lmax(cw->cs);
  double *cell=my_malloc((lmax+1)*sizeof(double));
  //Read signal and noise power spectrum
  FILE *fi=my_fopen("test/benchmarks/cls_lss.txt","r");
  for(ii=0;ii<=lmax;ii++) {
    double dum,cl,nl;
    int stat=fscanf(fi,"%lf %lf %lf %lf %lf %lf %lf %lf %lf",&dum,
		    &cl,&dum,&dum,&dum,&nl,&dum,&dum,&dum);
    ASSERT_EQUAL(stat,9);
    cell[ii]=cl+nl;
  }
  fclose(fi);

  double *covar=my_malloc(cw->bin_a->n_bands*cw->bin_b->n_bands*sizeof(double));
  nmt_compute_gaussian_covariance(cw,w,w,cell,cell,cell,cell,covar);

  fi=my_fopen("test/benchmarks/bm_nc_np_cov.txt","r");
  for(ii=0;ii<cw->bin_a->n_bands*cw->bin_b->n_bands;ii++) {
    double cov;
    int stat=fscanf(fi,"%lf",&cov);
    ASSERT_DBL_NEAR_TOL(cov,covar[ii],fabs(fmax(cov,covar[ii]))*1E-3);
  }
  fclose(fi);

  free(covar);
  free(cell);
  nmt_covar_workspace_free(cw);
  nmt_workspace_free(w);
}
  
CTEST(nmt,covar_errors) {
  nmt_covar_workspace *cw=NULL;
  nmt_curvedsky_info *cs=nmt_curvedsky_info_alloc(1,1,-1,-1,-1,-1,-1,-1);
  double *msk=he_read_map("test/benchmarks/msk.fits",cs,0);
  double *map=he_read_map("test/benchmarks/mps.fits",cs,0);
  nmt_field *f0=nmt_field_alloc_sph(cs,msk,0,&map,0,NULL,NULL,0,0,3,1E-10,HE_NITER_DEFAULT);
  nmt_field *f0b=nmt_field_alloc_sph(cs,msk,0,&map,0,NULL,NULL,0,0,3,1E-10,HE_NITER_DEFAULT);
  nmt_binning_scheme *bin=nmt_bins_constant(16,191,2);
  nmt_binning_scheme *binb=nmt_bins_constant(16,191,2);
  //nmt_workspace *wb,*wa=nmt_workspace_read("test/benchmarks/bm_nc_np_w00.dat");
  int nside=f0->cs->n_eq;
  free(msk); free(map);
  free(cs);

  set_error_policy(THROW_ON_ERROR);

  //All good
  try { cw=nmt_covar_workspace_init(f0,f0,bin,f0b,f0b,binb,HE_NITER_DEFAULT); }
  ASSERT_EQUAL(0,nmt_exception_status);
  nmt_covar_workspace_free(cw); cw=NULL;
  //Wrong reading
  try { cw=nmt_covar_workspace_read("none"); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(cw);

  //Correct reading
  try { cw=nmt_covar_workspace_read("test/benchmarks/bm_nc_np_cw00.dat"); }
  ASSERT_EQUAL(0,nmt_exception_status);
  nmt_covar_workspace_free(cw); cw=NULL;

  //Incompatible resolutions
  f0b->cs->n_eq=128;
  try { cw=nmt_covar_workspace_init(f0,f0,bin,f0b,f0b,binb,HE_NITER_DEFAULT); }
  f0b->cs->n_eq=nside;
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(cw);

  //Incompatible lmax
  binb->ell_max=3*128-1;
  try { cw=nmt_covar_workspace_init(f0,f0,bin,f0b,f0b,binb,HE_NITER_DEFAULT); }
  binb->ell_max=bin->ell_max;
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(cw);

  nmt_field_free(f0);
  nmt_field_free(f0b);
  nmt_bins_free(bin);
  nmt_bins_free(binb);
  set_error_policy(EXIT_ON_ERROR);
}
