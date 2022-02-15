#include "namaster.h"
#include "ctest.h"
#include "utils.h"
#include "nmt_test_utils.h"

CTEST(nmt,covar_f_ell) {
  int ii;
  nmt_curvedsky_info *cs=nmt_curvedsky_info_alloc(1,1,-1,-1,-1,-1,-1,-1,-1);
  double *msk=he_read_map("test/benchmarks/msk.fits",cs,0);
  double *map=he_read_map("test/benchmarks/mps.fits",cs,0);
  nmt_workspace *w=nmt_workspace_read_fits("test/benchmarks/bm_nc_np_w00.fits",1);
  nmt_field *f0=nmt_field_alloc_sph(cs,msk,0,&map,0,NULL,NULL,0,0,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  nmt_covar_workspace *cw=nmt_covar_workspace_init(f0,f0,f0,f0,w->bin->ell_max,HE_NITER_DEFAULT,-1,-1,-1,0);
  nmt_field_free(f0);
  free(msk); free(map);
  nmt_bins_free(w->bin);
  w->bin=nmt_bins_constant(16,191,2);

  //Init power spectra
  int ncls=1;
  long lmax=he_get_lmax(cs);
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

  double *covar=my_malloc(w->bin->n_bands*w->bin->n_bands*sizeof(double));
  nmt_compute_gaussian_covariance(cw,0,0,0,0,w,w,&cell,&cell,&cell,&cell,covar);

  fi=my_fopen("test/benchmarks/bm_nc_np_cov.txt","r");
  for(ii=0;ii<w->bin->n_bands*w->bin->n_bands;ii++) {
    double cov;
    int stat=fscanf(fi,"%lf",&cov);
    ASSERT_DBL_NEAR_TOL(cov,covar[ii],fabs(fmax(cov,covar[ii]))*1E-3);
  }
  fclose(fi);

  free(covar);
  free(cell);
  nmt_covar_workspace_free(cw);
  nmt_workspace_free(w);
  free(cs);
}

CTEST(nmt,covar) {
  int ii;
  nmt_curvedsky_info *cs=nmt_curvedsky_info_alloc(1,1,-1,-1,-1,-1,-1,-1,-1);
  double *msk=he_read_map("test/benchmarks/msk.fits",cs,0);
  double *map=he_read_map("test/benchmarks/mps.fits",cs,0);
  nmt_workspace *w=nmt_workspace_read_fits("test/benchmarks/bm_nc_np_w00.fits",1);
  nmt_field *f0=nmt_field_alloc_sph(cs,msk,0,&map,0,NULL,NULL,0,0,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  nmt_covar_workspace *cw=nmt_covar_workspace_init(f0,f0,f0,f0,w->bin->ell_max,HE_NITER_DEFAULT,-1,-1,-1,0);
  nmt_covar_workspace *cwr=nmt_covar_workspace_read_fits("test/benchmarks/bm_nc_np_cw00.fits", 0);
  nmt_field_free(f0);
  free(msk); free(map);

  ASSERT_EQUAL(cwr->lmax,cw->lmax);
  for(ii=0;ii<=cw->lmax;ii++) {
    int jj;
    for(jj=0;jj<=cw->lmax;jj++) {
      ASSERT_EQUAL(cwr->xi00_1122[ii][jj],cw->xi00_1122[ii][jj]);
      ASSERT_EQUAL(cwr->xi00_1221[ii][jj],cw->xi00_1221[ii][jj]);
    }
  }

  nmt_covar_workspace_free(cwr);

  //Init power spectra
  int ncls=1;
  long lmax=he_get_lmax(cs);
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

  double *covar=my_malloc(w->bin->n_bands*w->bin->n_bands*sizeof(double));
  nmt_compute_gaussian_covariance(cw,0,0,0,0,w,w,&cell,&cell,&cell,&cell,covar);

  fi=my_fopen("test/benchmarks/bm_nc_np_cov.txt","r");
  for(ii=0;ii<w->bin->n_bands*w->bin->n_bands;ii++) {
    double cov;
    int stat=fscanf(fi,"%lf",&cov);
    ASSERT_DBL_NEAR_TOL(cov,covar[ii],fabs(fmax(cov,covar[ii]))*1E-3);
  }
  fclose(fi);

  free(covar);
  free(cell);
  nmt_covar_workspace_free(cw);
  nmt_workspace_free(w);
  free(cs);
}
  
CTEST(nmt,covar_errors) {
  nmt_covar_workspace *cw=NULL;
  nmt_curvedsky_info *cs=nmt_curvedsky_info_alloc(1,1,-1,-1,-1,-1,-1,-1,-1);
  double *msk=he_read_map("test/benchmarks/msk.fits",cs,0);
  double *map=he_read_map("test/benchmarks/mps.fits",cs,0);
  nmt_field *f0=nmt_field_alloc_sph(cs,msk,0,&map,0,NULL,NULL,0,0,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  nmt_field *f0b=nmt_field_alloc_sph(cs,msk,0,&map,0,NULL,NULL,0,0,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  nmt_binning_scheme *binb=nmt_bins_constant(16,191,2);
  int nside=f0->cs->n_eq;
  free(msk); free(map);
  free(cs);
  
  set_error_policy(THROW_ON_ERROR);

  //All good
  try { cw=nmt_covar_workspace_init(f0,f0,f0b,f0b,binb->ell_max,HE_NITER_DEFAULT,-1,-1,-1,0); }
  ASSERT_EQUAL(0,nmt_exception_status);
  nmt_covar_workspace_free(cw); cw=NULL;
  //Wrong reading
  try { cw=nmt_covar_workspace_read_fits("none", 0); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(cw);

  //Correct reading
  try { cw=nmt_covar_workspace_read_fits("test/benchmarks/bm_nc_np_cw00.fits", 0); }
  ASSERT_EQUAL(0,nmt_exception_status);
  nmt_covar_workspace_free(cw); cw=NULL;

  //Incompatible resolutions
  f0b->cs->n_eq=128;
  try { cw=nmt_covar_workspace_init(f0,f0,f0b,f0b,binb->ell_max,HE_NITER_DEFAULT,-1,-1,-1,0); }
  f0b->cs->n_eq=nside;
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(cw);

  nmt_field_free(f0);
  nmt_field_free(f0b);
  nmt_bins_free(binb);
  set_error_policy(EXIT_ON_ERROR);
}
