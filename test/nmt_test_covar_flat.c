#include "namaster.h"
#include "ctest.h"
#include "utils.h"
#include "nmt_test_utils.h"

CTEST(nmt,covar_flat) {
  int ii,nx,ny;
  double lx,ly;
  nmt_binning_scheme_flat *bin;
  double *msk=fs_read_flat_map("test/benchmarks/msk_flat.fits",&nx,&ny,&lx,&ly,0);
  double *map=fs_read_flat_map("test/benchmarks/msk_flat.fits",&nx,&ny,&lx,&ly,0);
  int nell=25;
  double dell=20.;
  double *larr_e=my_malloc((nell+1)*sizeof(double));
  for(ii=0;ii<=nell;ii++)
    larr_e[ii]=ii*dell+2;
  bin=nmt_bins_flat_create(nell,larr_e,&(larr_e[1]));
  free(larr_e);

  nmt_field_flat *f0=nmt_field_flat_alloc(NX_TEST,NY_TEST,
					  DX_TEST*NX_TEST*M_PI/180,
					  DY_TEST*NY_TEST*M_PI/180,
					  msk,0,&map,0,NULL,0,NULL,NULL,0,0,1E-10,0,0,0);
  
  nmt_workspace_flat *w=nmt_workspace_flat_read_fits("test/benchmarks/bm_f_nc_np_w00.fits");
  nmt_covar_workspace_flat *cw=nmt_covar_workspace_flat_init(f0,f0,bin,f0,f0,bin);
  nmt_covar_workspace_flat *cwr=nmt_covar_workspace_flat_read_fits("test/benchmarks/bm_f_nc_np_cw00.fits");
  free(msk); free(map); nmt_bins_flat_free(bin); nmt_field_flat_free(f0);

  ASSERT_EQUAL(cwr->bin->n_bands,cw->bin->n_bands);
  for(ii=0;ii<cw->bin->n_bands;ii++) {
    int jj;
    for(jj=0;jj<cw->bin->n_bands;jj++) {
      ASSERT_EQUAL(cwr->xi00_1122[ii][jj],cw->xi00_1122[ii][jj]);
      ASSERT_EQUAL(cwr->xi00_1221[ii][jj],cw->xi00_1221[ii][jj]);
    }
  }
  nmt_covar_workspace_flat_free(cwr);

  //Init power spectra
  int lmax_th=2999;
  double *larr=my_malloc((lmax_th+1)*sizeof(double));
  double *cell=my_malloc((lmax_th+1)*sizeof(double));
  //Read signal and noise power spectrum
  FILE *fi=fopen("test/benchmarks/cls_lss.txt","r");
  for(ii=0;ii<=lmax_th;ii++) {
    double dum,l,cl,nl;
    int stat=fscanf(fi,"%lf %lf %lf %lf %lf %lf %lf %lf %lf",&l,
		    &cl,&dum,&dum,&dum,&nl,&dum,&dum,&dum);
    ASSERT_EQUAL(stat,9);
    cell[ii]=cl+nl;
    larr[ii]=l;
  }
  fclose(fi);

  double *covar=my_malloc(cw->bin->n_bands*cw->bin->n_bands*sizeof(double));
  nmt_compute_gaussian_covariance_flat(cw,0,0,0,0,w,w,lmax_th+1,larr,&cell,&cell,&cell,&cell,covar);

  fi=fopen("test/benchmarks/bm_f_nc_np_cov.txt","r");
  for(ii=0;ii<cw->bin->n_bands*cw->bin->n_bands;ii++) {
    double cov;
    int stat=fscanf(fi,"%lf",&cov);
    ASSERT_DBL_NEAR_TOL(cov,covar[ii],fabs(fmax(cov,covar[ii]))*1E-3);
  }
  fclose(fi);
  free(covar);
  free(cell);
  nmt_workspace_flat_free(w);
  nmt_covar_workspace_flat_free(cw);
}

CTEST(nmt,covar_flat_errors) {
  nmt_covar_workspace_flat *cw=NULL;
  int ii,nx,ny;
  double lx,ly;
  nmt_binning_scheme_flat *bin,*binb;
  double *msk=fs_read_flat_map("test/benchmarks/msk_flat.fits",&nx,&ny,&lx,&ly,0);
  double *map=fs_read_flat_map("test/benchmarks/msk_flat.fits",&nx,&ny,&lx,&ly,0);
  int nell=25;
  double dell=20.;
  double *larr_e=my_malloc((nell+1)*sizeof(double));
  for(ii=0;ii<=nell;ii++)
    larr_e[ii]=ii*dell+2;
  bin=nmt_bins_flat_create(nell,larr_e,&(larr_e[1]));
  binb=nmt_bins_flat_create(nell,larr_e,&(larr_e[1]));
  free(larr_e);

  nmt_field_flat *f0=nmt_field_flat_alloc(NX_TEST,NY_TEST,
					  DX_TEST*NX_TEST*M_PI/180,
					  DY_TEST*NY_TEST*M_PI/180,
					  msk,0,&map,0,NULL,0,NULL,NULL,0,0,1E-10,0,0,0);
  
  nmt_field_flat *f0b=nmt_field_flat_alloc(NX_TEST,NY_TEST,
					   DX_TEST*NX_TEST*M_PI/180,
					   DY_TEST*NY_TEST*M_PI/180,
					   msk,0,&map,0,NULL,0,NULL,NULL,0,0,1E-10,0,0,0);
  


  set_error_policy(THROW_ON_ERROR);

  //All good
  try { cw=nmt_covar_workspace_flat_init(f0,f0,bin,f0b,f0b,binb); }
  ASSERT_EQUAL(0,nmt_exception_status);
  nmt_covar_workspace_flat_free(cw); cw=NULL;

  //Wrong reading
  try { cw=nmt_covar_workspace_flat_read_fits("none"); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(cw);

  //Correct reading
  try { cw=nmt_covar_workspace_flat_read_fits("test/benchmarks/bm_f_nc_np_cw00.fits"); }
  ASSERT_EQUAL(0,nmt_exception_status);
  nmt_covar_workspace_flat_free(cw); cw=NULL;

  //Incompatible resolutions
  int n_bands=bin->n_bands;
  nx=f0->fs->nx;
  ny=f0->fs->ny;
  f0b->fs->nx=2;
  try { cw=nmt_covar_workspace_flat_init(f0,f0,bin,f0b,f0b,binb); }
  f0b->fs->nx=nx;
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(cw);
  f0b->fs->ny=2;
  try { cw=nmt_covar_workspace_flat_init(f0,f0,bin,f0b,f0b,binb); }
  f0b->fs->ny=ny;
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(cw);

  //Incompatible bandpowers
  binb->n_bands=2;
  try { cw=nmt_covar_workspace_flat_init(f0,f0,bin,f0b,f0b,binb); }
  binb->n_bands=n_bands;
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(cw);

  set_error_policy(EXIT_ON_ERROR);

  free(msk); free(map);
  nmt_bins_flat_free(bin); nmt_field_flat_free(f0);
  nmt_bins_flat_free(binb); nmt_field_flat_free(f0b);
}
