#include "namaster.h"
#include "ctest.h"
#include "utils.h"
#include "nmt_test_utils.h"

CTEST(nmt,covar) {
  int ii;
  nmt_workspace_flat *w=nmt_workspace_flat_read("test/benchmarks/bm_f_nc_np_w00.dat");
  nmt_covar_workspace_flat *cw=nmt_covar_workspace_flat_init(w,w);
  nmt_covar_workspace_flat *cwr=nmt_covar_workspace_flat_read("test/benchmarks/bm_f_nc_np_cw00.dat");

  ASSERT_EQUAL(cwr->ncls_a,cw->ncls_a);
  ASSERT_EQUAL(cwr->ncls_b,cw->ncls_b);
  ASSERT_EQUAL(cwr->bin->n_bands,cw->bin->n_bands);
  for(ii=0;ii<cw->ncls_a*cw->bin->n_bands;ii++) {
    int jj;
    for(jj=0;jj<cw->ncls_b*cw->bin->n_bands;jj++) {
      ASSERT_EQUAL(cwr->xi_1122[ii][jj],cw->xi_1122[ii][jj]);
      ASSERT_EQUAL(cwr->xi_1221[ii][jj],cw->xi_1221[ii][jj]);
    }
  }
  nmt_workspace_flat_free(w);
  nmt_covar_workspace_flat_free(cwr);

  //Init power spectra
  int ncls=1;
  int lmax_th=2999;
  double *larr=my_malloc((lmax_th+1)*sizeof(double));
  double *cell=my_malloc((lmax_th+1)*sizeof(double));
  //Read signal and noise power spectrum
  FILE *fi=my_fopen("test/benchmarks/cls_lss.txt","r");
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
  nmt_compute_gaussian_covariance_flat(cw,lmax_th+1,larr,cell,cell,cell,cell,covar);

  fi=my_fopen("test/benchmarks/bm_f_nc_np_cov.txt","r");
  for(ii=0;ii<cw->bin->n_bands*cw->bin->n_bands;ii++) {
    double cov;
    int stat=fscanf(fi,"%lf",&cov);
    ASSERT_DBL_NEAR_TOL(cov,covar[ii],fabs(fmax(cov,covar[ii]))*1E-3);
  }
  fclose(fi);
  free(covar);
  free(cell);
  nmt_covar_workspace_flat_free(cw);
}

CTEST(nmt,covar_flat_errors) {
  nmt_covar_workspace_flat *cw=NULL;
  nmt_workspace_flat *wb,*wa=nmt_workspace_flat_read("test/benchmarks/bm_f_nc_np_w00.dat");

  set_error_policy(THROW_ON_ERROR);

  //All good
  wb=nmt_workspace_flat_read("test/benchmarks/bm_f_nc_np_w00.dat");
  try { cw=nmt_covar_workspace_flat_init(wa,wb); }
  ASSERT_EQUAL(0,nmt_exception_status);
  nmt_covar_workspace_flat_free(cw); cw=NULL;

  //Wrong reading
  try { cw=nmt_covar_workspace_flat_read("none"); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(cw);

  //Correct reading
  try { cw=nmt_covar_workspace_flat_read("test/benchmarks/bm_f_nc_np_cw00.dat"); }
  ASSERT_EQUAL(0,nmt_exception_status);
  nmt_covar_workspace_flat_free(cw); cw=NULL;

  //Incompatible resolutions
  int nx=wb->fs->nx,ny=wb->fs->ny,n_bands=wb->bin->n_bands;
  wb->fs->nx=2;
  try { cw=nmt_covar_workspace_flat_init(wa,wb); }
  wb->fs->nx=nx;
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(cw);
  wb->fs->ny=2;
  try { cw=nmt_covar_workspace_flat_init(wa,wb); }
  wb->fs->ny=ny;
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(cw);

  //Incompatible bandpowers
  wb->bin->n_bands=2;
  try { cw=nmt_covar_workspace_flat_init(wa,wb); }
  wb->bin->n_bands=n_bands;
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(cw);
  nmt_workspace_flat_free(wb);

  //Spin-2
  wb=nmt_workspace_flat_read("test/benchmarks/bm_f_nc_np_w02.dat");
  try { cw=nmt_covar_workspace_flat_init(wa,wb); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(cw);
  nmt_workspace_flat_free(wb);

  set_error_policy(EXIT_ON_ERROR);

  nmt_workspace_flat_free(wa);
}
