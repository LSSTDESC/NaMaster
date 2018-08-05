#include "namaster.h"
#include "ctest.h"
#include "utils.h"
#include "nmt_test_utils.h"

CTEST_SKIP(nmt,cell_00_full) {
  long nside,ipix;
  double *mpt=he_read_healpix_map("test/maps.fits",&nside,0);
  double *msk=he_read_healpix_map("test/mask.fits",&nside,0);
  nmt_field *f0=nmt_field_alloc_sph(nside,msk,0,&mpt,0,NULL,NULL,0,0,3,1E-10);
  nmt_binning_scheme *bin=nmt_bins_constant(20,3*nside-1);
  nmt_workspace *w00=nmt_compute_coupling_matrix(f0,f0,bin);
  double **cell=malloc(1*sizeof(double *));
  double **cell_out=malloc(1*sizeof(double *));
  double **cell_noise=malloc(1*sizeof(double *));
  ASSERT_NOT_NULL(cell); ASSERT_NOT_NULL(cell_noise); ASSERT_NOT_NULL(cell_out);
  cell[0]=malloc(3*nside*sizeof(double));
  cell_noise[0]=calloc(3*nside,sizeof(double));
  cell_out[0]=malloc(bin->n_bands*sizeof(double));
  ASSERT_NOT_NULL(cell[0]); ASSERT_NOT_NULL(cell_noise[0]); ASSERT_NOT_NULL(cell_out[0]);

  nmt_compute_coupled_cell(f0,f0,cell,3);
  nmt_decouple_cl_l(w00,cell,cell_noise,cell_noise,cell_out);
  test_compare_arrays(bin->n_bands,cell_out[0],1,0,"test/benchmarks/cl_00_benchmark.txt");
  
  free(cell[0]); free(cell);
  free(cell_noise[0]); free(cell_noise);
  free(cell_out[0]); free(cell_out);
  nmt_workspace_free(w00);
  nmt_field_free(f0);
  free(mpt);
  free(msk);
}

CTEST_SKIP(nmt,cell_02_full) {
  long nside,ipix;
  double *mpt=he_read_healpix_map("test/maps.fits",&nside,0);
  double *mpq=he_read_healpix_map("test/maps.fits",&nside,1);
  double *mpu=he_read_healpix_map("test/maps.fits",&nside,2);
  double *msk=he_read_healpix_map("test/mask.fits",&nside,0);
  double **mpp=malloc(2*sizeof(double *));
  mpp[0]=mpq; mpp[1]=mpu;

  nmt_field *f0=nmt_field_alloc_sph(nside,msk,0,&mpt,0,NULL,NULL,0,0,3,1E-10);
  nmt_field *f2=nmt_field_alloc_sph(nside,msk,1,mpp,0,NULL,NULL,0,0,3,1E-10);
  nmt_binning_scheme *bin=nmt_bins_constant(20,3*nside-1);

  nmt_workspace *w02=nmt_compute_coupling_matrix(f0,f2,bin);
  double **cell=malloc(2*sizeof(double *));
  double **cell_out=malloc(2*sizeof(double *));
  double **cell_noise=malloc(2*sizeof(double *));
  ASSERT_NOT_NULL(cell); ASSERT_NOT_NULL(cell_noise); ASSERT_NOT_NULL(cell_out);
  for(ipix=0;ipix<2;ipix++) {
    cell[ipix]=malloc(3*nside*sizeof(double));
    cell_noise[ipix]=calloc(3*nside,sizeof(double));
    cell_out[ipix]=malloc(bin->n_bands*sizeof(double));
    ASSERT_NOT_NULL(cell[ipix]); ASSERT_NOT_NULL(cell_noise[ipix]); ASSERT_NOT_NULL(cell_out[ipix]);
  }

  nmt_compute_coupled_cell(f0,f2,cell,3);
  nmt_decouple_cl_l(w02,cell,cell_noise,cell_noise,cell_out);
  for(ipix=0;ipix<2;ipix++)
    test_compare_arrays(bin->n_bands,cell_out[ipix],2,ipix,"test/benchmarks/cl_02_benchmark.txt");
  
  for(ipix=0;ipix<2;ipix++) {
    free(cell[ipix]);
    free(cell_noise[ipix]);
    free(cell_out[ipix]); 
  }
  free(cell_out);
  free(cell_noise);
  free(cell);
  nmt_workspace_free(w02);
  nmt_field_free(f0);
  nmt_field_free(f2);
  free(mpt);
  free(mpq);
  free(mpu);
  free(mpp);
  free(msk);
}

CTEST_SKIP(nmt,cell_22_full) {
  long nside,ipix;
  double *mpq=he_read_healpix_map("test/maps.fits",&nside,1);
  double *mpu=he_read_healpix_map("test/maps.fits",&nside,2);
  double *msk=he_read_healpix_map("test/mask.fits",&nside,0);
  double **mpp=malloc(2*sizeof(double *));
  mpp[0]=mpq; mpp[1]=mpu;

  nmt_field *f2=nmt_field_alloc_sph(nside,msk,1,mpp,0,NULL,NULL,0,0,3,1E-10);
  nmt_binning_scheme *bin=nmt_bins_constant(20,3*nside-1);

  nmt_workspace *w22=nmt_compute_coupling_matrix(f2,f2,bin);
  double **cell=malloc(4*sizeof(double *));
  double **cell_out=malloc(4*sizeof(double *));
  double **cell_noise=malloc(4*sizeof(double *));
  ASSERT_NOT_NULL(cell); ASSERT_NOT_NULL(cell_noise); ASSERT_NOT_NULL(cell_out);
  for(ipix=0;ipix<4;ipix++) {
    cell[ipix]=malloc(3*nside*sizeof(double));
    cell_noise[ipix]=calloc(3*nside,sizeof(double));
    cell_out[ipix]=malloc(bin->n_bands*sizeof(double));
    ASSERT_NOT_NULL(cell[ipix]); ASSERT_NOT_NULL(cell_noise[ipix]); ASSERT_NOT_NULL(cell_out[ipix]);
  }

  nmt_compute_coupled_cell(f2,f2,cell,3);
  nmt_decouple_cl_l(w22,cell,cell_noise,cell_noise,cell_out);
  for(ipix=0;ipix<4;ipix++)
    test_compare_arrays(bin->n_bands,cell_out[ipix],4,ipix,"test/benchmarks/cl_22_benchmark.txt");
  
  for(ipix=0;ipix<4;ipix++) {
    free(cell[ipix]);
    free(cell_noise[ipix]);
    free(cell_out[ipix]); 
  }
  free(cell_out);
  free(cell_noise);
  free(cell);
  nmt_workspace_free(w22);
  nmt_field_free(f2);
  free(mpq);
  free(mpu);
  free(mpp);
  free(msk);
}
