#include "namaster.h"
#include "ctest.h"
#include "utils.h"
#include "nmt_test_utils.h"

CTEST(nmt,master_bias_uncorr) {
  //Generate fields and compute coupling matrix
  int ii,im1,ll;
  double prefac,f_fac;
  long nside,ipix,npix;
  nmt_field *f0,*f2;
  double **mps0=my_malloc(sizeof(double *));
  double **mps2=my_malloc(2*sizeof(double *));
  double **tmp0=my_malloc(sizeof(double *));
  double **tmp2=my_malloc(2*sizeof(double *));
  double **mp_dum=my_malloc(2*sizeof(double *));
  double *msk=he_read_healpix_map("test/benchmarks/msk.fits",&nside,0);
  mps0[0]=he_read_healpix_map("test/benchmarks/mps.fits",&nside,0);
  mps2[0]=he_read_healpix_map("test/benchmarks/mps.fits",&nside,1);
  mps2[1]=he_read_healpix_map("test/benchmarks/mps.fits",&nside,2);
  tmp0[0]=he_read_healpix_map("test/benchmarks/tmp.fits",&nside,0);
  tmp2[0]=he_read_healpix_map("test/benchmarks/tmp.fits",&nside,1);
  tmp2[1]=he_read_healpix_map("test/benchmarks/tmp.fits",&nside,2);
  npix=he_nside2npix(nside);
  for(ii=0;ii<2;ii++)
    mp_dum[ii]=my_malloc(npix*sizeof(double));
  
  //Init power spectra
  int ncls=4;
  double **cell=my_malloc(ncls*sizeof(double *));
  double **cell_tmp=my_malloc(ncls*sizeof(double *));
  double **cell_tmpb=my_malloc(ncls*sizeof(double *));
  for(ii=0;ii<ncls;ii++) {
    cell[ii]=my_malloc(3*nside*sizeof(double));
    cell_tmp[ii]=my_malloc(3*nside*sizeof(double));
    cell_tmpb[ii]=my_malloc(3*nside*sizeof(double));
  }
  
  f0=nmt_field_alloc_sph(nside,msk,0,mps0,1,&tmp0,NULL,0,0,3,1E-10);
  nmt_compute_uncorr_noise_deprojection_bias(f0,msk,cell);
  he_anafast(tmp0,tmp0,0,0,cell_tmp,nside,3*nside-1,3);
  f_fac=0;
  prefac=0;
  for(im1=0;im1<1;im1++) {
    he_map_product(nside,tmp0[im1],msk,tmp0[im1]); //Multiply by mask
    he_map_product(nside,tmp0[im1],msk,mp_dum[im1]); //f*v
    he_map_product(nside,mp_dum[im1],msk,mp_dum[im1]); //f*v^2
    he_map_product(nside,mp_dum[im1],msk,mp_dum[im1]); //f*v^2*s^2
    prefac+=he_map_dot(nside,mp_dum[im1],tmp0[im1]); //Sum[f*f*v^2*s^2 * dOmega]
    f_fac+=he_map_dot(nside,tmp0[im1],tmp0[im1]); //Sum[f*f * dOmega]
  }
  f_fac=1./f_fac; //1./Int[f^2]
  he_anafast(tmp0,tmp0,0,0,cell_tmp,nside,3*nside-1,3);
  he_anafast(mp_dum,tmp0,0,0,cell_tmpb,nside,3*nside-1,3);

  for(ii=0;ii<1;ii++) {
    for(ll=0;ll<3*nside;ll++) {
      double cl_pred=-2*f_fac*cell_tmpb[ii][ll]+f_fac*f_fac*prefac*cell_tmp[ii][ll];
      ASSERT_DBL_NEAR_TOL(cl_pred,cell[ii][ll],1E-3*fabs(cl_pred));
    }
  }
  
  f2=nmt_field_alloc_sph(nside,msk,1,mps2,1,&tmp2,NULL,0,0,3,1E-10);
  nmt_compute_uncorr_noise_deprojection_bias(f2,msk,cell);
  f_fac=0;
  prefac=0;
  for(im1=0;im1<2;im1++) {
    he_map_product(nside,tmp2[im1],msk,tmp2[im1]); //Multiply by mask
    he_map_product(nside,tmp2[im1],msk,mp_dum[im1]); //f*v
    he_map_product(nside,mp_dum[im1],msk,mp_dum[im1]); //f*v^2
    he_map_product(nside,mp_dum[im1],msk,mp_dum[im1]); //f*v^2*s^2
    prefac+=he_map_dot(nside,mp_dum[im1],tmp2[im1]); //Sum[f*f*v^2*s^2 * dOmega]
    f_fac+=he_map_dot(nside,tmp2[im1],tmp2[im1]); //Sum[f*f * dOmega]
  }
  f_fac=1./f_fac; //1./Int[f^2]
  he_anafast(tmp2,tmp2,1,1,cell_tmp,nside,3*nside-1,3);
  he_anafast(mp_dum,tmp2,1,1,cell_tmpb,nside,3*nside-1,3);

  for(ii=0;ii<4;ii++) {
    for(ll=0;ll<3*nside;ll++) {
      double cl_pred=-2*f_fac*cell_tmpb[ii][ll]+f_fac*f_fac*prefac*cell_tmp[ii][ll];
      ASSERT_DBL_NEAR_TOL(cl_pred,cell[ii][ll],1E-3*fabs(cl_pred));
    }
  }

  for(ii=0;ii<ncls;ii++) {
    free(cell[ii]);
    free(cell_tmp[ii]);
    free(cell_tmpb[ii]);
  }
  free(cell);
  free(cell_tmp);
  free(cell_tmpb);
  for(ii=0;ii<2;ii++) {
    free(tmp2[ii]);
    free(mps2[ii]);
  }
  free(tmp0[0]);
  free(mps0[0]);
  free(mps0);
  free(mps2);
  free(tmp0);
  free(tmp2);
  free(msk);
  nmt_field_free(f0);
  nmt_field_free(f2);
  for(ii=0;ii<2;ii++)
    free(mp_dum[ii]);
  free(mp_dum);
}

CTEST(nmt,master_teb_full) {
  //Checks that the TEB workspaces get the same thing as the 00, 02 and 22 workspaces put together.
  //Generate fields and compute coupling matrix
  int ii;
  long nside,ipix;
  nmt_field *f0,*f2;
  nmt_workspace *w_teb,*w00,*w02,*w22;
  double **mps0=my_malloc(sizeof(double *));
  double **mps2=my_malloc(2*sizeof(double *));
  double *msk=he_read_healpix_map("test/benchmarks/msk.fits",&nside,0);
  nmt_binning_scheme *bin=nmt_bins_constant(16,3*nside-1);
  mps0[0]=he_read_healpix_map("test/benchmarks/mps.fits",&nside,0);
  mps2[0]=he_read_healpix_map("test/benchmarks/mps.fits",&nside,1);
  mps2[1]=he_read_healpix_map("test/benchmarks/mps.fits",&nside,2);
  
  //Init power spectra
  int ncls=7;
  double **cell=my_malloc(ncls*sizeof(double *));
  double **cell_out=my_malloc(ncls*sizeof(double *));
  double **cell_out_teb=my_malloc(ncls*sizeof(double *));
  double **cell_signal=my_malloc(ncls*sizeof(double *));
  double **cell_noise=my_malloc(ncls*sizeof(double *));
  double **cell_deproj=my_malloc(ncls*sizeof(double *));
  for(ii=0;ii<ncls;ii++) {
    cell[ii]=my_malloc(3*nside*sizeof(double));
    cell_noise[ii]=my_calloc(3*nside,sizeof(double));
    cell_signal[ii]=my_calloc(3*nside,sizeof(double));
    cell_deproj[ii]=my_calloc(3*nside,sizeof(double));
    cell_out[ii]=my_malloc(bin->n_bands*sizeof(double));
    cell_out_teb[ii]=my_malloc(bin->n_bands*sizeof(double));
  }
  //Read signal and noise power spectrum
  FILE *fi=my_fopen("test/benchmarks/cls_lss.txt","r");
  for(ii=0;ii<3*nside;ii++) {
    int jj;
    double dum;
    int stat=fscanf(fi,"%lf %lf %lf %lf %lf %lf %lf %lf %lf",&dum,
		    &(cell_signal[0][ii]),&(cell_signal[3][ii]),&(cell_signal[6][ii]),&(cell_signal[1][ii]),
		    &(cell_noise[0][ii]),&(cell_noise[3][ii]),&(cell_noise[6][ii]),&(cell_noise[1][ii]));
    ASSERT_EQUAL(stat,9);
    for(jj=0;jj<ncls;jj++) //Add noise to signal
      cell_signal[jj][ii]+=cell_noise[jj][ii];
  }
  fclose(fi);
  
  //No contaminants
  f0=nmt_field_alloc_sph(nside,msk,0,mps0,0,NULL,NULL,0,0,3,1E-10);
  f2=nmt_field_alloc_sph(nside,msk,1,mps2,0,NULL,NULL,0,0,3,1E-10);
  w_teb=nmt_compute_coupling_matrix(f0,f2,bin,1);
  w00=nmt_compute_coupling_matrix(f0,f0,bin,0);
  w02=nmt_compute_coupling_matrix(f0,f2,bin,0);
  w22=nmt_compute_coupling_matrix(f2,f2,bin,0);
  nmt_compute_coupled_cell(f0,f0,&(cell[0]));
  nmt_compute_coupled_cell(f0,f2,&(cell[1]));
  nmt_compute_coupled_cell(f2,f2,&(cell[3]));
  nmt_decouple_cl_l(w_teb,cell,cell_noise,cell_deproj,cell_out_teb);
  nmt_decouple_cl_l(w00,&(cell[0]),&(cell_noise[0]),&(cell_deproj[0]),&(cell_out[0]));
  nmt_decouple_cl_l(w02,&(cell[1]),&(cell_noise[1]),&(cell_deproj[1]),&(cell_out[1]));
  nmt_decouple_cl_l(w22,&(cell[3]),&(cell_noise[3]),&(cell_deproj[3]),&(cell_out[3]));
  for(ii=0;ii<ncls;ii++) {
    int ll;
    for(ll=0;ll<bin->n_bands;ll++)
      ASSERT_DBL_NEAR_TOL(cell_out[ii][ll],cell_out_teb[ii][ll],1E-20);
  }
  nmt_workspace_free(w_teb);
  nmt_workspace_free(w00);
  nmt_workspace_free(w02);
  nmt_workspace_free(w22);
  nmt_field_free(f0);
  nmt_field_free(f2);
  
  //Free up power spectra
  for(ii=0;ii<ncls;ii++) {
    free(cell[ii]);
    free(cell_noise[ii]);
    free(cell_signal[ii]);
    free(cell_deproj[ii]);
    free(cell_out[ii]);
    free(cell_out_teb[ii]);
  }
  free(cell); free(cell_noise); free(cell_signal); free(cell_deproj);
  free(cell_out); free(cell_out_teb);

  nmt_bins_free(bin);
  for(ii=0;ii<2;ii++)
    free(mps2[ii]);
  free(mps0[0]);
  free(mps0);
  free(mps2);
  free(msk);
}

CTEST(nmt,master_22_full) {
  //Generate fields and compute coupling matrix
  int ii;
  long nside,ipix;
  nmt_field *f2;
  nmt_workspace *w22;
  double **mps2=my_malloc(2*sizeof(double *));
  double **tmp2=my_malloc(2*sizeof(double *));
  double *msk=he_read_healpix_map("test/benchmarks/msk.fits",&nside,0);
  nmt_binning_scheme *bin=nmt_bins_constant(16,3*nside-1);
  mps2[0]=he_read_healpix_map("test/benchmarks/mps.fits",&nside,1);
  mps2[1]=he_read_healpix_map("test/benchmarks/mps.fits",&nside,2);
  tmp2[0]=he_read_healpix_map("test/benchmarks/tmp.fits",&nside,1);
  tmp2[1]=he_read_healpix_map("test/benchmarks/tmp.fits",&nside,2);
  
  //Init power spectra
  int ncls=4;
  double **cell=my_malloc(ncls*sizeof(double *));
  double **cell_out=my_malloc(ncls*sizeof(double *));
  double **cell_signal=my_malloc(ncls*sizeof(double *));
  double **cell_noise=my_malloc(ncls*sizeof(double *));
  double **cell_deproj=my_malloc(ncls*sizeof(double *));
  for(ii=0;ii<ncls;ii++) {
    cell[ii]=my_malloc(3*nside*sizeof(double));
    cell_noise[ii]=my_calloc(3*nside,sizeof(double));
    cell_signal[ii]=my_calloc(3*nside,sizeof(double));
    cell_deproj[ii]=my_calloc(3*nside,sizeof(double));
    cell_out[ii]=my_malloc(bin->n_bands*sizeof(double));
  }
  //Read signal and noise power spectrum
  FILE *fi=my_fopen("test/benchmarks/cls_lss.txt","r");
  for(ii=0;ii<3*nside;ii++) {
    int jj;
    double dum;
    int stat=fscanf(fi,"%lf %lf %lf %lf %lf %lf %lf %lf %lf",&dum,
		    &dum,&(cell_signal[0][ii]),&(cell_signal[3][ii]),&dum,
		    &dum,&(cell_noise[0][ii]),&(cell_noise[3][ii]),&dum);
    ASSERT_EQUAL(stat,9);
    for(jj=0;jj<ncls;jj++) //Add noise to signal
      cell_signal[jj][ii]+=cell_noise[jj][ii];
  }
  fclose(fi);
  
  //No contaminants
  f2=nmt_field_alloc_sph(nside,msk,1,mps2,0,NULL,NULL,0,0,3,1E-10);
  w22=nmt_compute_coupling_matrix(f2,f2,bin,0);
  nmt_compute_coupled_cell(f2,f2,cell);
  nmt_decouple_cl_l(w22,cell,cell_noise,cell_deproj,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_nc_np_c22.txt",1E-3);
  nmt_workspace_free(w22);
  nmt_field_free(f2);

  //With purification
  f2=nmt_field_alloc_sph(nside,msk,1,mps2,0,NULL,NULL,0,1,3,1E-10);
  w22=nmt_compute_coupling_matrix(f2,f2,bin,0);
  nmt_compute_coupled_cell(f2,f2,cell);
  nmt_decouple_cl_l(w22,cell,cell_noise,cell_deproj,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_nc_yp_c22.txt",1E-3);
  nmt_workspace_free(w22);
  nmt_field_free(f2);

  //With contaminants
  f2=nmt_field_alloc_sph(nside,msk,1,mps2,1,&tmp2,NULL,0,0,3,1E-10);
  w22=nmt_compute_coupling_matrix(f2,f2,bin,0);
  nmt_compute_coupled_cell(f2,f2,cell);
  nmt_compute_deprojection_bias(f2,f2,cell_signal,cell_deproj);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(3*nside,cell_deproj[ii],ncls,ii,"test/benchmarks/bm_yc_np_cb22.txt",1E-3);
  nmt_decouple_cl_l(w22,cell,cell_noise,cell_deproj,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_yc_np_c22.txt",1E-3);
  nmt_workspace_free(w22);
  nmt_field_free(f2);

  //With contaminants, with purification
  f2=nmt_field_alloc_sph(nside,msk,1,mps2,1,&tmp2,NULL,0,1,3,1E-10);
  w22=nmt_compute_coupling_matrix(f2,f2,bin,0);
  nmt_compute_coupled_cell(f2,f2,cell);
  nmt_compute_deprojection_bias(f2,f2,cell_signal,cell_deproj);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(3*nside,cell_deproj[ii],ncls,ii,"test/benchmarks/bm_yc_yp_cb22.txt",1E-3);
  nmt_decouple_cl_l(w22,cell,cell_noise,cell_deproj,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_yc_yp_c22.txt",1E-3);
  nmt_workspace_free(w22);
  nmt_field_free(f2);
  
  //Free up power spectra
  for(ii=0;ii<ncls;ii++) {
    free(cell[ii]);
    free(cell_noise[ii]);
    free(cell_signal[ii]);
    free(cell_deproj[ii]);
    free(cell_out[ii]);
  }
  free(cell); free(cell_noise); free(cell_signal); free(cell_deproj); free(cell_out);

  nmt_bins_free(bin);
  for(ii=0;ii<2;ii++) {
    free(tmp2[ii]);
    free(mps2[ii]);
  }
  free(mps2);
  free(tmp2);
  free(msk);
}

CTEST(nmt,master_02_full) {
  //Generate fields and compute coupling matrix
  int ii;
  long nside,ipix;
  nmt_field *f0,*f2;
  nmt_workspace *w02;
  double **mps0=my_malloc(sizeof(double *));
  double **mps2=my_malloc(2*sizeof(double *));
  double **tmp0=my_malloc(sizeof(double *));
  double **tmp2=my_malloc(2*sizeof(double *));
  double *msk=he_read_healpix_map("test/benchmarks/msk.fits",&nside,0);
  nmt_binning_scheme *bin=nmt_bins_constant(16,3*nside-1);
  mps0[0]=he_read_healpix_map("test/benchmarks/mps.fits",&nside,0);
  mps2[0]=he_read_healpix_map("test/benchmarks/mps.fits",&nside,1);
  mps2[1]=he_read_healpix_map("test/benchmarks/mps.fits",&nside,2);
  tmp0[0]=he_read_healpix_map("test/benchmarks/tmp.fits",&nside,0);
  tmp2[0]=he_read_healpix_map("test/benchmarks/tmp.fits",&nside,1);
  tmp2[1]=he_read_healpix_map("test/benchmarks/tmp.fits",&nside,2);
  
  //Init power spectra
  int ncls=2;
  double **cell=my_malloc(ncls*sizeof(double *));
  double **cell_out=my_malloc(ncls*sizeof(double *));
  double **cell_signal=my_malloc(ncls*sizeof(double *));
  double **cell_noise=my_malloc(ncls*sizeof(double *));
  double **cell_deproj=my_malloc(ncls*sizeof(double *));
  for(ii=0;ii<ncls;ii++) {
    cell[ii]=my_malloc(3*nside*sizeof(double));
    cell_noise[ii]=my_calloc(3*nside,sizeof(double));
    cell_signal[ii]=my_calloc(3*nside,sizeof(double));
    cell_deproj[ii]=my_calloc(3*nside,sizeof(double));
    cell_out[ii]=my_malloc(bin->n_bands*sizeof(double));
  }
  //Read signal and noise power spectrum
  FILE *fi=my_fopen("test/benchmarks/cls_lss.txt","r");
  for(ii=0;ii<3*nside;ii++) {
    int jj;
    double dum;
    int stat=fscanf(fi,"%lf %lf %lf %lf %lf %lf %lf %lf %lf",&dum,
		    &dum,&dum,&dum,&(cell_signal[0][ii]),
		    &dum,&dum,&dum,&(cell_noise[0][ii]));
    ASSERT_EQUAL(stat,9);
    for(jj=0;jj<ncls;jj++) //Add noise to signal
      cell_signal[jj][ii]+=cell_noise[jj][ii];
  }
  fclose(fi);
  
  //No contaminants
  f0=nmt_field_alloc_sph(nside,msk,0,mps0,0,NULL,NULL,0,0,3,1E-10);
  f2=nmt_field_alloc_sph(nside,msk,1,mps2,0,NULL,NULL,0,0,3,1E-10);
  w02=nmt_compute_coupling_matrix(f0,f2,bin,0);
  nmt_compute_coupled_cell(f0,f2,cell);
  nmt_decouple_cl_l(w02,cell,cell_noise,cell_deproj,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_nc_np_c02.txt",1E-3);
  nmt_workspace_free(w02);
  nmt_field_free(f0);
  nmt_field_free(f2);

  //With purification
  f0=nmt_field_alloc_sph(nside,msk,0,mps0,0,NULL,NULL,0,0,3,1E-10);
  f2=nmt_field_alloc_sph(nside,msk,1,mps2,0,NULL,NULL,0,1,3,1E-10);
  w02=nmt_compute_coupling_matrix(f0,f2,bin,0);
  nmt_compute_coupled_cell(f0,f2,cell);
  nmt_decouple_cl_l(w02,cell,cell_noise,cell_deproj,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_nc_yp_c02.txt",1E-3);
  nmt_workspace_free(w02);
  nmt_field_free(f0);
  nmt_field_free(f2);

  //With contaminants
  f0=nmt_field_alloc_sph(nside,msk,0,mps0,1,&tmp0,NULL,0,0,3,1E-10);
  f2=nmt_field_alloc_sph(nside,msk,1,mps2,1,&tmp2,NULL,0,0,3,1E-10);
  w02=nmt_compute_coupling_matrix(f0,f2,bin,0);
  nmt_compute_coupled_cell(f0,f2,cell);
  nmt_compute_deprojection_bias(f0,f2,cell_signal,cell_deproj);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(3*nside,cell_deproj[ii],ncls,ii,"test/benchmarks/bm_yc_np_cb02.txt",1E-3);
  nmt_decouple_cl_l(w02,cell,cell_noise,cell_deproj,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_yc_np_c02.txt",1E-3);
  nmt_workspace_free(w02);
  nmt_field_free(f0);
  nmt_field_free(f2);

  //With contaminants, with purification
  f0=nmt_field_alloc_sph(nside,msk,0,mps0,1,&tmp0,NULL,0,1,3,1E-10);
  f2=nmt_field_alloc_sph(nside,msk,1,mps2,1,&tmp2,NULL,0,1,3,1E-10);
  w02=nmt_compute_coupling_matrix(f0,f2,bin,0);
  nmt_compute_coupled_cell(f0,f2,cell);
  nmt_compute_deprojection_bias(f0,f2,cell_signal,cell_deproj);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(3*nside,cell_deproj[ii],ncls,ii,"test/benchmarks/bm_yc_yp_cb02.txt",1E-3);
  nmt_decouple_cl_l(w02,cell,cell_noise,cell_deproj,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_yc_yp_c02.txt",1E-3);
  nmt_workspace_free(w02);
  nmt_field_free(f0);
  nmt_field_free(f2);

  //Free up power spectra
  for(ii=0;ii<ncls;ii++) {
    free(cell[ii]);
    free(cell_noise[ii]);
    free(cell_signal[ii]);
    free(cell_deproj[ii]);
    free(cell_out[ii]);
  }
  free(cell); free(cell_noise); free(cell_signal); free(cell_deproj); free(cell_out);

  nmt_bins_free(bin);
  for(ii=0;ii<2;ii++) {
    free(tmp2[ii]);
    free(mps2[ii]);
  }
  free(tmp0[0]);
  free(mps0[0]);
  free(mps0);
  free(mps2);
  free(tmp0);
  free(tmp2);
  free(msk);
}

CTEST(nmt,master_00_full) {
  //Generate fields and compute coupling matrix
  int ii;
  long nside,ipix;
  nmt_field *f0;
  nmt_workspace *w00;
  double *mps=he_read_healpix_map("test/benchmarks/mps.fits",&nside,0);
  double **tmp=my_malloc(sizeof(double *));
  double *msk=he_read_healpix_map("test/benchmarks/msk.fits",&nside,0);
  nmt_binning_scheme *bin=nmt_bins_constant(16,3*nside-1);
  tmp[0]=he_read_healpix_map("test/benchmarks/tmp.fits",&nside,0);
  
  //Init power spectra
  int ncls=1;
  double **cell=my_malloc(ncls*sizeof(double *));
  double **cell_out=my_malloc(ncls*sizeof(double *));
  double **cell_signal=my_malloc(ncls*sizeof(double *));
  double **cell_noise=my_malloc(ncls*sizeof(double *));
  double **cell_deproj=my_malloc(ncls*sizeof(double *));
  for(ii=0;ii<ncls;ii++) {
    cell[ii]=my_malloc(3*nside*sizeof(double));
    cell_noise[ii]=my_calloc(3*nside,sizeof(double));
    cell_signal[ii]=my_calloc(3*nside,sizeof(double));
    cell_deproj[ii]=my_calloc(3*nside,sizeof(double));
    cell_out[ii]=my_malloc(bin->n_bands*sizeof(double));
  }
  //Read signal and noise power spectrum
  FILE *fi=my_fopen("test/benchmarks/cls_lss.txt","r");
  for(ii=0;ii<3*nside;ii++) {
    int jj;
    double dum;
    int stat=fscanf(fi,"%lf %lf %lf %lf %lf %lf %lf %lf %lf",&dum,
		    &(cell_signal[0][ii]),&dum,&dum,&dum,
		    &(cell_noise[0][ii]),&dum,&dum,&dum);
    ASSERT_EQUAL(stat,9);
    for(jj=0;jj<ncls;jj++) //Add noise to signal
      cell_signal[jj][ii]+=cell_noise[jj][ii];
  }
  fclose(fi);
  
  //No contaminants
  f0=nmt_field_alloc_sph(nside,msk,0,&mps,0,NULL,NULL,0,0,3,1E-10);
  w00=nmt_compute_coupling_matrix(f0,f0,bin,0);
  nmt_compute_coupled_cell(f0,f0,cell);
  nmt_decouple_cl_l(w00,cell,cell_noise,cell_deproj,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_nc_np_c00.txt",1E-3);
  nmt_workspace_free(w00);
  nmt_field_free(f0);

  //With contaminants
  f0=nmt_field_alloc_sph(nside,msk,0,&mps,1,&tmp,NULL,0,0,3,1E-10);
  w00=nmt_compute_coupling_matrix(f0,f0,bin,0);
  nmt_compute_coupled_cell(f0,f0,cell);
  nmt_compute_deprojection_bias(f0,f0,cell_signal,cell_deproj);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(3*nside,cell_deproj[ii],ncls,ii,"test/benchmarks/bm_yc_np_cb00.txt",1E-3);
  nmt_decouple_cl_l(w00,cell,cell_noise,cell_deproj,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_yc_np_c00.txt",1E-3);
  nmt_workspace_free(w00);
  nmt_field_free(f0);

  //Free up power spectra
  for(ii=0;ii<ncls;ii++) {
    free(cell[ii]);
    free(cell_noise[ii]);
    free(cell_signal[ii]);
    free(cell_deproj[ii]);
    free(cell_out[ii]);
  }
  free(cell); free(cell_noise); free(cell_signal); free(cell_deproj); free(cell_out);

  nmt_bins_free(bin);
  free(tmp[0]);
  free(mps);
  free(tmp);
  free(msk);
}

CTEST(nmt,master_errors) {
  long nside;
  double *mpt=he_read_healpix_map("test/benchmarks/mps.fits",&nside,0);
  double *msk=he_read_healpix_map("test/benchmarks/msk.fits",&nside,0);
  nmt_binning_scheme *bin;
  nmt_workspace *w=NULL,*wb=NULL;

  nmt_field *f0=nmt_field_alloc_sph(nside,msk,0,&mpt,0,NULL,NULL,0,0,3,1E-10);
  nmt_field *f0b=nmt_field_alloc_sph(nside/2,msk,0,&mpt,0,NULL,NULL,0,0,3,1E-10);
  
  set_error_policy(THROW_ON_ERROR);

  //Read from non-existent file
  try { w=nmt_workspace_read("nofile"); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(w);
  //Wrong bins
  bin=nmt_bins_constant(20,6*nside-1);
  try { w=nmt_compute_coupling_matrix(f0,f0,bin,0); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(w);
  nmt_bins_free(bin);
  //Mismatching resolutions
  bin=nmt_bins_constant(20,3*nside-1);
  try { w=nmt_compute_coupling_matrix(f0,f0b,bin,0); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(w);
  //Wrong fields for teb
  try { w=nmt_compute_coupling_matrix(f0,f0,bin,1); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(w);
  //Try through nmt_compute_power_spectra
  try { w=nmt_compute_power_spectra(f0,f0b,bin,NULL,NULL,NULL,NULL); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(w);
  //nmt_compute_power_spectra with mis-matching input workspace
  w=nmt_compute_coupling_matrix(f0,f0,bin,0);
  nmt_bins_free(bin);
  bin=nmt_bins_constant(20,3*nside/2-1);
  try { wb=nmt_compute_power_spectra(f0b,f0b,bin,w,NULL,NULL,NULL); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(wb);
  //nmt_update_coupling_matrix with wrong input
  double *mat=calloc(3*nside*3*nside,sizeof(double));
  try { nmt_update_coupling_matrix(w,3*nside*2,mat); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  //nmt_update_coupling_matrix works as expected
  mat[3*nside*2+nside+1]=1.2345;
  nmt_update_coupling_matrix(w,3*nside,mat);
  ASSERT_DBL_NEAR_TOL(w->coupling_matrix_unbinned[2][nside+1],1.2345,1E-10);
  free(mat);
  nmt_workspace_free(w);
  set_error_policy(EXIT_ON_ERROR);

  nmt_field_free(f0);
  nmt_field_free(f0b);
  nmt_bins_free(bin);
  free(mpt);
  free(msk);
}
