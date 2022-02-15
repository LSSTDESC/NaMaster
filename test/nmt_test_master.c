#include "namaster.h"
#include "ctest.h"
#include "utils.h"
#include "nmt_test_utils.h"

CTEST(nmt,master_bias_uncorr) {
  //Generate fields and compute coupling matrix
  int ii,im1,ll;
  double prefac,f_fac;
  long ipix,npix;
  nmt_field *f0,*f2;
  nmt_curvedsky_info *cs=nmt_curvedsky_info_alloc(1,1,-1,-1,-1,-1,-1,-1,-1);
  double **mps0=my_malloc(sizeof(double *));
  double **mps2=my_malloc(2*sizeof(double *));
  double **tmp0=my_malloc(sizeof(double *));
  double **tmp2=my_malloc(2*sizeof(double *));
  double **mp_dum=my_malloc(2*sizeof(double *));
  double *msk=he_read_map("test/benchmarks/msk.fits",cs,0);
  mps0[0]=he_read_map("test/benchmarks/mps.fits",cs,0);
  mps2[0]=he_read_map("test/benchmarks/mps.fits",cs,1);
  mps2[1]=he_read_map("test/benchmarks/mps.fits",cs,2);
  tmp0[0]=he_read_map("test/benchmarks/tmp.fits",cs,0);
  tmp2[0]=he_read_map("test/benchmarks/tmp.fits",cs,1);
  tmp2[1]=he_read_map("test/benchmarks/tmp.fits",cs,2);
  npix=he_nside2npix(cs->n_eq);
  for(ii=0;ii<2;ii++)
    mp_dum[ii]=my_malloc(npix*sizeof(double));
  
  //Init power spectra
  int ncls=4;
  long lmax=he_get_lmax(cs);
  double **cell=my_malloc(ncls*sizeof(double *));
  double **cell_tmp=my_malloc(ncls*sizeof(double *));
  double **cell_tmpb=my_malloc(ncls*sizeof(double *));
  for(ii=0;ii<ncls;ii++) {
    cell[ii]=my_malloc((lmax+1)*sizeof(double));
    cell_tmp[ii]=my_malloc((lmax+1)*sizeof(double));
    cell_tmpb[ii]=my_malloc((lmax+1)*sizeof(double));
  }
  
  f0=nmt_field_alloc_sph(cs,msk,0,mps0,1,&tmp0,NULL,0,0,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  nmt_compute_uncorr_noise_deprojection_bias(f0,msk,cell,HE_NITER_DEFAULT);
  free(mps0[0]); mps0[0]=he_read_map("test/benchmarks/mps.fits",cs,0);
  free(tmp0[0]); tmp0[0]=he_read_map("test/benchmarks/tmp.fits",cs,0);
  he_anafast(tmp0,tmp0,0,0,cell_tmp,cs,lmax,3);
  f_fac=0;
  prefac=0;
  for(im1=0;im1<1;im1++) {
    he_map_product(cs,tmp0[im1],msk,tmp0[im1]); //Multiply by mask
    he_map_product(cs,tmp0[im1],msk,mp_dum[im1]); //f*v
    he_map_product(cs,mp_dum[im1],msk,mp_dum[im1]); //f*v^2
    he_map_product(cs,mp_dum[im1],msk,mp_dum[im1]); //f*v^2*s^2
    prefac+=he_map_dot(cs,mp_dum[im1],tmp0[im1]); //Sum[f*f*v^2*s^2 * dOmega]
    f_fac+=he_map_dot(cs,tmp0[im1],tmp0[im1]); //Sum[f*f * dOmega]
  }
  f_fac=1./f_fac; //1./Int[f^2]
  he_anafast(tmp0,tmp0,0,0,cell_tmp,cs,lmax,3);
  he_anafast(mp_dum,tmp0,0,0,cell_tmpb,cs,lmax,3);

  for(ii=0;ii<1;ii++) {
    for(ll=0;ll<=lmax;ll++) {
      double cl_pred=-2*f_fac*cell_tmpb[ii][ll]+f_fac*f_fac*prefac*cell_tmp[ii][ll];
      ASSERT_DBL_NEAR_TOL(cl_pred,cell[ii][ll],1E-3*fabs(cl_pred));
    }
  }
  
  f2=nmt_field_alloc_sph(cs,msk,2,mps2,1,&tmp2,NULL,0,0,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  nmt_compute_uncorr_noise_deprojection_bias(f2,msk,cell,HE_NITER_DEFAULT);
  free(mps2[0]); mps2[0]=he_read_map("test/benchmarks/mps.fits",cs,1);
  free(mps2[1]); mps2[1]=he_read_map("test/benchmarks/mps.fits",cs,2);
  free(tmp2[0]); tmp2[0]=he_read_map("test/benchmarks/tmp.fits",cs,1);
  free(tmp2[1]); tmp2[1]=he_read_map("test/benchmarks/tmp.fits",cs,2);
  f_fac=0;
  prefac=0;
  for(im1=0;im1<2;im1++) {
    he_map_product(cs,tmp2[im1],msk,tmp2[im1]); //Multiply by mask
    he_map_product(cs,tmp2[im1],msk,mp_dum[im1]); //f*v
    he_map_product(cs,mp_dum[im1],msk,mp_dum[im1]); //f*v^2
    he_map_product(cs,mp_dum[im1],msk,mp_dum[im1]); //f*v^2*s^2
    prefac+=he_map_dot(cs,mp_dum[im1],tmp2[im1]); //Sum[f*f*v^2*s^2 * dOmega]
    f_fac+=he_map_dot(cs,tmp2[im1],tmp2[im1]); //Sum[f*f * dOmega]
  }
  f_fac=1./f_fac; //1./Int[f^2]
  he_anafast(tmp2,tmp2,2,2,cell_tmp,cs,lmax,3);
  he_anafast(mp_dum,tmp2,2,2,cell_tmpb,cs,lmax,3);

  for(ii=0;ii<4;ii++) {
    for(ll=0;ll<=lmax;ll++) {
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
  free(cs);
}

CTEST(nmt,master_teb_full) {
  //Checks that the TEB workspaces get the same thing as the 00, 02 and 22 workspaces put together.
  //Generate fields and compute coupling matrix
  int ii;
  long ipix;
  nmt_field *f0,*f2;
  nmt_workspace *w_teb,*w00,*w02,*w22;
  nmt_curvedsky_info *cs=nmt_curvedsky_info_alloc(1,1,-1,-1,-1,-1,-1,-1,-1);
  double **mps0=my_malloc(sizeof(double *));
  double **mps2=my_malloc(2*sizeof(double *));
  double *msk=he_read_map("test/benchmarks/msk.fits",cs,0);
  long lmax=he_get_lmax(cs);
  nmt_binning_scheme *bin=nmt_bins_constant(16,lmax,0);
  mps0[0]=he_read_map("test/benchmarks/mps.fits",cs,0);
  mps2[0]=he_read_map("test/benchmarks/mps.fits",cs,1);
  mps2[1]=he_read_map("test/benchmarks/mps.fits",cs,2);
  
  //Init power spectra
  int ncls=7;
  double **cell=my_malloc(ncls*sizeof(double *));
  double **cell_out=my_malloc(ncls*sizeof(double *));
  double **cell_out_teb=my_malloc(ncls*sizeof(double *));
  double **cell_signal=my_malloc(ncls*sizeof(double *));
  double **cell_noise=my_malloc(ncls*sizeof(double *));
  double **cell_deproj=my_malloc(ncls*sizeof(double *));
  for(ii=0;ii<ncls;ii++) {
    cell[ii]=my_malloc((lmax+1)*sizeof(double));
    cell_noise[ii]=my_calloc((lmax+1),sizeof(double));
    cell_signal[ii]=my_calloc((lmax+1),sizeof(double));
    cell_deproj[ii]=my_calloc((lmax+1),sizeof(double));
    cell_out[ii]=my_malloc(bin->n_bands*sizeof(double));
    cell_out_teb[ii]=my_malloc(bin->n_bands*sizeof(double));
  }
  //Read signal and noise power spectrum
  FILE *fi=my_fopen("test/benchmarks/cls_lss.txt","r");
  for(ii=0;ii<=lmax;ii++) {
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
  f0=nmt_field_alloc_sph(cs,msk,0,mps0,0,NULL,NULL,0,0,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  f2=nmt_field_alloc_sph(cs,msk,2,mps2,0,NULL,NULL,0,0,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  w_teb=nmt_compute_coupling_matrix(f0,f2,bin,1,HE_NITER_DEFAULT,-1,-1,-1,-1);
  w00=nmt_compute_coupling_matrix(f0,f0,bin,0,HE_NITER_DEFAULT,-1,-1,-1,-1);
  w02=nmt_compute_coupling_matrix(f0,f2,bin,0,HE_NITER_DEFAULT,-1,-1,-1,-1);
  w22=nmt_compute_coupling_matrix(f2,f2,bin,0,HE_NITER_DEFAULT,-1,-1,-1,-1);
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
  free(cs);
}

CTEST(nmt,master_22_full) {
  //Generate fields and compute coupling matrix
  int ii;
  long ipix;
  nmt_field *f2;
  nmt_workspace *w22;
  nmt_curvedsky_info *cs=nmt_curvedsky_info_alloc(1,1,-1,-1,-1,-1,-1,-1,-1);
  double **mps2=my_malloc(2*sizeof(double *));
  double **tmp2=my_malloc(2*sizeof(double *));
  double *msk=he_read_map("test/benchmarks/msk.fits",cs,0);
  long lmax=he_get_lmax(cs);
  nmt_binning_scheme *bin=nmt_bins_constant(16,lmax,0);
  
  //Init power spectra
  int ncls=4;
  double **cell=my_malloc(ncls*sizeof(double *));
  double **cell_out=my_malloc(ncls*sizeof(double *));
  double **cell_signal=my_malloc(ncls*sizeof(double *));
  double **cell_noise=my_malloc(ncls*sizeof(double *));
  double **cell_deproj=my_malloc(ncls*sizeof(double *));
  for(ii=0;ii<ncls;ii++) {
    cell[ii]=my_malloc((lmax+1)*sizeof(double));
    cell_noise[ii]=my_calloc((lmax+1),sizeof(double));
    cell_signal[ii]=my_calloc((lmax+1),sizeof(double));
    cell_deproj[ii]=my_calloc((lmax+1),sizeof(double));
    cell_out[ii]=my_malloc(bin->n_bands*sizeof(double));
  }
  //Read signal and noise power spectrum
  FILE *fi=my_fopen("test/benchmarks/cls_lss.txt","r");
  for(ii=0;ii<=lmax;ii++) {
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
  mps2[0]=he_read_map("test/benchmarks/mps.fits",cs,1);
  mps2[1]=he_read_map("test/benchmarks/mps.fits",cs,2);
  f2=nmt_field_alloc_sph(cs,msk,2,mps2,0,NULL,NULL,0,0,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  w22=nmt_compute_coupling_matrix(f2,f2,bin,0,HE_NITER_DEFAULT,-1,-1,-1,-1);
  nmt_compute_coupled_cell(f2,f2,cell);
  nmt_decouple_cl_l(w22,cell,cell_noise,cell_deproj,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_nc_np_c22.txt",1E-3);
  nmt_workspace_free(w22);
  nmt_field_free(f2);
  free(mps2[0]); free(mps2[1]);

  //With purification
  mps2[0]=he_read_map("test/benchmarks/mps.fits",cs,1);
  mps2[1]=he_read_map("test/benchmarks/mps.fits",cs,2);
  f2=nmt_field_alloc_sph(cs,msk,2,mps2,0,NULL,NULL,0,1,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  w22=nmt_compute_coupling_matrix(f2,f2,bin,0,HE_NITER_DEFAULT,-1,-1,-1,-1);
  nmt_compute_coupled_cell(f2,f2,cell);
  nmt_decouple_cl_l(w22,cell,cell_noise,cell_deproj,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_nc_yp_c22.txt",1E-3);
  nmt_workspace_free(w22);
  nmt_field_free(f2);
  free(mps2[0]); free(mps2[1]);
  
  //With contaminants
  mps2[0]=he_read_map("test/benchmarks/mps.fits",cs,1);
  mps2[1]=he_read_map("test/benchmarks/mps.fits",cs,2);
  tmp2[0]=he_read_map("test/benchmarks/tmp.fits",cs,1);
  tmp2[1]=he_read_map("test/benchmarks/tmp.fits",cs,2);
  f2=nmt_field_alloc_sph(cs,msk,2,mps2,1,&tmp2,NULL,0,0,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  w22=nmt_compute_coupling_matrix(f2,f2,bin,0,HE_NITER_DEFAULT,-1,-1,-1,-1);
  nmt_compute_coupled_cell(f2,f2,cell);
  nmt_compute_deprojection_bias(f2,f2,cell_signal,cell_deproj,HE_NITER_DEFAULT);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays((lmax+1),cell_deproj[ii],ncls,ii,"test/benchmarks/bm_yc_np_cb22.txt",1E-3);
  nmt_decouple_cl_l(w22,cell,cell_noise,cell_deproj,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_yc_np_c22.txt",1E-3);
  nmt_workspace_free(w22);
  nmt_field_free(f2);
  free(mps2[0]); free(mps2[1]);
  free(tmp2[0]); free(tmp2[1]);

  //With contaminants, with purification
  mps2[0]=he_read_map("test/benchmarks/mps.fits",cs,1);
  mps2[1]=he_read_map("test/benchmarks/mps.fits",cs,2);
  tmp2[0]=he_read_map("test/benchmarks/tmp.fits",cs,1);
  tmp2[1]=he_read_map("test/benchmarks/tmp.fits",cs,2);
  f2=nmt_field_alloc_sph(cs,msk,2,mps2,1,&tmp2,NULL,0,1,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  w22=nmt_compute_coupling_matrix(f2,f2,bin,0,HE_NITER_DEFAULT,-1,-1,-1,-1);
  nmt_compute_coupled_cell(f2,f2,cell);
  nmt_compute_deprojection_bias(f2,f2,cell_signal,cell_deproj,HE_NITER_DEFAULT);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays((lmax+1),cell_deproj[ii],ncls,ii,"test/benchmarks/bm_yc_yp_cb22.txt",1E-3);
  nmt_decouple_cl_l(w22,cell,cell_noise,cell_deproj,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_yc_yp_c22.txt",1E-3);
  nmt_workspace_free(w22);
  nmt_field_free(f2);
  free(mps2[0]); free(mps2[1]);
  free(tmp2[0]); free(tmp2[1]);
 
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
  free(mps2);
  free(tmp2);
  free(msk);
  free(cs);
}

CTEST(nmt,master_11_full) {
  //Generate fields and compute coupling matrix
  int ii;
  long ipix;
  nmt_field *f1;
  nmt_workspace *w11;
  nmt_curvedsky_info *cs=nmt_curvedsky_info_alloc(1,1,-1,-1,-1,-1,-1,-1,-1);
  double **mps1=my_malloc(2*sizeof(double *));
  double *msk=he_read_map("test/benchmarks/msk.fits",cs,0);
  long lmax=he_get_lmax(cs);
  nmt_binning_scheme *bin=nmt_bins_constant(16,lmax,0);
  mps1[0]=he_read_map("test/benchmarks/mps_sp1.fits",cs,1);
  mps1[1]=he_read_map("test/benchmarks/mps_sp1.fits",cs,2);
  
  //Init power spectra
  int ncls=4;
  double **cell=my_malloc(ncls*sizeof(double *));
  double **cell_out=my_malloc(ncls*sizeof(double *));
  double **cell0=my_malloc(ncls*sizeof(double *));
  for(ii=0;ii<ncls;ii++) {
    cell[ii]=my_malloc((lmax+1)*sizeof(double));
    cell0[ii]=my_calloc((lmax+1),sizeof(double));
    cell_out[ii]=my_malloc(bin->n_bands*sizeof(double));
  }
  
  //No contaminants
  f1=nmt_field_alloc_sph(cs,msk,1,mps1,0,NULL,NULL,0,0,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  w11=nmt_compute_coupling_matrix(f1,f1,bin,0,HE_NITER_DEFAULT,-1,-1,-1,-1);
  nmt_compute_coupled_cell(f1,f1,cell);
  nmt_decouple_cl_l(w11,cell,cell0,cell0,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_sp1_c11.txt",1E-3);
  nmt_workspace_free(w11);
  nmt_field_free(f1);
  
  //Free up power spectra
  for(ii=0;ii<ncls;ii++) {
    free(cell[ii]);
    free(cell0[ii]);
    free(cell_out[ii]);
  }
  free(cell); free(cell0); free(cell_out);

  nmt_bins_free(bin);
  for(ii=0;ii<2;ii++)
    free(mps1[ii]);
  free(mps1);
  free(msk);
  free(cs);
}

CTEST(nmt,master_lite) {
  //Generate fields and compute coupling matrix
  int ii;
  long ipix;
  nmt_field *f0,*f2,*f2l,*f2e;
  nmt_workspace *w02;
  nmt_curvedsky_info *cs=nmt_curvedsky_info_alloc(1,1,-1,-1,-1,-1,-1,-1,-1);
  double **mps0=my_malloc(sizeof(double *));
  double **mps2=my_malloc(2*sizeof(double *));
  double **tmp0=my_malloc(sizeof(double *));
  double **tmp2=my_malloc(2*sizeof(double *));
  double *msk=he_read_map("test/benchmarks/msk.fits",cs,0);
  long lmax=he_get_lmax(cs);
  nmt_binning_scheme *bin=nmt_bins_constant(16,lmax,0);
  
  //Init power spectra
  int ncls=2;
  double **cell=my_malloc(ncls*sizeof(double *));
  double **cell_out=my_malloc(ncls*sizeof(double *));
  double **cell_signal=my_malloc(ncls*sizeof(double *));
  double **cell_noise=my_malloc(ncls*sizeof(double *));
  double **cell_deproj=my_malloc(ncls*sizeof(double *));
  for(ii=0;ii<ncls;ii++) {
    cell[ii]=my_malloc((lmax+1)*sizeof(double));
    cell_noise[ii]=my_calloc((lmax+1),sizeof(double));
    cell_signal[ii]=my_calloc((lmax+1),sizeof(double));
    cell_deproj[ii]=my_calloc((lmax+1),sizeof(double));
    cell_out[ii]=my_malloc(bin->n_bands*sizeof(double));
  }
  //Read signal and noise power spectrum
  FILE *fi=my_fopen("test/benchmarks/cls_lss.txt","r");
  for(ii=0;ii<=lmax;ii++) {
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

  //With contaminants, with purification
  mps0[0]=he_read_map("test/benchmarks/mps.fits",cs,0);
  mps2[0]=he_read_map("test/benchmarks/mps.fits",cs,1);
  mps2[1]=he_read_map("test/benchmarks/mps.fits",cs,2);
  tmp0[0]=he_read_map("test/benchmarks/tmp.fits",cs,0);
  tmp2[0]=he_read_map("test/benchmarks/tmp.fits",cs,1);
  tmp2[1]=he_read_map("test/benchmarks/tmp.fits",cs,2);
  f0=nmt_field_alloc_sph(cs,msk,0,mps0,1,&tmp0,NULL,0,0,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  f2=nmt_field_alloc_sph(cs,msk,2,mps2,1,&tmp2,NULL,0,1,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  // Lite field
  free(mps2[0]); free(mps2[1]);
  free(tmp2[0]); free(tmp2[1]);
  mps2[0]=he_read_map("test/benchmarks/mps.fits",cs,1);
  mps2[1]=he_read_map("test/benchmarks/mps.fits",cs,2);
  tmp2[0]=he_read_map("test/benchmarks/tmp.fits",cs,1);
  tmp2[1]=he_read_map("test/benchmarks/tmp.fits",cs,2);
  f2l=nmt_field_alloc_sph(cs,msk,2,mps2,1,&tmp2,NULL,0,1,3,1E-10,HE_NITER_DEFAULT,0,1,0);
  // Mask-only field
  f2e=nmt_field_alloc_sph(cs,msk,2,NULL,0,NULL,NULL,0,1,3,1E-10,HE_NITER_DEFAULT,0,1,1);
  w02=nmt_compute_coupling_matrix(f0,f2e,bin,0,HE_NITER_DEFAULT,-1,-1,-1,-1);
  nmt_compute_coupled_cell(f0,f2l,cell);
  nmt_compute_deprojection_bias(f0,f2,cell_signal,cell_deproj,HE_NITER_DEFAULT);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays((lmax+1),cell_deproj[ii],ncls,ii,"test/benchmarks/bm_yc_yp_cb02.txt",1E-3);
  nmt_decouple_cl_l(w02,cell,cell_noise,cell_deproj,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_yc_yp_c02.txt",1E-3);
  nmt_workspace_free(w02);
  nmt_field_free(f0);
  nmt_field_free(f2);
  nmt_field_free(f2l);
  nmt_field_free(f2e);
  free(mps0[0]); free(mps2[0]); free(mps2[1]);
  free(tmp0[0]); free(tmp2[0]); free(tmp2[1]);

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
  free(mps0);
  free(mps2);
  free(tmp0);
  free(tmp2);
  free(msk);
  free(cs);
}

CTEST(nmt,master_02_full) {
  //Generate fields and compute coupling matrix
  int ii;
  long ipix;
  nmt_field *f0,*f2;
  nmt_workspace *w02;
  nmt_curvedsky_info *cs=nmt_curvedsky_info_alloc(1,1,-1,-1,-1,-1,-1,-1,-1);
  double **mps0=my_malloc(sizeof(double *));
  double **mps2=my_malloc(2*sizeof(double *));
  double **tmp0=my_malloc(sizeof(double *));
  double **tmp2=my_malloc(2*sizeof(double *));
  double *msk=he_read_map("test/benchmarks/msk.fits",cs,0);
  long lmax=he_get_lmax(cs);
  nmt_binning_scheme *bin=nmt_bins_constant(16,lmax,0);
  
  //Init power spectra
  int ncls=2;
  double **cell=my_malloc(ncls*sizeof(double *));
  double **cell_out=my_malloc(ncls*sizeof(double *));
  double **cell_signal=my_malloc(ncls*sizeof(double *));
  double **cell_noise=my_malloc(ncls*sizeof(double *));
  double **cell_deproj=my_malloc(ncls*sizeof(double *));
  for(ii=0;ii<ncls;ii++) {
    cell[ii]=my_malloc((lmax+1)*sizeof(double));
    cell_noise[ii]=my_calloc((lmax+1),sizeof(double));
    cell_signal[ii]=my_calloc((lmax+1),sizeof(double));
    cell_deproj[ii]=my_calloc((lmax+1),sizeof(double));
    cell_out[ii]=my_malloc(bin->n_bands*sizeof(double));
  }
  //Read signal and noise power spectrum
  FILE *fi=my_fopen("test/benchmarks/cls_lss.txt","r");
  for(ii=0;ii<=lmax;ii++) {
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
  mps0[0]=he_read_map("test/benchmarks/mps.fits",cs,0);
  mps2[0]=he_read_map("test/benchmarks/mps.fits",cs,1);
  mps2[1]=he_read_map("test/benchmarks/mps.fits",cs,2);
  f0=nmt_field_alloc_sph(cs,msk,0,mps0,0,NULL,NULL,0,0,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  f2=nmt_field_alloc_sph(cs,msk,2,mps2,0,NULL,NULL,0,0,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  w02=nmt_compute_coupling_matrix(f0,f2,bin,0,HE_NITER_DEFAULT,-1,-1,-1,-1);
  nmt_compute_coupled_cell(f0,f2,cell);
  nmt_decouple_cl_l(w02,cell,cell_noise,cell_deproj,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_nc_np_c02.txt",1E-3);
  nmt_workspace_free(w02);
  nmt_field_free(f0);
  nmt_field_free(f2);
  free(mps0[0]); free(mps2[0]); free(mps2[1]);

  //With purification
  mps0[0]=he_read_map("test/benchmarks/mps.fits",cs,0);
  mps2[0]=he_read_map("test/benchmarks/mps.fits",cs,1);
  mps2[1]=he_read_map("test/benchmarks/mps.fits",cs,2);
  f0=nmt_field_alloc_sph(cs,msk,0,mps0,0,NULL,NULL,0,0,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  f2=nmt_field_alloc_sph(cs,msk,2,mps2,0,NULL,NULL,0,1,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  w02=nmt_compute_coupling_matrix(f0,f2,bin,0,HE_NITER_DEFAULT,-1,-1,-1,-1);
  nmt_compute_coupled_cell(f0,f2,cell);
  nmt_decouple_cl_l(w02,cell,cell_noise,cell_deproj,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_nc_yp_c02.txt",1E-3);
  nmt_workspace_free(w02);
  nmt_field_free(f0);
  nmt_field_free(f2);
  free(mps0[0]); free(mps2[0]); free(mps2[1]);

  //With contaminants
  mps0[0]=he_read_map("test/benchmarks/mps.fits",cs,0);
  mps2[0]=he_read_map("test/benchmarks/mps.fits",cs,1);
  mps2[1]=he_read_map("test/benchmarks/mps.fits",cs,2);
  tmp0[0]=he_read_map("test/benchmarks/tmp.fits",cs,0);
  tmp2[0]=he_read_map("test/benchmarks/tmp.fits",cs,1);
  tmp2[1]=he_read_map("test/benchmarks/tmp.fits",cs,2);
  f0=nmt_field_alloc_sph(cs,msk,0,mps0,1,&tmp0,NULL,0,0,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  f2=nmt_field_alloc_sph(cs,msk,2,mps2,1,&tmp2,NULL,0,0,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  w02=nmt_compute_coupling_matrix(f0,f2,bin,0,HE_NITER_DEFAULT,-1,-1,-1,-1);
  nmt_compute_coupled_cell(f0,f2,cell);
  nmt_compute_deprojection_bias(f0,f2,cell_signal,cell_deproj,HE_NITER_DEFAULT);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays((lmax+1),cell_deproj[ii],ncls,ii,"test/benchmarks/bm_yc_np_cb02.txt",1E-3);
  nmt_decouple_cl_l(w02,cell,cell_noise,cell_deproj,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_yc_np_c02.txt",1E-3);
  nmt_workspace_free(w02);
  nmt_field_free(f0);
  nmt_field_free(f2);
  free(mps0[0]); free(mps2[0]); free(mps2[1]);
  free(tmp0[0]); free(tmp2[0]); free(tmp2[1]);

  //With contaminants, with purification
  mps0[0]=he_read_map("test/benchmarks/mps.fits",cs,0);
  mps2[0]=he_read_map("test/benchmarks/mps.fits",cs,1);
  mps2[1]=he_read_map("test/benchmarks/mps.fits",cs,2);
  tmp0[0]=he_read_map("test/benchmarks/tmp.fits",cs,0);
  tmp2[0]=he_read_map("test/benchmarks/tmp.fits",cs,1);
  tmp2[1]=he_read_map("test/benchmarks/tmp.fits",cs,2);
  f0=nmt_field_alloc_sph(cs,msk,0,mps0,1,&tmp0,NULL,0,0,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  f2=nmt_field_alloc_sph(cs,msk,2,mps2,1,&tmp2,NULL,0,1,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  w02=nmt_compute_coupling_matrix(f0,f2,bin,0,HE_NITER_DEFAULT,-1,-1,-1,-1);
  nmt_compute_coupled_cell(f0,f2,cell);
  nmt_compute_deprojection_bias(f0,f2,cell_signal,cell_deproj,HE_NITER_DEFAULT);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays((lmax+1),cell_deproj[ii],ncls,ii,"test/benchmarks/bm_yc_yp_cb02.txt",1E-3);
  nmt_decouple_cl_l(w02,cell,cell_noise,cell_deproj,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_yc_yp_c02.txt",1E-3);
  nmt_workspace_free(w02);
  nmt_field_free(f0);
  nmt_field_free(f2);
  free(mps0[0]); free(mps2[0]); free(mps2[1]);
  free(tmp0[0]); free(tmp2[0]); free(tmp2[1]);

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
  free(mps0);
  free(mps2);
  free(tmp0);
  free(tmp2);
  free(msk);
  free(cs);
}

CTEST(nmt,master_sp1_full) {
  //Generate fields and compute coupling matrix
  int ii;
  long ipix;
  nmt_field *f0,*f1;
  nmt_workspace *w00,*w01,*w11;
  nmt_curvedsky_info *cs=nmt_curvedsky_info_alloc(1,1,-1,-1,-1,-1,-1,-1,-1);
  double **mps0=my_malloc(sizeof(double *));
  double **mps1=my_malloc(2*sizeof(double *));
  double *msk=he_read_map("test/benchmarks/msk.fits",cs,0);
  long lmax=he_get_lmax(cs);
  nmt_binning_scheme *bin=nmt_bins_constant(16,lmax,0);
  mps0[0]=he_read_map("test/benchmarks/mps_sp1.fits",cs,0);
  mps1[0]=he_read_map("test/benchmarks/mps_sp1.fits",cs,1);
  mps1[1]=he_read_map("test/benchmarks/mps_sp1.fits",cs,2);
  
  //Init power spectra
  int ncls=4;
  double **cell00=my_malloc(ncls*sizeof(double *));
  double **cell01=my_malloc(ncls*sizeof(double *));
  double **cell11=my_malloc(ncls*sizeof(double *));
  double **cell_out00=my_malloc(ncls*sizeof(double *));
  double **cell_out01=my_malloc(ncls*sizeof(double *));
  double **cell_out11=my_malloc(ncls*sizeof(double *));
  double **cell0=my_malloc(ncls*sizeof(double *));
  for(ii=0;ii<ncls;ii++) {
    cell00[ii]=my_malloc((lmax+1)*sizeof(double));
    cell01[ii]=my_malloc((lmax+1)*sizeof(double));
    cell11[ii]=my_malloc((lmax+1)*sizeof(double));
    cell0[ii]=my_calloc((lmax+1),sizeof(double));
    cell_out00[ii]=my_malloc(bin->n_bands*sizeof(double));
    cell_out01[ii]=my_malloc(bin->n_bands*sizeof(double));
    cell_out11[ii]=my_malloc(bin->n_bands*sizeof(double));
  }
  
  //No contaminants
  f0=nmt_field_alloc_sph(cs,msk,0,mps0,0,NULL,NULL,0,0,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  f1=nmt_field_alloc_sph(cs,msk,1,mps1,0,NULL,NULL,0,0,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  w00=nmt_compute_coupling_matrix(f0,f0,bin,0,HE_NITER_DEFAULT,-1,-1,-1,-1);
  w01=nmt_compute_coupling_matrix(f0,f1,bin,0,HE_NITER_DEFAULT,-1,-1,-1,-1);
  w11=nmt_compute_coupling_matrix(f1,f1,bin,0,HE_NITER_DEFAULT,-1,-1,-1,-1);
  nmt_compute_coupled_cell(f0,f0,cell00);
  nmt_decouple_cl_l(w00,cell00,cell0,cell0,cell_out00);
  nmt_compute_coupled_cell(f0,f1,cell01);
  nmt_decouple_cl_l(w01,cell01,cell0,cell0,cell_out01);
  nmt_compute_coupled_cell(f1,f1,cell11);
  nmt_decouple_cl_l(w11,cell11,cell0,cell0,cell_out11);
  for(ii=0;ii<bin->n_bands;ii++) {
    double cl00=cell_out00[0][ii];
    double tol=0.05*cl00;
    ASSERT_DBL_NEAR_TOL(cl00,cell_out01[0][ii],tol);
    ASSERT_DBL_NEAR_TOL(cl00,cell_out11[0][ii],tol);
  }
  nmt_workspace_free(w00);
  nmt_workspace_free(w01);
  nmt_workspace_free(w11);
  nmt_field_free(f0);
  nmt_field_free(f1);

  //Free up power spectra
  for(ii=0;ii<ncls;ii++) {
    free(cell00[ii]);
    free(cell01[ii]);
    free(cell11[ii]);
    free(cell0[ii]);
    free(cell_out00[ii]);
    free(cell_out01[ii]);
    free(cell_out11[ii]);
  }
  free(cell00);
  free(cell01);
  free(cell11);
  free(cell0);
  free(cell_out00);
  free(cell_out01);
  free(cell_out11);

  nmt_bins_free(bin);
  for(ii=0;ii<2;ii++)
    free(mps1[ii]);
  free(mps0[0]);
  free(mps0);
  free(mps1);
  free(msk);
  free(cs);
}

CTEST(nmt,master_01_full) {
  //Generate fields and compute coupling matrix
  int ii;
  long ipix;
  nmt_field *f0,*f1;
  nmt_workspace *w01;
  nmt_curvedsky_info *cs=nmt_curvedsky_info_alloc(1,1,-1,-1,-1,-1,-1,-1,-1);
  double **mps0=my_malloc(sizeof(double *));
  double **mps1=my_malloc(2*sizeof(double *));
  double *msk=he_read_map("test/benchmarks/msk.fits",cs,0);
  long lmax=he_get_lmax(cs);
  nmt_binning_scheme *bin=nmt_bins_constant(16,lmax,0);
  mps0[0]=he_read_map("test/benchmarks/mps_sp1.fits",cs,0);
  mps1[0]=he_read_map("test/benchmarks/mps_sp1.fits",cs,1);
  mps1[1]=he_read_map("test/benchmarks/mps_sp1.fits",cs,2);
  
  //Init power spectra
  int ncls=2;
  double **cell=my_malloc(ncls*sizeof(double *));
  double **cell_out=my_malloc(ncls*sizeof(double *));
  double **cell0=my_malloc(ncls*sizeof(double *));
  for(ii=0;ii<ncls;ii++) {
    cell[ii]=my_malloc((lmax+1)*sizeof(double));
    cell0[ii]=my_calloc((lmax+1),sizeof(double));
    cell_out[ii]=my_malloc(bin->n_bands*sizeof(double));
  }
  
  //No contaminants
  f0=nmt_field_alloc_sph(cs,msk,0,mps0,0,NULL,NULL,0,0,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  f1=nmt_field_alloc_sph(cs,msk,1,mps1,0,NULL,NULL,0,0,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  w01=nmt_compute_coupling_matrix(f0,f1,bin,0,HE_NITER_DEFAULT,-1,-1,-1,-1);
  nmt_compute_coupled_cell(f0,f1,cell);
  nmt_decouple_cl_l(w01,cell,cell0,cell0,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_sp1_c01.txt",1E-3);
  nmt_workspace_free(w01);
  nmt_field_free(f0);
  nmt_field_free(f1);

  //Free up power spectra
  for(ii=0;ii<ncls;ii++) {
    free(cell[ii]);
    free(cell0[ii]);
    free(cell_out[ii]);
  }
  free(cell); free(cell0); free(cell_out);

  nmt_bins_free(bin);
  for(ii=0;ii<2;ii++)
    free(mps1[ii]);
  free(mps0[0]);
  free(mps0);
  free(mps1);
  free(msk);
  free(cs);
}

CTEST(nmt,master_sp1) {
  int ii;
  long ipix, lmax;
  nmt_field *f0,*f1;
  nmt_curvedsky_info *cs=nmt_curvedsky_info_alloc(1,1,-1,-1,-1,-1,-1,-1,-1);
  double *msk;
  double **mps0=my_malloc(sizeof(double *));
  double **mps1=my_malloc(2*sizeof(double *));
  mps0[0]=he_read_map("test/benchmarks/mps_sp1.fits",cs,0);
  mps1[0]=he_read_map("test/benchmarks/mps_sp1.fits",cs,1);
  mps1[1]=he_read_map("test/benchmarks/mps_sp1.fits",cs,2);
  msk=my_malloc(cs->npix*sizeof(double));
  for(ipix=0;ipix<cs->npix;ipix++)
    msk[ipix]=1;
  lmax=he_get_lmax(cs);

  //Init power spectra
  int ncls=4;
  double **cell00=my_malloc(ncls*sizeof(double *));
  double **cell01=my_malloc(ncls*sizeof(double *));
  double **cell11=my_malloc(ncls*sizeof(double *));
  for(ii=0;ii<ncls;ii++) {
    cell00[ii]=my_malloc((lmax+1)*sizeof(double));
    cell01[ii]=my_malloc((lmax+1)*sizeof(double));
    cell11[ii]=my_malloc((lmax+1)*sizeof(double));
  }

  //No contaminants
  f0=nmt_field_alloc_sph(cs,msk,0,mps0,0,NULL,NULL,0,0,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  f1=nmt_field_alloc_sph(cs,msk,1,mps1,0,NULL,NULL,0,0,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  nmt_compute_coupled_cell(f0,f0,cell00);
  nmt_compute_coupled_cell(f0,f1,cell01);
  nmt_compute_coupled_cell(f1,f1,cell11);
  for(ii=2;ii<2*cs->n_eq;ii++) {
    ASSERT_DBL_NEAR_TOL(cell00[0][ii], cell01[0][ii], cell00[0][ii]*2E-3);
    ASSERT_DBL_NEAR_TOL(cell00[0][ii], cell11[0][ii], cell00[0][ii]*3E-3);
    ASSERT_DBL_NEAR_TOL(0., cell01[1][ii], 1E-9);
    ASSERT_DBL_NEAR_TOL(0., cell11[1][ii], 1E-9);
    ASSERT_DBL_NEAR_TOL(0., cell11[2][ii], 1E-9);
    ASSERT_DBL_NEAR_TOL(0., cell11[3][ii], 1E-9);
  }
  nmt_field_free(f0);
  nmt_field_free(f1);
  for(ii=0;ii<ncls;ii++) {
    free(cell00[ii]);
    free(cell01[ii]);
    free(cell11[ii]);
  }
  free(cell00);
  free(cell01);
  free(cell11);
  free(mps0[0]);
  free(mps1[0]);
  free(mps1[1]);
  free(msk);
  free(mps0);
  free(mps1);
  free(cs);
}

CTEST(nmt,master_00_full) {
  //Generate fields and compute coupling matrix
  int ii;
  long ipix;
  nmt_field *f0;
  nmt_workspace *w00;
  nmt_curvedsky_info *cs=nmt_curvedsky_info_alloc(1,1,-1,-1,-1,-1,-1,-1,-1);
  double *mps=NULL;
  double **tmp=my_malloc(sizeof(double *));
  double *msk=he_read_map("test/benchmarks/msk.fits",cs,0);
  long lmax=he_get_lmax(cs);
  nmt_binning_scheme *bin=nmt_bins_constant(16,lmax,0);
  
  //Init power spectra
  int ncls=1;
  double **cell=my_malloc(ncls*sizeof(double *));
  double **cell_out=my_malloc(ncls*sizeof(double *));
  double **cell_signal=my_malloc(ncls*sizeof(double *));
  double **cell_noise=my_malloc(ncls*sizeof(double *));
  double **cell_deproj=my_malloc(ncls*sizeof(double *));
  for(ii=0;ii<ncls;ii++) {
    cell[ii]=my_malloc((lmax+1)*sizeof(double));
    cell_noise[ii]=my_calloc((lmax+1),sizeof(double));
    cell_signal[ii]=my_calloc((lmax+1),sizeof(double));
    cell_deproj[ii]=my_calloc((lmax+1),sizeof(double));
    cell_out[ii]=my_malloc(bin->n_bands*sizeof(double));
  }
  //Read signal and noise power spectrum
  FILE *fi=my_fopen("test/benchmarks/cls_lss.txt","r");
  for(ii=0;ii<=lmax;ii++) {
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
  mps=he_read_map("test/benchmarks/mps.fits",cs,0);
  f0=nmt_field_alloc_sph(cs,msk,0,&mps,0,NULL,NULL,0,0,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  w00=nmt_compute_coupling_matrix(f0,f0,bin,0,HE_NITER_DEFAULT,-1,-1,-1,-1);
  nmt_compute_coupled_cell(f0,f0,cell);
  nmt_decouple_cl_l(w00,cell,cell_noise,cell_deproj,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_nc_np_c00.txt",1E-3);
  nmt_workspace_free(w00);
  nmt_field_free(f0);
  free(mps);

  //With contaminants
  mps=he_read_map("test/benchmarks/mps.fits",cs,0);
  tmp[0]=he_read_map("test/benchmarks/tmp.fits",cs,0);
  f0=nmt_field_alloc_sph(cs,msk,0,&mps,1,&tmp,NULL,0,0,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  w00=nmt_compute_coupling_matrix(f0,f0,bin,0,HE_NITER_DEFAULT,-1,-1,-1,-1);
  nmt_compute_coupled_cell(f0,f0,cell);
  nmt_compute_deprojection_bias(f0,f0,cell_signal,cell_deproj,HE_NITER_DEFAULT);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays((lmax+1),cell_deproj[ii],ncls,ii,"test/benchmarks/bm_yc_np_cb00.txt",1E-3);
  nmt_decouple_cl_l(w00,cell,cell_noise,cell_deproj,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_yc_np_c00.txt",1E-3);
  nmt_workspace_free(w00);
  nmt_field_free(f0);
  free(mps); free(tmp[0]);

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
  free(tmp);
  free(msk);
  free(cs);
}

CTEST(nmt,master_00_f_ell) {
  //Generate fields and compute coupling matrix
  int ii;
  long ipix;
  nmt_field *f0;
  nmt_workspace *w00;
  nmt_curvedsky_info *cs=nmt_curvedsky_info_alloc(1,1,-1,-1,-1,-1,-1,-1,-1);
  double *mps=he_read_map("test/benchmarks/mps.fits",cs,0);
  double *msk=he_read_map("test/benchmarks/msk.fits",cs,0);
  long lmax=he_get_lmax(cs);
  nmt_binning_scheme *bin=nmt_bins_constant(16,lmax,1);
  double *ell_eff=my_malloc(bin->n_bands*sizeof(double));
  nmt_ell_eff(bin,ell_eff);
  
  //Init power spectra
  int ncls=1;
  double **cell=my_malloc(ncls*sizeof(double *));
  double **cell_out=my_malloc(ncls*sizeof(double *));
  double **cell_noise=my_malloc(ncls*sizeof(double *));
  double **cell_deproj=my_malloc(ncls*sizeof(double *));
  for(ii=0;ii<ncls;ii++) {
    cell[ii]=my_malloc((lmax+1)*sizeof(double));
    cell_noise[ii]=my_calloc((lmax+1),sizeof(double));
    cell_deproj[ii]=my_calloc((lmax+1),sizeof(double));
    cell_out[ii]=my_malloc(bin->n_bands*sizeof(double));
  }
  //Read signal and noise power spectrum
  FILE *fi=my_fopen("test/benchmarks/cls_lss.txt","r");
  for(ii=0;ii<=lmax;ii++) {
    int jj;
    double dum;
    int stat=fscanf(fi,"%lf %lf %lf %lf %lf %lf %lf %lf %lf",&dum,
  		    &dum,&dum,&dum,&dum,
  		    &(cell_noise[0][ii]),&dum,&dum,&dum);
    ASSERT_EQUAL(stat,9);
  }
  fclose(fi);
  
  //No contaminants
  f0=nmt_field_alloc_sph(cs,msk,0,&mps,0,NULL,NULL,0,0,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  w00=nmt_compute_coupling_matrix(f0,f0,bin,0,HE_NITER_DEFAULT,-1,-1,-1,-1);
  nmt_compute_coupled_cell(f0,f0,cell);
  nmt_decouple_cl_l(w00,cell,cell_noise,cell_deproj,cell_out);
  for(ii=0;ii<bin->n_bands;ii++) { //Rough correction for ell-dependent prefactor
    double ell=ell_eff[ii]+1;
    cell_out[0][ii]*=(2*M_PI)/(ell*(ell+1));
  }
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_nc_np_c00.txt",1E-1);
  nmt_workspace_free(w00);
  nmt_field_free(f0);

  //Free up power spectra
  for(ii=0;ii<ncls;ii++) {
    free(cell[ii]);
    free(cell_noise[ii]);
    free(cell_deproj[ii]);
    free(cell_out[ii]);
  }
  free(cell); free(cell_noise); free(cell_deproj); free(cell_out);

  nmt_bins_free(bin);
  free(mps);
  free(msk);
  free(cs);
  free(ell_eff);
}

CTEST(nmt,bandpower_windows) {
  nmt_curvedsky_info *cs=nmt_curvedsky_info_alloc(1,1,-1,-1,-1,-1,-1,-1,-1);
  double *mpt=he_read_map("test/benchmarks/mps.fits",cs,0);
  double *msk=he_read_map("test/benchmarks/msk.fits",cs,0);
  long lmax=he_get_lmax(cs);
  nmt_binning_scheme *bin=nmt_bins_constant(20,lmax,0);
  nmt_field *f0=nmt_field_alloc_sph(cs,msk,0,&mpt,0,NULL,NULL,0,0,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  nmt_workspace *w=nmt_compute_coupling_matrix(f0,f0,bin,0,HE_NITER_DEFAULT,-1,-1,-1,-1);

  double *cls_in=malloc((lmax+1)*sizeof(double));
  double *cls_zero=calloc(lmax+1,sizeof(double));
  double *cls_coupled=malloc((lmax+1)*sizeof(double));
  double *cls_out_a=malloc(bin->n_bands*sizeof(double));
  double *cls_out_b=malloc(bin->n_bands*sizeof(double));
  double *bpw_windows=malloc(bin->n_bands*(lmax+1)*sizeof(double));

  // Input power spectra
  int ii;
  for(ii=0;ii<=lmax;ii++)
    cls_in[ii]=pow((ii+1.),-0.8);

  //Couple-decouple
  nmt_couple_cl_l(w,&cls_in,&cls_coupled);
  nmt_decouple_cl_l(w,&cls_coupled,&cls_zero,&cls_zero,&cls_out_a);

  nmt_compute_bandpower_windows(w,bpw_windows);

  for(ii=0;ii<bin->n_bands;ii++) {
    int jj;
    cls_out_b[ii]=0;
    for(jj=0;jj<=lmax;jj++)
      cls_out_b[ii]+=bpw_windows[ii*(lmax+1)+jj]*cls_in[jj];
  }

  for(ii=0;ii<bin->n_bands;ii++) {
    ASSERT_DBL_NEAR_TOL(cls_out_a[ii],
			cls_out_b[ii],
			1E-5*fabs(cls_out_a[ii]));
  }

  free(cls_in);
  free(cls_zero);
  free(cls_coupled);
  free(cls_out_a);
  free(cls_out_b);
  free(bpw_windows);
  nmt_workspace_free(w);
  nmt_field_free(f0);
  nmt_bins_free(bin);
  free(mpt);
  free(msk);
  free(cs);
}  

CTEST(nmt,master_lite_errors) {
  double *mpt,*msk;
  nmt_curvedsky_info *cs=nmt_curvedsky_info_alloc(1,1,-1,-1,-1,-1,-1,-1,-1);
  nmt_field *f0,*f0l,*f0e;
  msk=he_read_map("test/benchmarks/msk.fits",cs,0);
  mpt=he_read_map("test/benchmarks/mps.fits",cs,0);
  f0=nmt_field_alloc_sph(cs,msk,0,&mpt,0,NULL,NULL,0,0,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  free(mpt);
  mpt=he_read_map("test/benchmarks/mps.fits",cs,0);
  f0l=nmt_field_alloc_sph(cs,msk,0,&mpt,0,NULL,NULL,0,0,3,1E-10,HE_NITER_DEFAULT,0,1,0);
  f0e=nmt_field_alloc_sph(cs,msk,0,NULL,0,NULL,NULL,0,0,3,1E-10,HE_NITER_DEFAULT,0,1,1);
  
  set_error_policy(THROW_ON_ERROR);
  //Noise deprojection bias
  try { nmt_compute_uncorr_noise_deprojection_bias(f0l,NULL,NULL,0); }
  try { nmt_compute_uncorr_noise_deprojection_bias(f0e,NULL,NULL,0); }
  //Deprojection bias
  try { nmt_compute_deprojection_bias(f0l,f0,NULL,NULL,0); }
  try { nmt_compute_deprojection_bias(f0l,f0l,NULL,NULL,0); }
  try { nmt_compute_deprojection_bias(f0l,f0e,NULL,NULL,0); }
  try { nmt_compute_deprojection_bias(f0e,f0e,NULL,NULL,0); }
  //Correlate maskless fields
  try { nmt_compute_coupled_cell(f0e,f0e,NULL); }
  set_error_policy(EXIT_ON_ERROR);

  nmt_field_free(f0);
  nmt_field_free(f0e);
  nmt_field_free(f0l);
  free(mpt);
  free(msk);
  free(cs);
}

CTEST(nmt,master_errors) {
  nmt_curvedsky_info *cs=nmt_curvedsky_info_alloc(1,1,-1,-1,-1,-1,-1,-1,-1);
  double *mpt=he_read_map("test/benchmarks/mps.fits",cs,0);
  double *msk=he_read_map("test/benchmarks/msk.fits",cs,0);
  long lmax=he_get_lmax(cs);
  
  nmt_binning_scheme *bin;
  nmt_workspace *w=NULL,*wb=NULL;
  nmt_curvedsky_info *cs_half=nmt_curvedsky_info_alloc(1,cs->n_eq/2,-1,-1,-1,-1,-1,-1,-1);

  nmt_field *f0=nmt_field_alloc_sph(cs,msk,0,&mpt,0,NULL,NULL,0,0,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  nmt_field *f0b=nmt_field_alloc_sph(cs_half,msk,0,&mpt,0,NULL,NULL,0,0,3,1E-10,HE_NITER_DEFAULT,0,0,0);
  
  set_error_policy(THROW_ON_ERROR);

  //Read from non-existent file
  try { w=nmt_workspace_read_fits("nofile.fits",1); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(w);
  //Wrong bins
  bin=nmt_bins_constant(20,6*cs->n_eq-1,0);
  try { w=nmt_compute_coupling_matrix(f0,f0,bin,0,HE_NITER_DEFAULT,-1,-1,-1,-1); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(w);
  nmt_bins_free(bin);
  //Mismatching resolutions
  bin=nmt_bins_constant(20,lmax,0);
  try { w=nmt_compute_coupling_matrix(f0,f0b,bin,0,HE_NITER_DEFAULT,-1,-1,-1,-1); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(w);
  //Wrong fields for teb
  try { w=nmt_compute_coupling_matrix(f0,f0,bin,1,HE_NITER_DEFAULT,-1,-1,-1,-1); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(w);
  //Try through nmt_compute_power_spectra
  try { w=nmt_compute_power_spectra(f0,f0b,bin,NULL,NULL,NULL,NULL,HE_NITER_DEFAULT,-1,-1,-1,-1); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(w);
  //nmt_compute_power_spectra with mis-matching input workspace
  w=nmt_compute_coupling_matrix(f0,f0,bin,0,HE_NITER_DEFAULT,-1,-1,-1,-1);
  nmt_bins_free(bin);
  bin=nmt_bins_constant(20,3*cs->n_eq/2-1,0);
  try { wb=nmt_compute_power_spectra(f0b,f0b,bin,w,NULL,NULL,NULL,HE_NITER_DEFAULT,-1,-1,-1,-1); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(wb);
  //nmt_update_coupling_matrix with wrong input
  double *mat=calloc((lmax+1)*(lmax+1),sizeof(double));
  try { nmt_update_coupling_matrix(w,(lmax+1)*2,mat); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  //nmt_update_coupling_matrix works as expected
  mat[3*cs->n_eq*2+cs->n_eq+1]=1.2345;
  nmt_update_coupling_matrix(w,(lmax+1),mat);
  ASSERT_DBL_NEAR_TOL(w->coupling_matrix_unbinned[2][cs->n_eq+1],1.2345,1E-10);
  free(mat);
  //nmt_update_binning with wrong input
  nmt_binning_scheme *bin2=nmt_bins_constant(20,lmax/2,0);
  try { nmt_workspace_update_binning(w,bin2); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  nmt_bins_free(bin2);
  //nmt_update_binning works as expected
  bin2=nmt_bins_constant(4,lmax,0);
  nmt_workspace_update_binning(w,bin2);
  ASSERT_EQUAL(3*cs->n_eq-1,w->bin->ell_max);
  nmt_bins_free(bin2);
  //nmt_update_beams with wrong input
  try { nmt_workspace_update_beams(w,-1,NULL,-1,NULL); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  //nmt_update_beams works as expected
  int ii;
  double *beam2=malloc((lmax+1)*sizeof(flouble));
  for(ii=0;ii<=lmax;ii++)
    beam2[ii]=1;
  nmt_workspace_update_beams(w,
			     lmax+1,beam2,
			     lmax+1,beam2);
  ASSERT_DBL_NEAR_TOL(w->beam_prod[4],1,1E-10);
  free(beam2);
  nmt_workspace_free(w);
  set_error_policy(EXIT_ON_ERROR);

  nmt_field_free(f0);
  nmt_field_free(f0b);
  nmt_bins_free(bin);
  free(mpt);
  free(msk);
  free(cs);
  free(cs_half);
}
