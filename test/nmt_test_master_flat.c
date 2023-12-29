#include "namaster.h"
#include "ctest.h"
#include "utils.h"
#include "nmt_test_utils.h"

CTEST(nmt,master_teb_flat_full) {
  //Generate fields and compute coupling matrix
  int ii,nx,ny;
  double lx,ly;
  nmt_field_flat *f0,*f2;
  nmt_workspace_flat *w_teb,*w00,*w02,*w22;
  nmt_binning_scheme_flat *bin;
  int nell=25;
  double dell=20.;
  double *larr=my_malloc((nell+1)*sizeof(double));
  double *msk=fs_read_flat_map("test/benchmarks/msk_flat.fits",&nx,&ny,&lx,&ly,0);
  double **mps0=my_malloc(sizeof(double *));
  double **mps2=my_malloc(2*sizeof(double *));
  mps0[0]=fs_read_flat_map("test/benchmarks/mps_flat.fits",&nx,&ny,&lx,&ly,0);
  mps2[0]=fs_read_flat_map("test/benchmarks/mps_flat.fits",&nx,&ny,&lx,&ly,1);
  mps2[1]=fs_read_flat_map("test/benchmarks/mps_flat.fits",&nx,&ny,&lx,&ly,2);

  for(ii=0;ii<=nell;ii++)
    larr[ii]=ii*dell+2;
  bin=nmt_bins_flat_create(nell,larr,&(larr[1]));

  //Init power spectra
  int ncls=7;
  int nlth=1000;
  double *lth=my_malloc(nlth*sizeof(double));
  double **cell=my_malloc(ncls*sizeof(double *));
  double **cell_out=my_malloc(ncls*sizeof(double *));
  double **cell_out_teb=my_malloc(ncls*sizeof(double *));
  double **cell_signal=my_malloc(ncls*sizeof(double *));
  double **cell_noise=my_malloc(ncls*sizeof(double *));
  double **cell_noise_coup=my_malloc(ncls*sizeof(double *));
  double **cell_deproj=my_malloc(ncls*sizeof(double *));
  for(ii=0;ii<ncls;ii++) {
    cell[ii]=my_malloc(bin->n_bands*sizeof(double));
    cell_noise[ii]=my_calloc(nlth,sizeof(double));
    cell_signal[ii]=my_calloc(nlth,sizeof(double));
    cell_deproj[ii]=my_calloc(bin->n_bands,sizeof(double));
    cell_out[ii]=my_malloc(bin->n_bands*sizeof(double));
    cell_out_teb[ii]=my_malloc(bin->n_bands*sizeof(double));
    cell_noise_coup[ii]=my_calloc(bin->n_bands,sizeof(double));
  }

  //Read signal and noise power spectrum
  FILE *fi=fopen("test/benchmarks/cls_lss.txt","r");
  for(ii=0;ii<nlth;ii++) {
    int jj;
    double dum;
    int stat=fscanf(fi,"%lf %lf %lf %lf %lf %lf %lf %lf %lf",&(lth[ii]),
		    &(cell_signal[0][ii]),&(cell_signal[3][ii]),&(cell_signal[6][ii]),&(cell_signal[1][ii]),
		    &(cell_noise[0][ii]),&(cell_noise[3][ii]),&(cell_noise[6][ii]),&(cell_noise[1][ii]));
    ASSERT_EQUAL(stat,9);
    for(jj=0;jj<ncls;jj++) //Add noise to signal
      cell_signal[jj][ii]+=cell_noise[jj][ii];
  }
  fclose(fi);

  //No contaminants
  f0=nmt_field_flat_alloc(NX_TEST,NY_TEST,
  			  DX_TEST*NX_TEST*M_PI/180,
			  DY_TEST*NY_TEST*M_PI/180,
			  msk,0,mps0,0,NULL,0,NULL,NULL,0,0,1E-10,0,0,0);
  f2=nmt_field_flat_alloc(NX_TEST,NY_TEST,
  			  DX_TEST*NX_TEST*M_PI/180,
			  DY_TEST*NY_TEST*M_PI/180,
			  msk,2,mps2,0,NULL,0,NULL,NULL,0,0,1E-10,0,0,0);
  w_teb=nmt_compute_coupling_matrix_flat(f0,f2,bin,1,-1,1,-1,1);
  w00=nmt_compute_coupling_matrix_flat(f0,f0,bin,1,-1,1,-1,0);
  w02=nmt_compute_coupling_matrix_flat(f0,f2,bin,1,-1,1,-1,0);
  w22=nmt_compute_coupling_matrix_flat(f2,f2,bin,1,-1,1,-1,0);
  nmt_couple_cl_l_flat_fast(w00,nlth,lth,&(cell_noise[0]),&(cell_noise_coup[0]));
  nmt_couple_cl_l_flat_fast(w02,nlth,lth,&(cell_noise[1]),&(cell_noise_coup[1]));
  nmt_couple_cl_l_flat_fast(w22,nlth,lth,&(cell_noise[3]),&(cell_noise_coup[3]));
  nmt_compute_coupled_cell_flat(f0,f0,bin,&(cell[0]),1,-1,1,-1);
  nmt_compute_coupled_cell_flat(f0,f2,bin,&(cell[1]),1,-1,1,-1);
  nmt_compute_coupled_cell_flat(f2,f2,bin,&(cell[3]),1,-1,1,-1);
  nmt_decouple_cl_l_flat(w_teb,cell,cell_noise_coup,cell_deproj,cell_out_teb);
  nmt_decouple_cl_l_flat(w00,&(cell[0]),&(cell_noise_coup[0]),&(cell_deproj[0]),&(cell_out[0]));
  nmt_decouple_cl_l_flat(w02,&(cell[1]),&(cell_noise_coup[1]),&(cell_deproj[1]),&(cell_out[1]));
  nmt_decouple_cl_l_flat(w22,&(cell[3]),&(cell_noise_coup[3]),&(cell_deproj[3]),&(cell_out[3]));
  for(ii=0;ii<ncls;ii++) {
    int ll;
    for(ll=0;ll<bin->n_bands;ll++)
      ASSERT_DBL_NEAR_TOL(cell_out[ii][ll],cell_out_teb[ii][ll],1E-20);
  }
  nmt_workspace_flat_free(w_teb);
  nmt_workspace_flat_free(w00);
  nmt_workspace_flat_free(w02);
  nmt_workspace_flat_free(w22);
  nmt_field_flat_free(f0);
  nmt_field_flat_free(f2);
  
  //Free up power spectra
  for(ii=0;ii<ncls;ii++) {
    free(cell[ii]);
    free(cell_noise[ii]);
    free(cell_signal[ii]);
    free(cell_deproj[ii]);
    free(cell_out[ii]);
    free(cell_out_teb[ii]);
    free(cell_noise_coup[ii]);
  }
  free(cell); free(cell_noise); free(cell_signal); free(cell_deproj);
  free(cell_out); free(cell_out_teb); free(cell_noise_coup); free(lth);

  nmt_bins_flat_free(bin);
  free(larr);
  for(ii=0;ii<2;ii++)
    free(mps2[ii]);
  free(mps0[0]);
  free(mps0);
  free(mps2);
  free(msk);
}

CTEST(nmt,master_22_flat_full) {
  //Generate fields and compute coupling matrix
  int ii,nx,ny;
  double lx,ly;
  nmt_field_flat *f2;
  nmt_workspace_flat *w22;
  nmt_binning_scheme_flat *bin;
  int nell=25;
  double dell=20.;
  double *larr=my_malloc((nell+1)*sizeof(double));
  double *msk=fs_read_flat_map("test/benchmarks/msk_flat.fits",&nx,&ny,&lx,&ly,0);
  double **mps2=my_malloc(2*sizeof(double *));
  double **tmp2=my_malloc(2*sizeof(double *));

  for(ii=0;ii<=nell;ii++)
    larr[ii]=ii*dell+2;
  bin=nmt_bins_flat_create(nell,larr,&(larr[1]));

  //Init power spectra
  int ncls=4;
  int nlth=1000;
  double *lth=my_malloc(nlth*sizeof(double));
  double **cell=my_malloc(ncls*sizeof(double *));
  double **cell_out=my_malloc(ncls*sizeof(double *));
  double **cell_signal=my_malloc(ncls*sizeof(double *));
  double **cell_noise=my_malloc(ncls*sizeof(double *));
  double **cell_noise_coup=my_malloc(ncls*sizeof(double *));
  double **cell_deproj=my_malloc(ncls*sizeof(double *));
  for(ii=0;ii<ncls;ii++) {
    cell[ii]=my_malloc(bin->n_bands*sizeof(double));
    cell_noise[ii]=my_calloc(nlth,sizeof(double));
    cell_signal[ii]=my_calloc(nlth,sizeof(double));
    cell_deproj[ii]=my_calloc(bin->n_bands,sizeof(double));
    cell_out[ii]=my_malloc(bin->n_bands*sizeof(double));
    cell_noise_coup[ii]=my_calloc(bin->n_bands,sizeof(double));
  }

  //Read signal and noise power spectrum
  FILE *fi=fopen("test/benchmarks/cls_lss.txt","r");
  for(ii=0;ii<nlth;ii++) {
    int jj;
    double dum;
    int stat=fscanf(fi,"%lf %lf %lf %lf %lf %lf %lf %lf %lf",&(lth[ii]),
		    &dum,&(cell_signal[0][ii]),&(cell_signal[3][ii]),&dum,
		    &dum,&(cell_noise[0][ii]),&(cell_noise[3][ii]),&dum);
    ASSERT_EQUAL(stat,9);
    for(jj=0;jj<ncls;jj++) //Add noise to signal
      cell_signal[jj][ii]+=cell_noise[jj][ii];
  }
  fclose(fi);

  //No contaminants
  mps2[0]=fs_read_flat_map("test/benchmarks/mps_flat.fits",&nx,&ny,&lx,&ly,1);
  mps2[1]=fs_read_flat_map("test/benchmarks/mps_flat.fits",&nx,&ny,&lx,&ly,2);
  f2=nmt_field_flat_alloc(NX_TEST,NY_TEST,
  			  DX_TEST*NX_TEST*M_PI/180,
			  DY_TEST*NY_TEST*M_PI/180,
			  msk,2,mps2,0,NULL,0,NULL,NULL,0,0,1E-10,0,0,0);
  w22=nmt_compute_coupling_matrix_flat(f2,f2,bin,1,-1,1,-1,0);
  nmt_couple_cl_l_flat_fast(w22,nlth,lth,cell_noise,cell_noise_coup);
  nmt_compute_coupled_cell_flat(f2,f2,bin,cell,1,-1,1,-1);
  nmt_decouple_cl_l_flat(w22,cell,cell_noise_coup,cell_deproj,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_f_nc_np_c22.txt",1E-3);
  nmt_workspace_flat_free(w22);
  nmt_field_flat_free(f2);
  for(ii=0;ii<2;ii++)
    free(mps2[ii]);

  //With purification
  mps2[0]=fs_read_flat_map("test/benchmarks/mps_flat.fits",&nx,&ny,&lx,&ly,1);
  mps2[1]=fs_read_flat_map("test/benchmarks/mps_flat.fits",&nx,&ny,&lx,&ly,2);
  f2=nmt_field_flat_alloc(NX_TEST,NY_TEST,
  			  DX_TEST*NX_TEST*M_PI/180,
			  DY_TEST*NY_TEST*M_PI/180,
			  msk,2,mps2,0,NULL,0,NULL,NULL,0,1,1E-10,0,0,0);
  w22=nmt_compute_coupling_matrix_flat(f2,f2,bin,1,-1,1,-1,0);
  nmt_couple_cl_l_flat_fast(w22,nlth,lth,cell_noise,cell_noise_coup);
  nmt_compute_coupled_cell_flat(f2,f2,bin,cell,1,-1,1,-1);
  nmt_decouple_cl_l_flat(w22,cell,cell_noise_coup,cell_deproj,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_f_nc_yp_c22.txt",1E-3);
  nmt_workspace_flat_free(w22);
  nmt_field_flat_free(f2);
  for(ii=0;ii<2;ii++)
    free(mps2[ii]);

  //With contaminants
  mps2[0]=fs_read_flat_map("test/benchmarks/mps_flat.fits",&nx,&ny,&lx,&ly,1);
  mps2[1]=fs_read_flat_map("test/benchmarks/mps_flat.fits",&nx,&ny,&lx,&ly,2);
  tmp2[0]=fs_read_flat_map("test/benchmarks/tmp_flat.fits",&nx,&ny,&lx,&ly,1);
  tmp2[1]=fs_read_flat_map("test/benchmarks/tmp_flat.fits",&nx,&ny,&lx,&ly,2);
  f2=nmt_field_flat_alloc(NX_TEST,NY_TEST,
  			  DX_TEST*NX_TEST*M_PI/180,
			  DY_TEST*NY_TEST*M_PI/180,
			  msk,2,mps2,1,&tmp2,0,NULL,NULL,0,0,1E-10,0,0,0);
  w22=nmt_compute_coupling_matrix_flat(f2,f2,bin,1,-1,1,-1,0);
  nmt_couple_cl_l_flat_fast(w22,nlth,lth,cell_noise,cell_noise_coup);
  nmt_compute_deprojection_bias_flat(f2,f2,bin,1,-1,1,-1,nlth,lth,cell_signal,cell_deproj);
  for(ii=0;ii<ncls;ii++) {
    test_compare_arrays(bin->n_bands,cell_deproj[ii],ncls,ii,
			"test/benchmarks/bm_f_yc_np_cb22.txt",1E-3);
  }
  nmt_compute_coupled_cell_flat(f2,f2,bin,cell,1,-1,1,-1);
  nmt_decouple_cl_l_flat(w22,cell,cell_noise_coup,cell_deproj,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_f_yc_np_c22.txt",1E-3);
  nmt_workspace_flat_free(w22);
  nmt_field_flat_free(f2);
  for(ii=0;ii<2;ii++) {
    free(tmp2[ii]);
    free(mps2[ii]);
  }

  //With contaminants, with purification
  mps2[0]=fs_read_flat_map("test/benchmarks/mps_flat.fits",&nx,&ny,&lx,&ly,1);
  mps2[1]=fs_read_flat_map("test/benchmarks/mps_flat.fits",&nx,&ny,&lx,&ly,2);
  tmp2[0]=fs_read_flat_map("test/benchmarks/tmp_flat.fits",&nx,&ny,&lx,&ly,1);
  tmp2[1]=fs_read_flat_map("test/benchmarks/tmp_flat.fits",&nx,&ny,&lx,&ly,2);
  f2=nmt_field_flat_alloc(NX_TEST,NY_TEST,
  			  DX_TEST*NX_TEST*M_PI/180,
			  DY_TEST*NY_TEST*M_PI/180,
			  msk,2,mps2,1,&tmp2,0,NULL,NULL,0,1,1E-10,0,0,0);
  w22=nmt_compute_coupling_matrix_flat(f2,f2,bin,1,-1,1,-1,0);
  nmt_couple_cl_l_flat_fast(w22,nlth,lth,cell_noise,cell_noise_coup);
  nmt_compute_deprojection_bias_flat(f2,f2,bin,1,-1,1,-1,nlth,lth,cell_signal,cell_deproj);
  for(ii=0;ii<ncls;ii++) {
    test_compare_arrays(bin->n_bands,cell_deproj[ii],ncls,ii,
			"test/benchmarks/bm_f_yc_yp_cb22.txt",1E-10);
  }
  nmt_compute_coupled_cell_flat(f2,f2,bin,cell,1,-1,1,-1);
  nmt_decouple_cl_l_flat(w22,cell,cell_noise_coup,cell_deproj,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_f_yc_yp_c22.txt",1E-3);
  nmt_workspace_flat_free(w22);
  nmt_field_flat_free(f2);
  for(ii=0;ii<2;ii++) {
    free(tmp2[ii]);
    free(mps2[ii]);
  }
  
  //Free up power spectra
  for(ii=0;ii<ncls;ii++) {
    free(cell[ii]);
    free(cell_noise[ii]);
    free(cell_signal[ii]);
    free(cell_deproj[ii]);
    free(cell_out[ii]);
    free(cell_noise_coup[ii]);
  }
  free(cell); free(cell_noise); free(cell_signal); free(cell_deproj);
  free(cell_out); free(cell_noise_coup); free(lth);

  nmt_bins_flat_free(bin);
  free(larr);
  free(mps2);
  free(tmp2);
  free(msk);
}

CTEST(nmt,master_11_flat_full) {
  //Generate fields and compute coupling matrix
  int ii,nx,ny;
  double lx,ly;
  nmt_field_flat *f1;
  nmt_workspace_flat *w11;
  nmt_binning_scheme_flat *bin;
  int nell=25;
  double dell=20.;
  double *larr=my_malloc((nell+1)*sizeof(double));
  double *msk=fs_read_flat_map("test/benchmarks/msk_flat.fits",&nx,&ny,&lx,&ly,0);
  double **mps1=my_malloc(2*sizeof(double *));
  mps1[0]=fs_read_flat_map("test/benchmarks/mps_sp1_flat.fits",&nx,&ny,&lx,&ly,1);
  mps1[1]=fs_read_flat_map("test/benchmarks/mps_sp1_flat.fits",&nx,&ny,&lx,&ly,2);

  for(ii=0;ii<=nell;ii++)
    larr[ii]=ii*dell+2;
  bin=nmt_bins_flat_create(nell,larr,&(larr[1]));

  //Init power spectra
  int ncls=4;
  int nlth=1000;
  double *lth=my_malloc(nlth*sizeof(double));
  double **cell=my_malloc(ncls*sizeof(double *));
  double **cell_out=my_malloc(ncls*sizeof(double *));
  double **cell0=my_malloc(ncls*sizeof(double *));
  double **cell0_coup=my_malloc(ncls*sizeof(double *));
  for(ii=0;ii<ncls;ii++) {
    cell[ii]=my_malloc(bin->n_bands*sizeof(double));
    cell0[ii]=my_calloc(nlth,sizeof(double));
    cell0_coup[ii]=my_calloc(bin->n_bands,sizeof(double));
    cell_out[ii]=my_malloc(bin->n_bands*sizeof(double));
  }

  //Read signal and noise power spectrum
  FILE *fi=fopen("test/benchmarks/cls_lss.txt","r");
  for(ii=0;ii<nlth;ii++) {
    int jj;
    double dum;
    int stat=fscanf(fi,"%lf %lf %lf %lf %lf %lf %lf %lf %lf",&(lth[ii]),
		    &dum,&dum,&dum,&dum,&dum,&dum,&dum,&dum);
    ASSERT_EQUAL(stat,9);
  }
  fclose(fi);

  //No contaminants
  f1=nmt_field_flat_alloc(NX_TEST,NY_TEST,
  			  DX_TEST*NX_TEST*M_PI/180,
			  DY_TEST*NY_TEST*M_PI/180,
			  msk,1,mps1,0,NULL,0,NULL,NULL,0,0,1E-10,0,0,0);
  w11=nmt_compute_coupling_matrix_flat(f1,f1,bin,1,-1,1,-1,0);
  nmt_couple_cl_l_flat_fast(w11,nlth,lth,cell0,cell0_coup);
  nmt_compute_coupled_cell_flat(f1,f1,bin,cell,1,-1,1,-1);
  nmt_decouple_cl_l_flat(w11,cell,cell0_coup,cell0_coup,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_f_sp1_c11.txt",1E-3);
  nmt_workspace_flat_free(w11);
  nmt_field_flat_free(f1);
  
  //Free up power spectra
  for(ii=0;ii<ncls;ii++) {
    free(cell[ii]);
    free(cell0[ii]);
    free(cell0_coup[ii]);
    free(cell_out[ii]);
  }
  free(cell); free(cell0); free(cell0_coup);
  free(cell_out); free(lth);

  nmt_bins_flat_free(bin);
  free(larr);
  for(ii=0;ii<2;ii++)
    free(mps1[ii]);
  free(mps1);
  free(msk);
}

CTEST(nmt,master_lite_flat) {
  //Generate fields and compute coupling matrix
  int ii,nx,ny;
  double lx,ly;
  nmt_field_flat *f0,*f2,*f2l,*f2e;
  nmt_workspace_flat *w02;
  nmt_binning_scheme_flat *bin;
  int nell=25;
  double dell=20.;
  double *larr=my_malloc((nell+1)*sizeof(double));
  double *msk=fs_read_flat_map("test/benchmarks/msk_flat.fits",&nx,&ny,&lx,&ly,0);
  double **mps0=my_malloc(sizeof(double *));
  double **mps2=my_malloc(2*sizeof(double *));
  double **tmp0=my_malloc(sizeof(double *));
  double **tmp2=my_malloc(2*sizeof(double *));

  for(ii=0;ii<=nell;ii++)
    larr[ii]=ii*dell+2;
  bin=nmt_bins_flat_create(nell,larr,&(larr[1]));

  //Init power spectra
  int ncls=2;
  int nlth=1000;
  double *lth=my_malloc(nlth*sizeof(double));
  double **cell=my_malloc(ncls*sizeof(double *));
  double **cell_out=my_malloc(ncls*sizeof(double *));
  double **cell_signal=my_malloc(ncls*sizeof(double *));
  double **cell_noise=my_malloc(ncls*sizeof(double *));
  double **cell_noise_coup=my_malloc(ncls*sizeof(double *));
  double **cell_deproj=my_malloc(ncls*sizeof(double *));
  for(ii=0;ii<ncls;ii++) {
    cell[ii]=my_malloc(bin->n_bands*sizeof(double));
    cell_noise[ii]=my_calloc(nlth,sizeof(double));
    cell_signal[ii]=my_calloc(nlth,sizeof(double));
    cell_deproj[ii]=my_calloc(bin->n_bands,sizeof(double));
    cell_out[ii]=my_malloc(bin->n_bands*sizeof(double));
    cell_noise_coup[ii]=my_calloc(bin->n_bands,sizeof(double));
  }
  //Read signal and noise power spectrum
  FILE *fi=fopen("test/benchmarks/cls_lss.txt","r");
  for(ii=0;ii<nlth;ii++) {
    int jj;
    double dum;
    int stat=fscanf(fi,"%lf %lf %lf %lf %lf %lf %lf %lf %lf",&(lth[ii]),
		    &dum,&dum,&dum,&(cell_signal[0][ii]),
		    &dum,&dum,&dum,&(cell_noise[0][ii]));
    ASSERT_EQUAL(stat,9);
    for(jj=0;jj<ncls;jj++) //Add noise to signal
      cell_signal[jj][ii]+=cell_noise[jj][ii];
  }
  fclose(fi);

  //With contaminants, with purification
  mps0[0]=fs_read_flat_map("test/benchmarks/mps_flat.fits",&nx,&ny,&lx,&ly,0);
  mps2[0]=fs_read_flat_map("test/benchmarks/mps_flat.fits",&nx,&ny,&lx,&ly,1);
  mps2[1]=fs_read_flat_map("test/benchmarks/mps_flat.fits",&nx,&ny,&lx,&ly,2);
  tmp0[0]=fs_read_flat_map("test/benchmarks/tmp_flat.fits",&nx,&ny,&lx,&ly,0);
  tmp2[0]=fs_read_flat_map("test/benchmarks/tmp_flat.fits",&nx,&ny,&lx,&ly,1);
  tmp2[1]=fs_read_flat_map("test/benchmarks/tmp_flat.fits",&nx,&ny,&lx,&ly,2);
  f0=nmt_field_flat_alloc(NX_TEST,NY_TEST,
    			  DX_TEST*NX_TEST*M_PI/180,
			  DY_TEST*NY_TEST*M_PI/180,
			  msk,0,mps0,1,&tmp0,0,NULL,NULL,0,0,1E-10,0,0,0);
  f2=nmt_field_flat_alloc(NX_TEST,NY_TEST,
    			  DX_TEST*NX_TEST*M_PI/180,
			  DY_TEST*NY_TEST*M_PI/180,
			  msk,2,mps2,1,&tmp2,0,NULL,NULL,0,1,1E-10,0,0,0);
  // Lite field
  for(ii=0;ii<2;ii++) {
    free(tmp2[ii]);
    free(mps2[ii]);
  }
  mps2[0]=fs_read_flat_map("test/benchmarks/mps_flat.fits",&nx,&ny,&lx,&ly,1);
  mps2[1]=fs_read_flat_map("test/benchmarks/mps_flat.fits",&nx,&ny,&lx,&ly,2);
  tmp2[0]=fs_read_flat_map("test/benchmarks/tmp_flat.fits",&nx,&ny,&lx,&ly,1);
  tmp2[1]=fs_read_flat_map("test/benchmarks/tmp_flat.fits",&nx,&ny,&lx,&ly,2);
  f2l=nmt_field_flat_alloc(NX_TEST,NY_TEST,
                           DX_TEST*NX_TEST*M_PI/180,
                           DY_TEST*NY_TEST*M_PI/180,
                           msk,2,mps2,1,&tmp2,0,NULL,NULL,0,1,1E-10,0,1,0);
  // Mask-only field
  f2e=nmt_field_flat_alloc(NX_TEST,NY_TEST,
                           DX_TEST*NX_TEST*M_PI/180,
                           DY_TEST*NY_TEST*M_PI/180,
                           msk,2,mps2,1,&tmp2,0,NULL,NULL,0,1,1E-10,0,1,1);
  w02=nmt_compute_coupling_matrix_flat(f0,f2e,bin,1,-1,1,-1,0);
  nmt_couple_cl_l_flat_fast(w02,nlth,lth,cell_noise,cell_noise_coup);
  nmt_compute_deprojection_bias_flat(f0,f2,bin,1,-1,1,-1,nlth,lth,cell_signal,cell_deproj);
  for(ii=0;ii<ncls;ii++) {
    test_compare_arrays(bin->n_bands,cell_deproj[ii],ncls,ii,
    			"test/benchmarks/bm_f_yc_yp_cb02.txt",1E-3);
  }
  nmt_compute_coupled_cell_flat(f0,f2l,bin,cell,1,-1,1,-1);
  nmt_decouple_cl_l_flat(w02,cell,cell_noise_coup,cell_deproj,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_f_yc_yp_c02.txt",1E-10);
  nmt_workspace_flat_free(w02);
  nmt_field_flat_free(f0);
  nmt_field_flat_free(f2);
  nmt_field_flat_free(f2l);
  nmt_field_flat_free(f2e);
  for(ii=0;ii<2;ii++) {
    free(tmp2[ii]);
    free(mps2[ii]);
  }
  free(tmp0[0]);
  free(mps0[0]);
  
  //Free up power spectra
  for(ii=0;ii<ncls;ii++) {
    free(cell[ii]);
    free(cell_noise[ii]);
    free(cell_signal[ii]);
    free(cell_deproj[ii]);
    free(cell_out[ii]);
    free(cell_noise_coup[ii]);
  }
  free(cell); free(cell_noise); free(cell_signal); free(cell_deproj);
  free(cell_out); free(cell_noise_coup); free(lth);

  nmt_bins_flat_free(bin);
  free(larr);
  free(mps0);
  free(mps2);
  free(tmp0);
  free(tmp2);
  free(msk);
}

CTEST(nmt,master_02_flat_full) {
  //Generate fields and compute coupling matrix
  int ii,nx,ny;
  double lx,ly;
  nmt_field_flat *f0,*f2;
  nmt_workspace_flat *w02;
  nmt_binning_scheme_flat *bin;
  int nell=25;
  double dell=20.;
  double *larr=my_malloc((nell+1)*sizeof(double));
  double *msk=fs_read_flat_map("test/benchmarks/msk_flat.fits",&nx,&ny,&lx,&ly,0);
  double **mps0=my_malloc(sizeof(double *));
  double **mps2=my_malloc(2*sizeof(double *));
  double **tmp0=my_malloc(sizeof(double *));
  double **tmp2=my_malloc(2*sizeof(double *));

  for(ii=0;ii<=nell;ii++)
    larr[ii]=ii*dell+2;
  bin=nmt_bins_flat_create(nell,larr,&(larr[1]));

  //Init power spectra
  int ncls=2;
  int nlth=1000;
  double *lth=my_malloc(nlth*sizeof(double));
  double **cell=my_malloc(ncls*sizeof(double *));
  double **cell_out=my_malloc(ncls*sizeof(double *));
  double **cell_signal=my_malloc(ncls*sizeof(double *));
  double **cell_noise=my_malloc(ncls*sizeof(double *));
  double **cell_noise_coup=my_malloc(ncls*sizeof(double *));
  double **cell_deproj=my_malloc(ncls*sizeof(double *));
  for(ii=0;ii<ncls;ii++) {
    cell[ii]=my_malloc(bin->n_bands*sizeof(double));
    cell_noise[ii]=my_calloc(nlth,sizeof(double));
    cell_signal[ii]=my_calloc(nlth,sizeof(double));
    cell_deproj[ii]=my_calloc(bin->n_bands,sizeof(double));
    cell_out[ii]=my_malloc(bin->n_bands*sizeof(double));
    cell_noise_coup[ii]=my_calloc(bin->n_bands,sizeof(double));
  }

  //Read signal and noise power spectrum
  FILE *fi=fopen("test/benchmarks/cls_lss.txt","r");
  for(ii=0;ii<nlth;ii++) {
    int jj;
    double dum;
    int stat=fscanf(fi,"%lf %lf %lf %lf %lf %lf %lf %lf %lf",&(lth[ii]),
		    &dum,&dum,&dum,&(cell_signal[0][ii]),
		    &dum,&dum,&dum,&(cell_noise[0][ii]));
    ASSERT_EQUAL(stat,9);
    for(jj=0;jj<ncls;jj++) //Add noise to signal
      cell_signal[jj][ii]+=cell_noise[jj][ii];
  }
  fclose(fi);

  //No contaminants
  mps0[0]=fs_read_flat_map("test/benchmarks/mps_flat.fits",&nx,&ny,&lx,&ly,0);
  mps2[0]=fs_read_flat_map("test/benchmarks/mps_flat.fits",&nx,&ny,&lx,&ly,1);
  mps2[1]=fs_read_flat_map("test/benchmarks/mps_flat.fits",&nx,&ny,&lx,&ly,2);
  f0=nmt_field_flat_alloc(NX_TEST,NY_TEST,
  			  DX_TEST*NX_TEST*M_PI/180,
			  DY_TEST*NY_TEST*M_PI/180,
			  msk,0,mps0,0,NULL,0,NULL,NULL,0,0,1E-10,0,0,0);
  f2=nmt_field_flat_alloc(NX_TEST,NY_TEST,
  			  DX_TEST*NX_TEST*M_PI/180,
			  DY_TEST*NY_TEST*M_PI/180,
			  msk,2,mps2,0,NULL,0,NULL,NULL,0,0,1E-10,0,0,0);
  w02=nmt_compute_coupling_matrix_flat(f0,f2,bin,1,-1,1,-1,0);
  nmt_couple_cl_l_flat_fast(w02,nlth,lth,cell_noise,cell_noise_coup);
  nmt_compute_coupled_cell_flat(f0,f2,bin,cell,1,-1,1,-1);
  nmt_decouple_cl_l_flat(w02,cell,cell_noise_coup,cell_deproj,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_f_nc_np_c02.txt",1E-3);
  nmt_workspace_flat_free(w02);
  nmt_field_flat_free(f0);
  nmt_field_flat_free(f2);
  for(ii=0;ii<2;ii++)
    free(mps2[ii]);
  free(mps0[0]);

  //With purification
  mps0[0]=fs_read_flat_map("test/benchmarks/mps_flat.fits",&nx,&ny,&lx,&ly,0);
  mps2[0]=fs_read_flat_map("test/benchmarks/mps_flat.fits",&nx,&ny,&lx,&ly,1);
  mps2[1]=fs_read_flat_map("test/benchmarks/mps_flat.fits",&nx,&ny,&lx,&ly,2);
  f0=nmt_field_flat_alloc(NX_TEST,NY_TEST,
  			  DX_TEST*NX_TEST*M_PI/180,
			  DY_TEST*NY_TEST*M_PI/180,
			  msk,0,mps0,0,NULL,0,NULL,NULL,0,0,1E-10,0,0,0);
  f2=nmt_field_flat_alloc(NX_TEST,NY_TEST,
  			  DX_TEST*NX_TEST*M_PI/180,
			  DY_TEST*NY_TEST*M_PI/180,
			  msk,2,mps2,0,NULL,0,NULL,NULL,0,1,1E-10,0,0,0);
  w02=nmt_compute_coupling_matrix_flat(f0,f2,bin,1,-1,1,-1,0);
  nmt_couple_cl_l_flat_fast(w02,nlth,lth,cell_noise,cell_noise_coup);
  nmt_compute_coupled_cell_flat(f0,f2,bin,cell,1,-1,1,-1);
  nmt_decouple_cl_l_flat(w02,cell,cell_noise_coup,cell_deproj,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_f_nc_yp_c02.txt",1E-3);
  nmt_workspace_flat_free(w02);
  nmt_field_flat_free(f0);
  nmt_field_flat_free(f2);
  for(ii=0;ii<2;ii++)
    free(mps2[ii]);
  free(mps0[0]);

  //With contaminants
  mps0[0]=fs_read_flat_map("test/benchmarks/mps_flat.fits",&nx,&ny,&lx,&ly,0);
  mps2[0]=fs_read_flat_map("test/benchmarks/mps_flat.fits",&nx,&ny,&lx,&ly,1);
  mps2[1]=fs_read_flat_map("test/benchmarks/mps_flat.fits",&nx,&ny,&lx,&ly,2);
  tmp0[0]=fs_read_flat_map("test/benchmarks/tmp_flat.fits",&nx,&ny,&lx,&ly,0);
  tmp2[0]=fs_read_flat_map("test/benchmarks/tmp_flat.fits",&nx,&ny,&lx,&ly,1);
  tmp2[1]=fs_read_flat_map("test/benchmarks/tmp_flat.fits",&nx,&ny,&lx,&ly,2);
  f0=nmt_field_flat_alloc(NX_TEST,NY_TEST,
  			  DX_TEST*NX_TEST*M_PI/180,
			  DY_TEST*NY_TEST*M_PI/180,
			  msk,0,mps0,1,&tmp0,0,NULL,NULL,0,0,1E-10,0,0,0);
  f2=nmt_field_flat_alloc(NX_TEST,NY_TEST,
  			  DX_TEST*NX_TEST*M_PI/180,
			  DY_TEST*NY_TEST*M_PI/180,
			  msk,2,mps2,1,&tmp2,0,NULL,NULL,0,0,1E-10,0,0,0);
  w02=nmt_compute_coupling_matrix_flat(f0,f2,bin,1,-1,1,-1,0);
  nmt_couple_cl_l_flat_fast(w02,nlth,lth,cell_noise,cell_noise_coup);
  nmt_compute_deprojection_bias_flat(f0,f2,bin,1,-1,1,-1,nlth,lth,cell_signal,cell_deproj);
  for(ii=0;ii<ncls;ii++) {
    test_compare_arrays(bin->n_bands,cell_deproj[ii],ncls,ii,
			"test/benchmarks/bm_f_yc_np_cb02.txt",1E-3);
  }
  nmt_compute_coupled_cell_flat(f0,f2,bin,cell,1,-1,1,-1);
  nmt_decouple_cl_l_flat(w02,cell,cell_noise_coup,cell_deproj,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_f_yc_np_c02.txt",1E-3);
  nmt_workspace_flat_free(w02);
  nmt_field_flat_free(f0);
  nmt_field_flat_free(f2);
  for(ii=0;ii<2;ii++) {
    free(tmp2[ii]);
    free(mps2[ii]);
  }
  free(tmp0[0]);
  free(mps0[0]);

  //With contaminants, with purification
  mps0[0]=fs_read_flat_map("test/benchmarks/mps_flat.fits",&nx,&ny,&lx,&ly,0);
  mps2[0]=fs_read_flat_map("test/benchmarks/mps_flat.fits",&nx,&ny,&lx,&ly,1);
  mps2[1]=fs_read_flat_map("test/benchmarks/mps_flat.fits",&nx,&ny,&lx,&ly,2);
  tmp0[0]=fs_read_flat_map("test/benchmarks/tmp_flat.fits",&nx,&ny,&lx,&ly,0);
  tmp2[0]=fs_read_flat_map("test/benchmarks/tmp_flat.fits",&nx,&ny,&lx,&ly,1);
  tmp2[1]=fs_read_flat_map("test/benchmarks/tmp_flat.fits",&nx,&ny,&lx,&ly,2);
  f0=nmt_field_flat_alloc(NX_TEST,NY_TEST,
    			  DX_TEST*NX_TEST*M_PI/180,
			  DY_TEST*NY_TEST*M_PI/180,
			  msk,0,mps0,1,&tmp0,0,NULL,NULL,0,0,1E-10,0,0,0);
  f2=nmt_field_flat_alloc(NX_TEST,NY_TEST,
    			  DX_TEST*NX_TEST*M_PI/180,
			  DY_TEST*NY_TEST*M_PI/180,
			  msk,2,mps2,1,&tmp2,0,NULL,NULL,0,1,1E-10,0,0,0);
  w02=nmt_compute_coupling_matrix_flat(f0,f2,bin,1,-1,1,-1,0);
  nmt_couple_cl_l_flat_fast(w02,nlth,lth,cell_noise,cell_noise_coup);
  nmt_compute_deprojection_bias_flat(f0,f2,bin,1,-1,1,-1,nlth,lth,cell_signal,cell_deproj);
  for(ii=0;ii<ncls;ii++) {
    test_compare_arrays(bin->n_bands,cell_deproj[ii],ncls,ii,
    			"test/benchmarks/bm_f_yc_yp_cb02.txt",1E-3);
  }
  nmt_compute_coupled_cell_flat(f0,f2,bin,cell,1,-1,1,-1);
  nmt_decouple_cl_l_flat(w02,cell,cell_noise_coup,cell_deproj,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_f_yc_yp_c02.txt",1E-10);
  nmt_workspace_flat_free(w02);
  nmt_field_flat_free(f0);
  nmt_field_flat_free(f2);
  for(ii=0;ii<2;ii++) {
    free(tmp2[ii]);
    free(mps2[ii]);
  }
  free(tmp0[0]);
  free(mps0[0]);
  
  //Free up power spectra
  for(ii=0;ii<ncls;ii++) {
    free(cell[ii]);
    free(cell_noise[ii]);
    free(cell_signal[ii]);
    free(cell_deproj[ii]);
    free(cell_out[ii]);
    free(cell_noise_coup[ii]);
  }
  free(cell); free(cell_noise); free(cell_signal); free(cell_deproj);
  free(cell_out); free(cell_noise_coup); free(lth);

  nmt_bins_flat_free(bin);
  free(larr);
  free(mps0);
  free(mps2);
  free(tmp0);
  free(tmp2);
  free(msk);
}

CTEST(nmt,master_01_flat_full) {
  //Generate fields and compute coupling matrix
  int ii,nx,ny;
  double lx,ly;
  nmt_field_flat *f0,*f1;
  nmt_workspace_flat *w01;
  nmt_binning_scheme_flat *bin;
  int nell=25;
  double dell=20.;
  double *larr=my_malloc((nell+1)*sizeof(double));
  double *msk=fs_read_flat_map("test/benchmarks/msk_flat.fits",&nx,&ny,&lx,&ly,0);
  double **mps0=my_malloc(sizeof(double *));
  double **mps1=my_malloc(2*sizeof(double *));
  mps0[0]=fs_read_flat_map("test/benchmarks/mps_sp1_flat.fits",&nx,&ny,&lx,&ly,0);
  mps1[0]=fs_read_flat_map("test/benchmarks/mps_sp1_flat.fits",&nx,&ny,&lx,&ly,1);
  mps1[1]=fs_read_flat_map("test/benchmarks/mps_sp1_flat.fits",&nx,&ny,&lx,&ly,2);

  for(ii=0;ii<=nell;ii++)
    larr[ii]=ii*dell+2;
  bin=nmt_bins_flat_create(nell,larr,&(larr[1]));

  //Init power spectra
  int ncls=2;
  int nlth=1000;
  double *lth=my_malloc(nlth*sizeof(double));
  double **cell=my_malloc(ncls*sizeof(double *));
  double **cell_out=my_malloc(ncls*sizeof(double *));
  double **cell0=my_malloc(ncls*sizeof(double *));
  double **cell0_coup=my_malloc(ncls*sizeof(double *));
  for(ii=0;ii<ncls;ii++) {
    cell[ii]=my_malloc(bin->n_bands*sizeof(double));
    cell0[ii]=my_calloc(nlth,sizeof(double));
    cell_out[ii]=my_malloc(bin->n_bands*sizeof(double));
    cell0_coup[ii]=my_calloc(bin->n_bands,sizeof(double));
  }

  //Read signal and noise power spectrum
  FILE *fi=fopen("test/benchmarks/cls_lss.txt","r");
  for(ii=0;ii<nlth;ii++) {
    int jj;
    double dum;
    int stat=fscanf(fi,"%lf %lf %lf %lf %lf %lf %lf %lf %lf",&(lth[ii]),
		    &dum,&dum,&dum,&dum,&dum,&dum,&dum,&dum);
    ASSERT_EQUAL(stat,9);
  }
  fclose(fi);

  //No contaminants
  f0=nmt_field_flat_alloc(NX_TEST,NY_TEST,
  			  DX_TEST*NX_TEST*M_PI/180,
			  DY_TEST*NY_TEST*M_PI/180,
			  msk,0,mps0,0,NULL,0,NULL,NULL,0,0,1E-10,0,0,0);
  f1=nmt_field_flat_alloc(NX_TEST,NY_TEST,
  			  DX_TEST*NX_TEST*M_PI/180,
			  DY_TEST*NY_TEST*M_PI/180,
			  msk,1,mps1,0,NULL,0,NULL,NULL,0,0,1E-10,0,0,0);
  w01=nmt_compute_coupling_matrix_flat(f0,f1,bin,1,-1,1,-1,0);
  nmt_couple_cl_l_flat_fast(w01,nlth,lth,cell0,cell0_coup);
  nmt_compute_coupled_cell_flat(f0,f1,bin,cell,1,-1,1,-1);
  nmt_decouple_cl_l_flat(w01,cell,cell0_coup,cell0_coup,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_f_sp1_c01.txt",1E-3);
  nmt_workspace_flat_free(w01);
  nmt_field_flat_free(f0);
  nmt_field_flat_free(f1);
  
  //Free up power spectra
  for(ii=0;ii<ncls;ii++) {
    free(cell[ii]);
    free(cell0[ii]);
    free(cell_out[ii]);
    free(cell0_coup[ii]);
  }
  free(cell); free(cell0);
  free(cell_out); free(cell0_coup); free(lth);

  nmt_bins_flat_free(bin);
  free(larr);
  for(ii=0;ii<1;ii++)
    free(mps1[ii]);
  free(mps0[0]);
  free(mps0);
  free(mps1);
  free(msk);
}

CTEST(nmt,master_sp1_flat_full) {
  //Generate fields and compute coupling matrix
  int ii,nx,ny;
  double lx,ly;
  nmt_field_flat *f0,*f1;
  nmt_workspace_flat *w00,*w01,*w11;
  nmt_binning_scheme_flat *bin;
  int nell=25;
  double dell=20.;
  double *larr=my_malloc((nell+1)*sizeof(double));
  double *msk=fs_read_flat_map("test/benchmarks/msk_flat.fits",&nx,&ny,&lx,&ly,0);
  double **mps0=my_malloc(sizeof(double *));
  double **mps1=my_malloc(2*sizeof(double *));
  mps0[0]=fs_read_flat_map("test/benchmarks/mps_sp1_flat.fits",&nx,&ny,&lx,&ly,0);
  mps1[0]=fs_read_flat_map("test/benchmarks/mps_sp1_flat.fits",&nx,&ny,&lx,&ly,1);
  mps1[1]=fs_read_flat_map("test/benchmarks/mps_sp1_flat.fits",&nx,&ny,&lx,&ly,2);

  for(ii=0;ii<=nell;ii++)
    larr[ii]=ii*dell+2;
  bin=nmt_bins_flat_create(nell,larr,&(larr[1]));

  //Init power spectra
  int ncls=4;
  int nlth=1000;
  double *lth=my_malloc(nlth*sizeof(double));
  double **cell00=my_malloc(ncls*sizeof(double *));
  double **cell01=my_malloc(ncls*sizeof(double *));
  double **cell11=my_malloc(ncls*sizeof(double *));
  double **cell00_out=my_malloc(ncls*sizeof(double *));
  double **cell01_out=my_malloc(ncls*sizeof(double *));
  double **cell11_out=my_malloc(ncls*sizeof(double *));
  double **cell0=my_malloc(ncls*sizeof(double *));
  double **cell0_coup=my_malloc(ncls*sizeof(double *));
  for(ii=0;ii<ncls;ii++) {
    cell00[ii]=my_malloc(bin->n_bands*sizeof(double));
    cell01[ii]=my_malloc(bin->n_bands*sizeof(double));
    cell11[ii]=my_malloc(bin->n_bands*sizeof(double));
    cell0[ii]=my_calloc(nlth,sizeof(double));
    cell00_out[ii]=my_malloc(bin->n_bands*sizeof(double));
    cell01_out[ii]=my_malloc(bin->n_bands*sizeof(double));
    cell11_out[ii]=my_malloc(bin->n_bands*sizeof(double));
    cell0_coup[ii]=my_calloc(bin->n_bands,sizeof(double));
  }

  //Read signal and noise power spectrum
  FILE *fi=fopen("test/benchmarks/cls_lss.txt","r");
  for(ii=0;ii<nlth;ii++) {
    int jj;
    double dum;
    int stat=fscanf(fi,"%lf %lf %lf %lf %lf %lf %lf %lf %lf",&(lth[ii]),
		    &dum,&dum,&dum,&dum,&dum,&dum,&dum,&dum);
    ASSERT_EQUAL(stat,9);
  }
  fclose(fi);

  //No contaminants
  f0=nmt_field_flat_alloc(NX_TEST,NY_TEST,
  			  DX_TEST*NX_TEST*M_PI/180,
			  DY_TEST*NY_TEST*M_PI/180,
			  msk,0,mps0,0,NULL,0,NULL,NULL,0,0,1E-10,0,0,0);
  f1=nmt_field_flat_alloc(NX_TEST,NY_TEST,
  			  DX_TEST*NX_TEST*M_PI/180,
			  DY_TEST*NY_TEST*M_PI/180,
			  msk,1,mps1,0,NULL,0,NULL,NULL,0,0,1E-10,0,0,0);
  w00=nmt_compute_coupling_matrix_flat(f0,f0,bin,1,-1,1,-1,0);
  w01=nmt_compute_coupling_matrix_flat(f0,f1,bin,1,-1,1,-1,0);
  w11=nmt_compute_coupling_matrix_flat(f1,f1,bin,1,-1,1,-1,0);
  nmt_couple_cl_l_flat_fast(w11,nlth,lth,cell0,cell0_coup);
  nmt_compute_coupled_cell_flat(f0,f0,bin,cell00,1,-1,1,-1);
  nmt_compute_coupled_cell_flat(f0,f1,bin,cell01,1,-1,1,-1);
  nmt_compute_coupled_cell_flat(f1,f1,bin,cell11,1,-1,1,-1);
  nmt_decouple_cl_l_flat(w00,cell00,cell0_coup,cell0_coup,cell00_out);
  nmt_decouple_cl_l_flat(w01,cell01,cell0_coup,cell0_coup,cell01_out);
  nmt_decouple_cl_l_flat(w11,cell11,cell0_coup,cell0_coup,cell11_out);
  for(ii=1;ii<bin->n_bands;ii++) {
    double cl=cell00[0][ii];
    double tol=0.03*cl;
    ASSERT_DBL_NEAR_TOL(cl, cell01[0][ii], tol);
    ASSERT_DBL_NEAR_TOL(cl, cell11[0][ii], tol);
  }
  nmt_workspace_flat_free(w00);
  nmt_workspace_flat_free(w01);
  nmt_workspace_flat_free(w11);
  nmt_field_flat_free(f0);
  nmt_field_flat_free(f1);
  
  //Free up power spectra
  for(ii=0;ii<ncls;ii++) {
    free(cell00[ii]);
    free(cell01[ii]);
    free(cell11[ii]);
    free(cell0[ii]);
    free(cell00_out[ii]);
    free(cell01_out[ii]);
    free(cell11_out[ii]);
    free(cell0_coup[ii]);
  }
  free(cell00); free(cell01); free(cell11);
  free(cell0); free(cell0_coup); free(lth);
  free(cell00_out); free(cell01_out); free(cell11_out);

  nmt_bins_flat_free(bin);
  free(larr);
  for(ii=0;ii<1;ii++)
    free(mps1[ii]);
  free(mps0[0]);
  free(mps0);
  free(mps1);
  free(msk);
}

CTEST(nmt,master_00_flat_full) {
  //Generate fields and compute coupling matrix
  int ii,nx,ny;
  double lx,ly;
  nmt_field_flat *f0;
  nmt_workspace_flat *w00;
  nmt_binning_scheme_flat *bin;
  int nell=25;
  double dell=20.;
  double *larr=my_malloc((nell+1)*sizeof(double));
  double *msk=fs_read_flat_map("test/benchmarks/msk_flat.fits",&nx,&ny,&lx,&ly,0);
  double *mps;
  double **tmp=my_malloc(sizeof(double *));

  for(ii=0;ii<=nell;ii++)
    larr[ii]=ii*dell+2;
  bin=nmt_bins_flat_create(nell,larr,&(larr[1]));

  //Init power spectra
  int ncls=1;
  int nlth=1000;
  double *lth=my_malloc(nlth*sizeof(double));
  double **cell=my_malloc(ncls*sizeof(double *));
  double **cell_out=my_malloc(ncls*sizeof(double *));
  double **cell_signal=my_malloc(ncls*sizeof(double *));
  double **cell_noise=my_malloc(ncls*sizeof(double *));
  double **cell_noise_coup=my_malloc(ncls*sizeof(double *));
  double **cell_deproj=my_malloc(ncls*sizeof(double *));
  for(ii=0;ii<ncls;ii++) {
    cell[ii]=my_malloc(bin->n_bands*sizeof(double));
    cell_noise[ii]=my_calloc(nlth,sizeof(double));
    cell_signal[ii]=my_calloc(nlth,sizeof(double));
    cell_deproj[ii]=my_calloc(bin->n_bands,sizeof(double));
    cell_out[ii]=my_malloc(bin->n_bands*sizeof(double));
    cell_noise_coup[ii]=my_calloc(bin->n_bands,sizeof(double));
  }

  //Read signal and noise power spectrum
  FILE *fi=fopen("test/benchmarks/cls_lss.txt","r");
  for(ii=0;ii<nlth;ii++) {
    int jj;
    double dum;
    int stat=fscanf(fi,"%lf %lf %lf %lf %lf %lf %lf %lf %lf",&(lth[ii]),
		    &(cell_signal[0][ii]),&dum,&dum,&dum,
		    &(cell_noise[0][ii]),&dum,&dum,&dum);
    ASSERT_EQUAL(stat,9);
    for(jj=0;jj<ncls;jj++) //Add noise to signal
      cell_signal[jj][ii]+=cell_noise[jj][ii];
  }
  fclose(fi);

  //No contaminants
  mps=fs_read_flat_map("test/benchmarks/mps_flat.fits",&nx,&ny,&lx,&ly,0);
  f0=nmt_field_flat_alloc(NX_TEST,NY_TEST,
  			  DX_TEST*NX_TEST*M_PI/180,
			  DY_TEST*NY_TEST*M_PI/180,
			  msk,0,&mps,0,NULL,0,NULL,NULL,0,0,1E-10,0,0,0);
  w00=nmt_compute_coupling_matrix_flat(f0,f0,bin,1,-1,1,-1,0);
  nmt_couple_cl_l_flat_fast(w00,nlth,lth,cell_noise,cell_noise_coup);
  nmt_compute_coupled_cell_flat(f0,f0,bin,cell,1,-1,1,-1);
  nmt_decouple_cl_l_flat(w00,cell,cell_noise_coup,cell_deproj,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_f_nc_np_c00.txt",1E-3);
  nmt_workspace_flat_free(w00);
  nmt_field_flat_free(f0);
  free(mps);

  //With contaminants
  mps=fs_read_flat_map("test/benchmarks/mps_flat.fits",&nx,&ny,&lx,&ly,0);
  tmp[0]=fs_read_flat_map("test/benchmarks/tmp_flat.fits",&nx,&ny,&lx,&ly,0);
  f0=nmt_field_flat_alloc(NX_TEST,NY_TEST,
  			  DX_TEST*NX_TEST*M_PI/180,
			  DY_TEST*NY_TEST*M_PI/180,
			  msk,0,&mps,1,&tmp,0,NULL,NULL,0,0,1E-10,0,0,0);
  w00=nmt_compute_coupling_matrix_flat(f0,f0,bin,1,-1,1,-1,0);
  nmt_couple_cl_l_flat_fast(w00,nlth,lth,cell_noise,cell_noise_coup);
  nmt_compute_deprojection_bias_flat(f0,f0,bin,1,-1,1,-1,nlth,lth,cell_signal,cell_deproj);
  for(ii=0;ii<ncls;ii++) {
    test_compare_arrays(bin->n_bands,cell_deproj[ii],ncls,ii,
			"test/benchmarks/bm_f_yc_np_cb00.txt",1E-3);
  }
  nmt_compute_coupled_cell_flat(f0,f0,bin,cell,1,-1,1,-1);
  nmt_decouple_cl_l_flat(w00,cell,cell_noise_coup,cell_deproj,cell_out);
  for(ii=0;ii<ncls;ii++)
    test_compare_arrays(bin->n_bands,cell_out[ii],ncls,ii,"test/benchmarks/bm_f_yc_np_c00.txt",1E-3);
  nmt_workspace_flat_free(w00);
  nmt_field_flat_free(f0);
  free(tmp[0]);
  free(mps);
  
  //Free up power spectra
  for(ii=0;ii<ncls;ii++) {
    free(cell[ii]);
    free(cell_noise[ii]);
    free(cell_signal[ii]);
    free(cell_deproj[ii]);
    free(cell_out[ii]);
    free(cell_noise_coup[ii]);
  }
  free(cell); free(cell_noise); free(cell_signal); free(cell_deproj);
  free(cell_out); free(cell_noise_coup); free(lth);

  nmt_bins_flat_free(bin);
  free(larr);
  free(tmp);
  free(msk);
}

CTEST(nmt,master_lite_flat_errors) {
  int ii,nx,ny;
  double lx,ly;
  double *msk,*mpt;
  nmt_field_flat *f0,*f0l,*f0e;
  msk=fs_read_flat_map("test/benchmarks/msk_flat.fits",&nx,&ny,&lx,&ly,0);
  mpt=fs_read_flat_map("test/benchmarks/mps_flat.fits",&nx,&ny,&lx,&ly,0);
  f0=nmt_field_flat_alloc(NX_TEST,NY_TEST,
                          DX_TEST*NX_TEST*M_PI/180,
                          DY_TEST*NY_TEST*M_PI/180,
                          msk,0,&mpt,0,NULL,0,NULL,NULL,0,0,1E-10,0,0,0);
  mpt=fs_read_flat_map("test/benchmarks/mps_flat.fits",&nx,&ny,&lx,&ly,0);
  f0l=nmt_field_flat_alloc(NX_TEST,NY_TEST,
                           DX_TEST*NX_TEST*M_PI/180,
                           DY_TEST*NY_TEST*M_PI/180,
                           msk,0,&mpt,0,NULL,0,NULL,NULL,0,0,1E-10,0,1,0);
  f0e=nmt_field_flat_alloc(NX_TEST,NY_TEST,
                           DX_TEST*NX_TEST*M_PI/180,
                           DY_TEST*NY_TEST*M_PI/180,
                           msk,0,NULL,0,NULL,0,NULL,NULL,0,0,1E-10,0,1,1);

  set_error_policy(THROW_ON_ERROR);
  try { nmt_compute_deprojection_bias_flat(f0l,f0,NULL,0,0,0,0,0,NULL,NULL,NULL); }
  try { nmt_compute_deprojection_bias_flat(f0l,f0l,NULL,0,0,0,0,0,NULL,NULL,NULL); }
  try { nmt_compute_deprojection_bias_flat(f0l,f0e,NULL,0,0,0,0,0,NULL,NULL,NULL); }
  try { nmt_compute_deprojection_bias_flat(f0e,f0e,NULL,0,0,0,0,0,NULL,NULL,NULL); }
  try { nmt_compute_coupled_cell_flat(f0e,f0e,NULL,NULL,0,0,0,0); }
  set_error_policy(EXIT_ON_ERROR);

  nmt_field_flat_free(f0);
  nmt_field_flat_free(f0l);
  nmt_field_flat_free(f0e);
  free(msk);
  free(mpt);
}

CTEST(nmt,master_flat_errors) {
  int ii,nx,ny;
  double lx,ly;
  nmt_binning_scheme_flat *bin;
  int nell=25;
  double dell=20.;
  double *larr=my_malloc((nell+1)*sizeof(double));
  double *msk=fs_read_flat_map("test/benchmarks/msk_flat.fits",&nx,&ny,&lx,&ly,0);
  double **mps=my_malloc(3*sizeof(double *));
  nmt_workspace_flat *w=NULL,*wb=NULL;
  
  for(ii=0;ii<3;ii++)
    mps[ii]=fs_read_flat_map("test/benchmarks/mps_flat.fits",&nx,&ny,&lx,&ly,ii);

  for(ii=0;ii<=nell;ii++)
    larr[ii]=ii*dell+2;

  nmt_field_flat *f0=nmt_field_flat_alloc(NX_TEST,NY_TEST,
					  DX_TEST*NX_TEST*M_PI/180,
					  DY_TEST*NY_TEST*M_PI/180,
					  msk,0,mps,0,NULL,0,NULL,NULL,0,0,1E-10,0,0,0);
  nmt_field_flat *f0b=nmt_field_flat_alloc(NX_TEST/2,NY_TEST/2,
					   DX_TEST*(NX_TEST/2)*M_PI/180,
					   DY_TEST*(NY_TEST/2)*M_PI/180,
					   msk,0,mps,0,NULL,0,NULL,NULL,0,0,1E-10,0,0,0);
  
  set_error_policy(THROW_ON_ERROR);

  //Read from non-existent file
  try { w=nmt_workspace_flat_read_fits("nofile"); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(w);

  //Mismatching resolutions
  bin=nmt_bins_flat_create(nell,larr,&(larr[1]));
  try { w=nmt_compute_coupling_matrix_flat(f0,f0b,bin,1,-1,1,-1,0); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(w);
  //Wrong fields for teb
  try { w=nmt_compute_coupling_matrix_flat(f0,f0,bin,1,-1,1,-1,1); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(w);
  //Try through nmt_compute_power_spectra
  try { w=nmt_compute_power_spectra_flat(f0,f0b,bin,1,-1,1,-1,NULL,NULL,0,NULL,NULL,NULL); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(w); 
  //Try through nmt_compute_coupled_cell_flat
  try { nmt_compute_coupled_cell_flat(f0,f0b,bin,NULL,1,-1,1,-1); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(w);
  //nmt_compute_power_spectra with mis-matching input workspace
  w=nmt_compute_coupling_matrix_flat(f0,f0,bin,1,-1,1,-1,0);
  nmt_bins_flat_free(bin);
  bin=nmt_bins_flat_create(nell/2,larr,&(larr[1]));
  try { wb=nmt_compute_power_spectra_flat(f0b,f0b,bin,1,-1,1,-1,w,NULL,0,NULL,NULL,NULL); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(wb);
  nmt_workspace_flat_free(w);
  set_error_policy(EXIT_ON_ERROR);

  nmt_bins_flat_free(bin);
  nmt_field_flat_free(f0);
  nmt_field_flat_free(f0b);
  free(msk);
  for(ii=0;ii<3;ii++)
    free(mps[ii]);
  free(mps);
  free(larr);
}
