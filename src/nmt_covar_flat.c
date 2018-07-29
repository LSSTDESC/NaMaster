#include "utils.h"

static fcomplex *product_and_transform(nmt_flatsky_info *fs,flouble *m1,flouble *m2)
{
  flouble  *m12=dftw_malloc(fs->npix*sizeof(flouble));
  fs_map_product(fs,m1,m2,m12);
  fcomplex *cm12=dftw_malloc(fs->ny*(fs->nx/2+1)*sizeof(fcomplex));
  fs_map2alm(fs,1,0,&m12,&cm12);

  dftw_free(m12);
  return cm12;
}
  
static nmt_binning_scheme_flat *nmt_bins_copy(nmt_binning_scheme_flat *b_or)
{
  nmt_binning_scheme_flat *b=my_malloc(sizeof(nmt_binning_scheme_flat));
  b->n_bands=b_or->n_bands;
  b->ell_0_list=my_malloc(b->n_bands*sizeof(flouble));
  memcpy(b->ell_0_list,b_or->ell_0_list,b->n_bands*sizeof(flouble));
  b->ell_f_list=my_malloc(b->n_bands*sizeof(flouble));
  memcpy(b->ell_f_list,b_or->ell_f_list,b->n_bands*sizeof(flouble));
  
  return b;
}

nmt_covar_workspace_flat *nmt_covar_workspace_flat_init(nmt_workspace_flat *wa,nmt_workspace_flat *wb)
{
  int ii;
  
  if((wa->fs->nx!=wb->fs->nx) || (wa->fs->ny!=wa->fs->ny))
    report_error(1,"Can't compute covariance for fields with different resolutions\n");
  nmt_flatsky_info *fs=wa->fs;
  if((wa->ncls!=1) || (wb->ncls!=1))
    report_error(1,"Gaussian covariance only implemented for spin-0 fields\n");
  if(wa->bin->n_bands!=wb->bin->n_bands)
    report_error(1,"Can't compute covariance for different binning schemes");

  nmt_covar_workspace_flat *cw=my_malloc(sizeof(nmt_covar_workspace_flat));

  cw->ncls_a=wa->ncls;
  cw->ncls_b=wb->ncls;
  cw->bin=nmt_bins_copy(wa->bin);
  cw->xi_1122=my_malloc(cw->ncls_a*cw->bin->n_bands*sizeof(flouble *));
  cw->xi_1221=my_malloc(cw->ncls_a*cw->bin->n_bands*sizeof(flouble *));
  for(ii=0;ii<cw->ncls_a*cw->bin->n_bands;ii++) {
    cw->xi_1122[ii]=my_calloc(cw->ncls_b*cw->bin->n_bands,sizeof(flouble));
    cw->xi_1221[ii]=my_calloc(cw->ncls_b*cw->bin->n_bands,sizeof(flouble));
  }
  cw->coupling_binned_a=gsl_matrix_alloc(cw->ncls_a*cw->bin->n_bands,cw->ncls_a*cw->bin->n_bands);
  gsl_matrix_memcpy(cw->coupling_binned_a,wa->coupling_matrix_binned_gsl);
  cw->coupling_binned_b=gsl_matrix_alloc(cw->ncls_b*cw->bin->n_bands,cw->ncls_b*cw->bin->n_bands);
  gsl_matrix_memcpy(cw->coupling_binned_b,wb->coupling_matrix_binned_gsl);
  
  cw->coupling_binned_perm_a=gsl_permutation_alloc(cw->ncls_a*cw->bin->n_bands);
  gsl_permutation_memcpy(cw->coupling_binned_perm_a,wa->coupling_matrix_perm);
  cw->coupling_binned_perm_b=gsl_permutation_alloc(cw->ncls_b*cw->bin->n_bands);
  gsl_permutation_memcpy(cw->coupling_binned_perm_b,wb->coupling_matrix_perm);

  //Multiply masks and Fourier-transform
  fcomplex *cm_a1b1=product_and_transform(fs,wa->mask1,wb->mask1);
  fcomplex *cm_a1b2=product_and_transform(fs,wa->mask1,wb->mask2);
  fcomplex *cm_a2b1=product_and_transform(fs,wa->mask2,wb->mask1);
  fcomplex *cm_a2b2=product_and_transform(fs,wa->mask2,wb->mask2);

  //Compute squared-mask power spectra
  int *i_band,*i_band_nocut;
  flouble *cl_mask_1122=my_malloc(fs->npix*sizeof(double));
  flouble *cl_mask_1221=my_malloc(fs->npix*sizeof(double));
  i_band=my_malloc(fs->npix*sizeof(int));
  i_band_nocut=my_malloc(fs->npix*sizeof(int));

#pragma omp parallel default(none)			\
  shared(cw,fs,cm_a1b1,cm_a1b2,cm_a2b1,cm_a2b2)		\
  shared(i_band_nocut,i_band,cl_mask_1122,cl_mask_1221)
  {
    flouble dkx=2*M_PI/fs->lx;
    flouble dky=2*M_PI/fs->ly;
    int iy1,ix1;
    
#pragma omp for
    for(iy1=0;iy1<fs->ny;iy1++) {
      flouble ky;
      int ik=0;
      if(2*iy1<=fs->ny)
	ky=iy1*dky;
      else
	ky=-(fs->ny-iy1)*dky;
      for(ix1=0;ix1<fs->nx;ix1++) {
	flouble kx,kmod;
	int ix_here,index_here,index;
	index=ix1+fs->nx*iy1;
	if(2*ix1<=fs->nx) {
	  kx=ix1*dkx;
	  ix_here=ix1;
	}
	else {
	  kx=-(fs->nx-ix1)*dkx;
	  ix_here=fs->nx-ix1;
	}
	index_here=ix_here+(fs->nx/2+1)*iy1;
	
	cl_mask_1122[index]=(creal(cm_a1b1[index_here])*creal(cm_a2b2[index_here])+
			     cimag(cm_a1b1[index_here])*cimag(cm_a2b2[index_here]));
	cl_mask_1221[index]=(creal(cm_a1b2[index_here])*creal(cm_a2b1[index_here])+
			     cimag(cm_a1b2[index_here])*cimag(cm_a2b1[index_here]));

	kmod=sqrt(kx*kx+ky*ky);
	ik=nmt_bins_flat_search_fast(cw->bin,kmod,ik);
	if(ik>=0)
	  i_band[index]=ik;
	else
	  i_band[index]=-1;
	i_band_nocut[index]=ik;
      }
    } //end omp for
  } //end omp parallel

  dftw_free(cm_a1b1);
  dftw_free(cm_a1b2);
  dftw_free(cm_a2b1);
  dftw_free(cm_a2b2);

  //Compute Xis
#pragma omp parallel default(none)			\
  shared(fs,i_band,cw,cl_mask_1122,cl_mask_1221)
  {
    int iy1,ix1,ix2,iy2;

    flouble **xi_1122=my_malloc(cw->bin->n_bands*cw->ncls_a*sizeof(flouble *));
    flouble **xi_1221=my_malloc(cw->bin->n_bands*cw->ncls_a*sizeof(flouble *));
    for(iy1=0;iy1<cw->bin->n_bands*cw->ncls_a;iy1++) {
      xi_1122[iy1]=my_calloc(cw->bin->n_bands*cw->ncls_a,sizeof(flouble));
      xi_1221[iy1]=my_calloc(cw->bin->n_bands*cw->ncls_a,sizeof(flouble));
    }

#pragma omp for
    for(iy1=0;iy1<fs->ny;iy1++) {
      for(ix1=0;ix1<fs->nx;ix1++) {
	int index1=ix1+fs->nx*iy1;
	int ik1=i_band[index1];
	if(ik1>=0) {
	  for(iy2=0;iy2<fs->ny;iy2++) {
	    for(ix2=0;ix2<fs->nx;ix2++) {
	      int index,index2=ix2+fs->nx*iy2;
	      int ik2=i_band[index2];
	      int iy=iy1-iy2;
	      int ix=ix1-ix2;
	      if(iy<0) iy+=fs->ny;
	      if(ix<0) ix+=fs->nx;
	      index=ix+fs->nx*iy;

	      if(ik2>=0) {
		xi_1122[ik1+0][ik2+0]+=cl_mask_1122[index];
		xi_1221[ik1+0][ik2+0]+=cl_mask_1221[index];
	      }
	    }
	  }
	}
      }
    } //end omp for

#pragma omp critical
    {
      for(iy1=0;iy1<cw->ncls_a*cw->bin->n_bands;iy1++) {
	for(iy2=0;iy2<cw->ncls_b*cw->bin->n_bands;iy2++) {
	  cw->xi_1122[iy1][iy2]+=xi_1122[iy1][iy2];
	  cw->xi_1221[iy1][iy2]+=xi_1221[iy1][iy2];
	}
      }
    } //end omp critical
    for(iy1=0;iy1<cw->ncls_a*cw->bin->n_bands;iy1++) {
      free(xi_1122[iy1]);
      free(xi_1221[iy1]);
    }
    free(xi_1122);
    free(xi_1221);
  } //end omp parallel

#pragma omp parallel default(none)		\
  shared(fs,cw,wa,wb)
  {
    int ib1;
    flouble fac_norm=4*M_PI*M_PI/(fs->lx*fs->lx*fs->ly*fs->ly);

#pragma omp for
    for(ib1=0;ib1<cw->bin->n_bands;ib1++) {
      int icl1;
      for(icl1=0;icl1<cw->ncls_a;icl1++) {
	int ib2;
	int ind1=cw->ncls_a*ib1+icl1;
	for(ib2=0;ib2<cw->bin->n_bands;ib2++) {
	  int icl2;
	  for(icl2=0;icl2<cw->ncls_b;icl2++) {
	    int ind2=cw->ncls_b*ib2+icl2;
	    flouble norm;
	    if(wa->n_cells[ib1]*wb->n_cells[ib2]>0)
	      norm=fac_norm/(wa->n_cells[ib1]*wb->n_cells[ib2]);
	    else
	      norm=0;
	    cw->xi_1122[ind1][ind2]*=norm;
	    cw->xi_1221[ind1][ind2]*=norm;
	  }
	}
      }
    } //end omp for
  } //end omp parallel
    
  free(i_band);
  free(i_band_nocut);
  free(cl_mask_1122);
  free(cl_mask_1221);

  return cw;
}

void nmt_covar_workspace_flat_free(nmt_covar_workspace_flat *cw)
{
  int ii;
  gsl_permutation_free(cw->coupling_binned_perm_b);
  gsl_permutation_free(cw->coupling_binned_perm_a);
  gsl_matrix_free(cw->coupling_binned_b);
  gsl_matrix_free(cw->coupling_binned_a);
  for(ii=0;ii<cw->ncls_a*cw->bin->n_bands;ii++) {
    free(cw->xi_1122[ii]);
    free(cw->xi_1221[ii]);
  }
  free(cw->xi_1122);
  free(cw->xi_1221);
  nmt_bins_flat_free(cw->bin);

  free(cw);
}

void nmt_compute_gaussian_covariance_flat(nmt_covar_workspace_flat *cw,
					  int nl,flouble *larr,flouble *cla1b1,flouble *cla1b2,
					  flouble *cla2b1,flouble *cla2b2,flouble *covar_out)
{
  //Compute binned spectra
  flouble *cbla1b1=my_malloc(cw->bin->n_bands*sizeof(flouble));
  flouble *cbla1b2=my_malloc(cw->bin->n_bands*sizeof(flouble));
  flouble *cbla2b1=my_malloc(cw->bin->n_bands*sizeof(flouble));
  flouble *cbla2b2=my_malloc(cw->bin->n_bands*sizeof(flouble));
  nmt_bin_cls_flat(cw->bin,nl,larr,&cla1b1,&cbla1b1,1);
  nmt_bin_cls_flat(cw->bin,nl,larr,&cla1b2,&cbla1b2,1);
  nmt_bin_cls_flat(cw->bin,nl,larr,&cla2b1,&cbla2b1,1);
  nmt_bin_cls_flat(cw->bin,nl,larr,&cla2b2,&cbla2b2,1);

  //Convolve with Xi
  gsl_matrix *covar_binned=gsl_matrix_alloc(cw->ncls_a*cw->bin->n_bands,cw->ncls_b*cw->bin->n_bands);
  int iba;
  for(iba=0;iba<cw->bin->n_bands;iba++) {
    int icl_a;
    for(icl_a=0;icl_a<cw->ncls_a;icl_a++) {
      int index_a=cw->ncls_a*iba+icl_a;
      int icl_b;
      for(icl_b=0;icl_b<cw->ncls_b;icl_b++) {
	int ibb;
	for(ibb=0;ibb<cw->bin->n_bands;ibb++) {
	  int index_b=cw->ncls_b*ibb+icl_b;
	  double xi_1122=cw->xi_1122[index_a][index_b];
	  double xi_1221=cw->xi_1221[index_a][index_b];
	  double fac_1122=0.5*(cbla1b1[index_a]*cbla2b2[index_b]+cbla1b1[index_b]*cbla2b2[index_a]);
	  double fac_1221=0.5*(cbla1b2[index_a]*cbla2b1[index_b]+cbla1b2[index_b]*cbla2b1[index_a]);
	  gsl_matrix_set(covar_binned,index_a,index_b,xi_1122*fac_1122+xi_1221*fac_1221);
	}
      }
    }
  }

  //Sandwich with 
  gsl_matrix *covar_out_g =gsl_matrix_alloc(cw->ncls_a*cw->bin->n_bands,cw->ncls_b*cw->bin->n_bands);
  gsl_matrix *mat_tmp     =gsl_matrix_alloc(cw->ncls_a*cw->bin->n_bands,cw->ncls_b*cw->bin->n_bands);
  gsl_matrix *inverse_a   =gsl_matrix_alloc(cw->ncls_a*cw->bin->n_bands,cw->ncls_a*cw->bin->n_bands);
  gsl_matrix *inverse_b   =gsl_matrix_alloc(cw->ncls_b*cw->bin->n_bands,cw->ncls_b*cw->bin->n_bands);
  gsl_linalg_LU_invert(cw->coupling_binned_b,cw->coupling_binned_perm_b,inverse_b); //M_b^-1
  gsl_linalg_LU_invert(cw->coupling_binned_a,cw->coupling_binned_perm_a,inverse_a); //M_a^-1
  gsl_blas_dgemm(CblasNoTrans,CblasTrans  ,1,covar_binned,inverse_b,0,mat_tmp    ); //tmp = C * M_b^-1^T
  gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1,inverse_a   ,mat_tmp  ,0,covar_out_g); //C' = M_a^-1 * C * M_b^-1^T

  //Flatten
  int ii;
  long elem=0;
  for(ii=0;ii<cw->ncls_a*cw->bin->n_bands;ii++) {
    int jj;
    for(jj=0;jj<cw->ncls_b*cw->bin->n_bands;jj++) {
      covar_out[elem]=gsl_matrix_get(covar_out_g,ii,jj);
      elem++;
    }
  }

  free(cbla1b1);
  free(cbla1b2);
  free(cbla2b1);
  free(cbla2b2);
  gsl_matrix_free(mat_tmp);
  gsl_matrix_free(inverse_a);
  gsl_matrix_free(inverse_b);
  gsl_matrix_free(covar_out_g);
  gsl_matrix_free(covar_binned);
}

void nmt_covar_workspace_flat_write(nmt_covar_workspace_flat *cw,char *fname)
{
  int ii;
  FILE *fo=my_fopen(fname,"wb");

  my_fwrite(&(cw->ncls_a),sizeof(int),1,fo);
  my_fwrite(&(cw->ncls_b),sizeof(int),1,fo);
  
  my_fwrite(&(cw->bin->n_bands),sizeof(int),1,fo);
  my_fwrite(cw->bin->ell_0_list,sizeof(flouble),cw->bin->n_bands,fo);
  my_fwrite(cw->bin->ell_f_list,sizeof(flouble),cw->bin->n_bands,fo);

  for(ii=0;ii<cw->ncls_a*cw->bin->n_bands;ii++)
    my_fwrite(cw->xi_1122[ii],sizeof(flouble),cw->ncls_b*cw->bin->n_bands,fo);
  for(ii=0;ii<cw->ncls_a*cw->bin->n_bands;ii++)
    my_fwrite(cw->xi_1221[ii],sizeof(flouble),cw->ncls_b*cw->bin->n_bands,fo);

  gsl_matrix_fwrite(fo,cw->coupling_binned_a);
  gsl_matrix_fwrite(fo,cw->coupling_binned_b);
  gsl_permutation_fwrite(fo,cw->coupling_binned_perm_a);
  gsl_permutation_fwrite(fo,cw->coupling_binned_perm_b);
  
  fclose(fo);
}

nmt_covar_workspace_flat *nmt_covar_workspace_flat_read(char *fname)
{
  int ii;
  nmt_covar_workspace_flat *cw=my_malloc(sizeof(nmt_covar_workspace));
  FILE *fi=my_fopen(fname,"rb");

  my_fread(&(cw->ncls_a),sizeof(int),1,fi);
  my_fread(&(cw->ncls_b),sizeof(int),1,fi);

  cw->bin=my_malloc(sizeof(nmt_binning_scheme_flat));
  my_fread(&(cw->bin->n_bands),sizeof(int),1,fi);
  cw->bin->ell_0_list=my_malloc(cw->bin->n_bands*sizeof(flouble));
  cw->bin->ell_f_list=my_malloc(cw->bin->n_bands*sizeof(flouble));
  my_fread(cw->bin->ell_0_list,sizeof(flouble),cw->bin->n_bands,fi);
  my_fread(cw->bin->ell_f_list,sizeof(flouble),cw->bin->n_bands,fi);

  cw->xi_1122=my_malloc(cw->ncls_a*cw->bin->n_bands*sizeof(flouble *));
  for(ii=0;ii<cw->ncls_a*cw->bin->n_bands;ii++) {
    cw->xi_1122[ii]=my_malloc(cw->ncls_b*cw->bin->n_bands*sizeof(flouble));
    my_fread(cw->xi_1122[ii],sizeof(flouble),cw->ncls_b*cw->bin->n_bands,fi);
  }
  cw->xi_1221=my_malloc(cw->ncls_a*cw->bin->n_bands*sizeof(flouble *));
  for(ii=0;ii<cw->ncls_a*cw->bin->n_bands;ii++) {
    cw->xi_1221[ii]=my_malloc(cw->ncls_b*cw->bin->n_bands*sizeof(flouble));
    my_fread(cw->xi_1221[ii],sizeof(flouble),cw->ncls_b*cw->bin->n_bands,fi);
  }

  cw->coupling_binned_a=gsl_matrix_alloc(cw->ncls_a*cw->bin->n_bands,cw->ncls_a*cw->bin->n_bands);
  gsl_matrix_fread(fi,cw->coupling_binned_a);
  cw->coupling_binned_b=gsl_matrix_alloc(cw->ncls_b*cw->bin->n_bands,cw->ncls_b*cw->bin->n_bands);
  gsl_matrix_fread(fi,cw->coupling_binned_b);
  cw->coupling_binned_perm_a=gsl_permutation_alloc(cw->ncls_a*cw->bin->n_bands);
  gsl_permutation_fread(fi,cw->coupling_binned_perm_a);
  cw->coupling_binned_perm_b=gsl_permutation_alloc(cw->ncls_b*cw->bin->n_bands);
  gsl_permutation_fread(fi,cw->coupling_binned_perm_b);

  fclose(fi);
  
  return cw;
}
