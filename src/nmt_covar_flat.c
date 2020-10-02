#include "config.h"
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

nmt_covar_workspace_flat *nmt_covar_workspace_flat_init(nmt_field_flat *fla1,nmt_field_flat *fla2,
							nmt_binning_scheme_flat *ba,
							nmt_field_flat *flb1,nmt_field_flat *flb2,
							nmt_binning_scheme_flat *bb)
{
  int ii;
  
  if((fla1->fs->nx!=fla2->fs->nx) || (fla1->fs->ny!=fla2->fs->ny) ||
     (fla1->fs->nx!=flb1->fs->nx) || (fla1->fs->ny!=flb1->fs->ny) ||     
     (fla1->fs->nx!=flb2->fs->nx) || (fla1->fs->ny!=flb2->fs->ny))
    report_error(NMT_ERROR_COVAR,"Can't compute covariance for fields with different resolutions\n");
  nmt_flatsky_info *fs=fla1->fs;
  if(ba->n_bands!=bb->n_bands)
    report_error(NMT_ERROR_COVAR,"Can't compute covariance for different binning schemes\n");

  nmt_covar_workspace_flat *cw=my_malloc(sizeof(nmt_covar_workspace_flat));

  cw->bin=nmt_bins_copy(ba);
  cw->xi00_1122=my_malloc(cw->bin->n_bands*sizeof(flouble *));
  cw->xi00_1221=my_malloc(cw->bin->n_bands*sizeof(flouble *));
  cw->xi02_1122=my_malloc(cw->bin->n_bands*sizeof(flouble *));
  cw->xi02_1221=my_malloc(cw->bin->n_bands*sizeof(flouble *));
  cw->xi22p_1122=my_malloc(cw->bin->n_bands*sizeof(flouble *));
  cw->xi22p_1221=my_malloc(cw->bin->n_bands*sizeof(flouble *));
  cw->xi22m_1122=my_malloc(cw->bin->n_bands*sizeof(flouble *));
  cw->xi22m_1221=my_malloc(cw->bin->n_bands*sizeof(flouble *));
  for(ii=0;ii<cw->bin->n_bands;ii++) {
    cw->xi00_1122[ii]=my_calloc(cw->bin->n_bands,sizeof(flouble));
    cw->xi00_1221[ii]=my_calloc(cw->bin->n_bands,sizeof(flouble));
    cw->xi02_1122[ii]=my_calloc(cw->bin->n_bands,sizeof(flouble));
    cw->xi02_1221[ii]=my_calloc(cw->bin->n_bands,sizeof(flouble));
    cw->xi22p_1122[ii]=my_calloc(cw->bin->n_bands,sizeof(flouble));
    cw->xi22p_1221[ii]=my_calloc(cw->bin->n_bands,sizeof(flouble));
    cw->xi22m_1122[ii]=my_calloc(cw->bin->n_bands,sizeof(flouble));
    cw->xi22m_1221[ii]=my_calloc(cw->bin->n_bands,sizeof(flouble));
  }

  int *n_cells=my_calloc(cw->bin->n_bands,sizeof(int));
  
  //Multiply masks and Fourier-transform
  fcomplex *cm_a1b1=product_and_transform(fs,fla1->mask,flb1->mask);
  fcomplex *cm_a1b2=product_and_transform(fs,fla1->mask,flb2->mask);
  fcomplex *cm_a2b1=product_and_transform(fs,fla2->mask,flb1->mask);
  fcomplex *cm_a2b2=product_and_transform(fs,fla2->mask,flb2->mask);

  //Compute squared-mask power spectra
  int *i_band,*i_band_nocut;
  flouble *cl_mask_1122=my_malloc(fs->npix*sizeof(double));
  flouble *cl_mask_1221=my_malloc(fs->npix*sizeof(double));
  flouble *cosarr=dftw_malloc(fs->npix*sizeof(double));
  flouble *sinarr=dftw_malloc(fs->npix*sizeof(double));
  i_band=my_malloc(fs->npix*sizeof(int));
  i_band_nocut=my_malloc(fs->npix*sizeof(int));

#pragma omp parallel default(none)			\
  shared(cw,fs,cm_a1b1,cm_a1b2,cm_a2b1,cm_a2b2,n_cells)	\
  shared(i_band_nocut,i_band,cl_mask_1122,cl_mask_1221)	\
  shared(cosarr,sinarr)
  {
    flouble dkx=2*M_PI/fs->lx;
    flouble dky=2*M_PI/fs->ly;
    int *n_cells_thr=my_calloc(cw->bin->n_bands,sizeof(int));
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
	flouble kx,kmod,c,s;
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
	if(ik>=0) {
	  i_band[index]=ik;
	  n_cells_thr[ik]++;
	}
	else
	  i_band[index]=-1;
	i_band_nocut[index]=ik;
	if(kmod>0) {
	  c=kx/kmod;
	  s=ky/kmod;
	}
	else {
	  c=1.;
	  s=0.;
	}
	cosarr[index]=c*c-s*s;
	sinarr[index]=2*s*c;
      }
    } //end omp for
    
#pragma omp critical
    {
      for(iy1=0;iy1<cw->bin->n_bands;iy1++)
	n_cells[iy1]+=n_cells_thr[iy1];
    } //end omp critical
    free(n_cells_thr);
  } //end omp parallel

  dftw_free(cm_a1b1);
  dftw_free(cm_a1b2);
  dftw_free(cm_a2b1);
  dftw_free(cm_a2b2);

  //Compute Xis
#pragma omp parallel default(none)			\
  shared(fs,i_band,cw,cl_mask_1122,cl_mask_1221)	\
  shared(cosarr,sinarr)
  {
    int iy1,ix1,ix2,iy2;

    flouble **xi00_1122=my_malloc(cw->bin->n_bands*sizeof(flouble *));
    flouble **xi00_1221=my_malloc(cw->bin->n_bands*sizeof(flouble *));
    flouble **xi02_1122=my_malloc(cw->bin->n_bands*sizeof(flouble *));
    flouble **xi02_1221=my_malloc(cw->bin->n_bands*sizeof(flouble *));
    flouble **xi22p_1122=my_malloc(cw->bin->n_bands*sizeof(flouble *));
    flouble **xi22p_1221=my_malloc(cw->bin->n_bands*sizeof(flouble *));
    flouble **xi22m_1122=my_malloc(cw->bin->n_bands*sizeof(flouble *));
    flouble **xi22m_1221=my_malloc(cw->bin->n_bands*sizeof(flouble *));
    for(iy1=0;iy1<cw->bin->n_bands;iy1++) {
      xi00_1122[iy1]=my_calloc(cw->bin->n_bands,sizeof(flouble));
      xi00_1221[iy1]=my_calloc(cw->bin->n_bands,sizeof(flouble));
      xi02_1122[iy1]=my_calloc(cw->bin->n_bands,sizeof(flouble));
      xi02_1221[iy1]=my_calloc(cw->bin->n_bands,sizeof(flouble));
      xi22p_1122[iy1]=my_calloc(cw->bin->n_bands,sizeof(flouble));
      xi22p_1221[iy1]=my_calloc(cw->bin->n_bands,sizeof(flouble));
      xi22m_1122[iy1]=my_calloc(cw->bin->n_bands,sizeof(flouble));
      xi22m_1221[iy1]=my_calloc(cw->bin->n_bands,sizeof(flouble));
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
	      flouble cdiff=1,sdiff=0;
	      int iy=iy1-iy2;
	      int ix=ix1-ix2;
	      if(iy<0) iy+=fs->ny;
	      if(ix<0) ix+=fs->nx;
	      index=ix+fs->nx*iy;

	      if(ik2>=0) {
		double clm1122=cl_mask_1122[index];
		double clm1221=cl_mask_1221[index];
		cdiff=cosarr[index1]*cosarr[index2]+sinarr[index1]*sinarr[index2];
		sdiff=sinarr[index1]*cosarr[index2]-cosarr[index1]*sinarr[index2];
		xi00_1122[ik1][ik2]+=clm1122;
		xi00_1221[ik1][ik2]+=clm1221;
		xi02_1122[ik1][ik2]+=clm1122*cdiff;
		xi02_1221[ik1][ik2]+=clm1221*cdiff;
		xi22p_1122[ik1][ik2]+=clm1122*cdiff*cdiff;
		xi22p_1221[ik1][ik2]+=clm1221*cdiff*cdiff;
		xi22m_1122[ik1][ik2]+=clm1122*sdiff*sdiff;
		xi22m_1221[ik1][ik2]+=clm1221*sdiff*sdiff;
	      }
	    }
	  }
	}
      }
    } //end omp for

#pragma omp critical
    {
      for(iy1=0;iy1<cw->bin->n_bands;iy1++) {
	for(iy2=0;iy2<cw->bin->n_bands;iy2++) {
	  cw->xi00_1122[iy1][iy2]+=xi00_1122[iy1][iy2];
	  cw->xi00_1221[iy1][iy2]+=xi00_1221[iy1][iy2];
	  cw->xi02_1122[iy1][iy2]+=xi02_1122[iy1][iy2];
	  cw->xi02_1221[iy1][iy2]+=xi02_1221[iy1][iy2];
	  cw->xi22p_1122[iy1][iy2]+=xi22p_1122[iy1][iy2];
	  cw->xi22p_1221[iy1][iy2]+=xi22p_1221[iy1][iy2];
	  cw->xi22m_1122[iy1][iy2]+=xi22m_1122[iy1][iy2];
	  cw->xi22m_1221[iy1][iy2]+=xi22m_1221[iy1][iy2];
	}
      }
    } //end omp critical
    for(iy1=0;iy1<cw->bin->n_bands;iy1++) {
      free(xi00_1122[iy1]);
      free(xi00_1221[iy1]);
      free(xi02_1122[iy1]);
      free(xi02_1221[iy1]);
      free(xi22p_1122[iy1]);
      free(xi22p_1221[iy1]);
      free(xi22m_1122[iy1]);
      free(xi22m_1221[iy1]);
    }
    free(xi00_1122);
    free(xi00_1221);
    free(xi02_1122);
    free(xi02_1221);
    free(xi22p_1122);
    free(xi22p_1221);
    free(xi22m_1122);
    free(xi22m_1221);
  } //end omp parallel

#pragma omp parallel default(none)		\
  shared(fs,cw,n_cells)
  {
    int ib1;
    flouble fac_norm=4*M_PI*M_PI/(fs->lx*fs->lx*fs->ly*fs->ly);

#pragma omp for
    for(ib1=0;ib1<cw->bin->n_bands;ib1++) {
      int ib2;
      for(ib2=0;ib2<cw->bin->n_bands;ib2++) {
	flouble norm;
	if(n_cells[ib1]*n_cells[ib2]>0)
	  norm=fac_norm/(n_cells[ib1]*n_cells[ib2]);
	else
	  norm=0;
	cw->xi00_1122[ib1][ib2]*=norm;
	cw->xi00_1221[ib1][ib2]*=norm;
	cw->xi02_1122[ib1][ib2]*=norm;
	cw->xi02_1221[ib1][ib2]*=norm;
	cw->xi22p_1122[ib1][ib2]*=norm;
	cw->xi22p_1221[ib1][ib2]*=norm;
	cw->xi22m_1122[ib1][ib2]*=norm;
	cw->xi22m_1221[ib1][ib2]*=norm;
      }
    } //end omp for
  } //end omp parallel
    
  free(i_band);
  free(i_band_nocut);
  free(cl_mask_1122);
  free(cl_mask_1221);
  dftw_free(cosarr);
  dftw_free(sinarr);
  free(n_cells);

  return cw;
}

void nmt_covar_workspace_flat_free(nmt_covar_workspace_flat *cw)
{
  int ii;
  for(ii=0;ii<cw->bin->n_bands;ii++) {
    free(cw->xi00_1122[ii]);
    free(cw->xi00_1221[ii]);
    free(cw->xi02_1122[ii]);
    free(cw->xi02_1221[ii]);
    free(cw->xi22p_1122[ii]);
    free(cw->xi22p_1221[ii]);
    free(cw->xi22m_1122[ii]);
    free(cw->xi22m_1221[ii]);
  }
  free(cw->xi00_1122);
  free(cw->xi00_1221);
  free(cw->xi02_1122);
  free(cw->xi02_1221);
  free(cw->xi22p_1122);
  free(cw->xi22p_1221);
  free(cw->xi22m_1122);
  free(cw->xi22m_1221);
  nmt_bins_flat_free(cw->bin);

  free(cw);
}

void nmt_compute_gaussian_covariance_flat(nmt_covar_workspace_flat *cw,
					  int spin_a,int spin_b,int spin_c,int spin_d,
					  nmt_workspace_flat *wa,nmt_workspace_flat *wb,
					  int nl,flouble *larr,
					  flouble **clac,flouble **clad,
					  flouble **clbc,flouble **clbd,flouble *covar_out)
{
  if((wa->bin->n_bands!=cw->bin->n_bands) || (wb->bin->n_bands!=cw->bin->n_bands))
    report_error(NMT_ERROR_COVAR,"Coupling coefficients were computed for a different binning scheme\n");

  int nmaps_a=spin_a ? 2 : 1;
  int nmaps_b=spin_b ? 2 : 1;
  int nmaps_c=spin_c ? 2 : 1;
  int nmaps_d=spin_d ? 2 : 1;
  if((wa->ncls!=nmaps_a*nmaps_b) || (wb->ncls!=nmaps_c*nmaps_d))
    report_error(NMT_ERROR_COVAR,"Input spins don't match input workspaces\n");

  //Compute binned spectra
  int i_cl;
  flouble **cblac=my_malloc(nmaps_a*nmaps_c*sizeof(flouble *));
  for(i_cl=0;i_cl<nmaps_a*nmaps_c;i_cl++)
    cblac[i_cl]=my_malloc(cw->bin->n_bands*sizeof(flouble));
  nmt_bin_cls_flat(cw->bin,nl,larr,clac,cblac,nmaps_a*nmaps_c);
  flouble **cblad=my_malloc(nmaps_a*nmaps_d*sizeof(flouble *));
  for(i_cl=0;i_cl<nmaps_a*nmaps_d;i_cl++)
    cblad[i_cl]=my_malloc(cw->bin->n_bands*sizeof(flouble));
  nmt_bin_cls_flat(cw->bin,nl,larr,clad,cblad,nmaps_a*nmaps_d);
  flouble **cblbc=my_malloc(nmaps_b*nmaps_c*sizeof(flouble *));
  for(i_cl=0;i_cl<nmaps_b*nmaps_c;i_cl++)
    cblbc[i_cl]=my_malloc(cw->bin->n_bands*sizeof(flouble));
  nmt_bin_cls_flat(cw->bin,nl,larr,clbc,cblbc,nmaps_b*nmaps_c);
  flouble **cblbd=my_malloc(nmaps_b*nmaps_d*sizeof(flouble *));
  for(i_cl=0;i_cl<nmaps_b*nmaps_d;i_cl++)
    cblbd[i_cl]=my_malloc(cw->bin->n_bands*sizeof(flouble));
  nmt_bin_cls_flat(cw->bin,nl,larr,clbd,cblbd,nmaps_b*nmaps_d);

  //Convolve with Xi
  gsl_matrix *covar_binned=gsl_matrix_alloc(wa->ncls*cw->bin->n_bands,wb->ncls*cw->bin->n_bands);
#pragma omp parallel default(none)		\
  shared(cw,spin_a,spin_b,spin_c,spin_d)        \
  shared(wa,wb,nl,larr,covar_binned)		\
  shared(nmaps_a,nmaps_b,nmaps_c,nmaps_d)	\
  shared(cblac,cblad,cblbc,cblbd)
  {
    int band_a;

#pragma omp for
    for(band_a=0;band_a<cw->bin->n_bands;band_a++) {
      int band_b;
      for(band_b=0;band_b<cw->bin->n_bands;band_b++) {
	int ia;
	double xis_1122[6]={cw->xi00_1122[band_a][band_b],cw->xi02_1122[band_a][band_b],
			    cw->xi22p_1122[band_a][band_b],cw->xi22m_1122[band_a][band_b],
			    -cw->xi22m_1122[band_a][band_b],0};
	double xis_1221[6]={cw->xi00_1221[band_a][band_b],cw->xi02_1221[band_a][band_b],
			    cw->xi22p_1221[band_a][band_b],cw->xi22m_1221[band_a][band_b],
			    -cw->xi22m_1221[band_a][band_b],0};
	for(ia=0;ia<nmaps_a;ia++) {
	  int ib;
	  for(ib=0;ib<nmaps_b;ib++) {
	    int ic;
	    int icl_a=ib+nmaps_b*ia;
	    int index_a=wa->ncls*band_a+icl_a;
	    for(ic=0;ic<nmaps_c;ic++) {
	      int id;
	      for(id=0;id<nmaps_d;id++) {
		int iap;
		int icl_b=id+nmaps_d*ic;
		int index_b=wb->ncls*band_b+icl_b;
		double cbinned=0;
		for(iap=0;iap<nmaps_a;iap++) {
		  int ibp;
		  for(ibp=0;ibp<nmaps_b;ibp++) {
		    int icp;
		    for(icp=0;icp<nmaps_c;icp++) {
		      int idp;
		      for(idp=0;idp<nmaps_d;idp++) {
			double *cl_ac=cblac[icp+nmaps_c*iap];
			double *cl_ad=cblad[idp+nmaps_d*iap];
			double *cl_bc=cblbc[icp+nmaps_c*ibp];
			double *cl_bd=cblbd[idp+nmaps_d*ibp];
			double fac_1122=0.5*(cl_ac[band_a]*cl_bd[band_b]+cl_ac[band_b]*cl_bd[band_a]);
			double fac_1221=0.5*(cl_ad[band_a]*cl_bc[band_b]+cl_ad[band_b]*cl_bc[band_a]);
			int ind_1122=cov_get_coupling_pair_index(nmaps_a,nmaps_c,nmaps_b,nmaps_d,
								 ia,iap,ic,icp,ib,ibp,id,idp);
			int ind_1221=cov_get_coupling_pair_index(nmaps_a,nmaps_d,nmaps_b,nmaps_c,
								 ia,iap,id,idp,ib,ibp,ic,icp);
			cbinned+=xis_1122[ind_1122]*fac_1122+xis_1221[ind_1221]*fac_1221;
		      }
		    }
		  }
		}
		gsl_matrix_set(covar_binned,index_a,index_b,cbinned);
	      }
	    }
	  }
	}
      }
    } //end omp for
  } //end omp parallel

  //Sandwich with inverse MCM
  gsl_matrix *covar_out_g =gsl_matrix_alloc(wa->ncls*cw->bin->n_bands,wb->ncls*cw->bin->n_bands);
  gsl_matrix *mat_tmp     =gsl_matrix_alloc(wa->ncls*cw->bin->n_bands,wb->ncls*cw->bin->n_bands);
  gsl_matrix *inverse_a   =gsl_matrix_alloc(wa->ncls*cw->bin->n_bands,wa->ncls*cw->bin->n_bands);
  gsl_matrix *inverse_b   =gsl_matrix_alloc(wb->ncls*cw->bin->n_bands,wb->ncls*cw->bin->n_bands);
  gsl_linalg_LU_invert(wb->coupling_matrix_binned_gsl,wb->coupling_matrix_perm,inverse_b); //M_b^-1
  gsl_linalg_LU_invert(wa->coupling_matrix_binned_gsl,wa->coupling_matrix_perm,inverse_a); //M_a^-1
  gsl_blas_dgemm(CblasNoTrans,CblasTrans  ,1,covar_binned,inverse_b,0,mat_tmp    ); //tmp = C * M_b^-1^T
  gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1,inverse_a   ,mat_tmp  ,0,covar_out_g); //C' = M_a^-1 * C * M_b^-1^T

  //Flatten
  int ii;
  long elem=0;
  for(ii=0;ii<wa->ncls*cw->bin->n_bands;ii++) {
    int jj;
    for(jj=0;jj<wb->ncls*cw->bin->n_bands;jj++) {
      covar_out[elem]=gsl_matrix_get(covar_out_g,ii,jj);
      elem++;
    }
  }

  for(i_cl=0;i_cl<nmaps_a*nmaps_c;i_cl++)
    free(cblac[i_cl]);
  free(cblac);
  for(i_cl=0;i_cl<nmaps_a*nmaps_d;i_cl++)
    free(cblad[i_cl]);
  free(cblad);
  for(i_cl=0;i_cl<nmaps_b*nmaps_c;i_cl++)
    free(cblbc[i_cl]);
  free(cblbc);
  for(i_cl=0;i_cl<nmaps_b*nmaps_d;i_cl++)
    free(cblbd[i_cl]);
  free(cblbd);
  gsl_matrix_free(mat_tmp);
  gsl_matrix_free(inverse_a);
  gsl_matrix_free(inverse_b);
  gsl_matrix_free(covar_out_g);
  gsl_matrix_free(covar_binned);
}
