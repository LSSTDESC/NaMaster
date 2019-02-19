#include "utils.h"

static nmt_binning_scheme *nmt_bins_copy(nmt_binning_scheme *b_or)
{
  int ii;
  nmt_binning_scheme *b=my_malloc(sizeof(nmt_binning_scheme));
  b->n_bands=b_or->n_bands;
  b->nell_list=my_malloc(b->n_bands*sizeof(int));
  memcpy(b->nell_list,b_or->nell_list,b->n_bands*sizeof(int));
  b->ell_list=my_malloc(b->n_bands*sizeof(int *));
  b->w_list=my_malloc(b->n_bands*sizeof(flouble *));
  b->f_ell=my_malloc(b->n_bands*sizeof(flouble *));
  for(ii=0;ii<b->n_bands;ii++) {
    b->ell_list[ii]=my_malloc(b->nell_list[ii]*sizeof(int));
    b->w_list[ii]=my_malloc(b->nell_list[ii]*sizeof(flouble));
    b->f_ell[ii]=my_malloc(b->nell_list[ii]*sizeof(flouble));
    memcpy(b->ell_list[ii],b_or->ell_list[ii],b->nell_list[ii]*sizeof(int));
    memcpy(b->w_list[ii],b_or->w_list[ii],b->nell_list[ii]*sizeof(flouble));
    memcpy(b->f_ell[ii],b_or->f_ell[ii],b->nell_list[ii]*sizeof(flouble));
  }
  
  return b;
}

nmt_covar_workspace *nmt_covar_workspace_init(nmt_field *fla1,nmt_field *fla2,nmt_binning_scheme *ba,
					      nmt_field *flb1,nmt_field *flb2,nmt_binning_scheme *bb,
					      int niter)
					      
{
  if(!(nmt_diff_curvedsky_info(fla1->cs,fla2->cs)) || !(nmt_diff_curvedsky_info(fla1->cs,flb1->cs)) ||
     !(nmt_diff_curvedsky_info(fla1->cs,flb2->cs)) || (ba->ell_max!=bb->ell_max))
    report_error(NMT_ERROR_COVAR,"Can't compute covariance for fields with different resolutions\n");
  if(fla1->pol || fla2->pol || flb1->pol || flb2->pol)
    report_error(NMT_ERROR_COVAR,"Gaussian covariance only implemented for spin-0 fields\n");

  nmt_covar_workspace *cw=my_malloc(sizeof(nmt_covar_workspace));
  int ii;
  int npix=fla1->cs->npix;
  flouble *mask_a1b1=my_malloc(npix*sizeof(flouble));
  flouble *mask_a1b2=my_malloc(npix*sizeof(flouble));
  flouble *mask_a2b1=my_malloc(npix*sizeof(flouble));
  flouble *mask_a2b2=my_malloc(npix*sizeof(flouble));

  cw->cs=nmt_curvedsky_info_copy(fla1->cs);
  cw->lmax_a=ba->ell_max;
  cw->lmax_b=bb->ell_max;
  cw->ncls_a=fla1->nmaps*fla2->nmaps;
  cw->ncls_b=flb1->nmaps*flb2->nmaps;
  cw->bin_a=nmt_bins_copy(ba);
  cw->bin_b=nmt_bins_copy(bb);
  flouble *cl_mask_1122=my_malloc((cw->lmax_a+1)*sizeof(flouble));
  flouble *cl_mask_1221=my_malloc((cw->lmax_a+1)*sizeof(flouble));
  cw->xi_1122=my_malloc(cw->ncls_a*(cw->lmax_a+1)*sizeof(flouble *));
  cw->xi_1221=my_malloc(cw->ncls_a*(cw->lmax_a+1)*sizeof(flouble *));
  for(ii=0;ii<cw->ncls_a*(cw->lmax_a+1);ii++) {
    cw->xi_1122[ii]=my_malloc(cw->ncls_b*(cw->lmax_a+1)*sizeof(flouble));
    cw->xi_1221[ii]=my_malloc(cw->ncls_b*(cw->lmax_a+1)*sizeof(flouble));
  }
  
  he_map_product(cw->cs,fla1->mask,flb1->mask,mask_a1b1);
  he_map_product(cw->cs,fla1->mask,flb2->mask,mask_a1b2);
  he_map_product(cw->cs,fla2->mask,flb1->mask,mask_a2b1);
  he_map_product(cw->cs,fla2->mask,flb2->mask,mask_a2b2);
  he_anafast(&mask_a1b1,&mask_a2b2,0,0,&cl_mask_1122,cw->cs,cw->lmax_a,niter);
  he_anafast(&mask_a1b2,&mask_a2b1,0,0,&cl_mask_1221,cw->cs,cw->lmax_a,niter);
  free(mask_a1b1); free(mask_a1b2); free(mask_a2b1); free(mask_a2b2);
  for(ii=0;ii<=cw->lmax_a;ii++) {
    cl_mask_1122[ii]*=(ii+0.5)/(2*M_PI);
    cl_mask_1221[ii]*=(ii+0.5)/(2*M_PI);
  }

#pragma omp parallel default(none)		\
  shared(cw,cl_mask_1122,cl_mask_1221)
  {
    int ll2,ll3;
    double *wigner_00=NULL;
    int lstart=0;
    
    wigner_00=my_malloc(2*(cw->lmax_a+1)*sizeof(double));
    
    if(cw->ncls_a>1)
      lstart=2;

#pragma omp for schedule(dynamic)
    for(ll2=lstart;ll2<=cw->lmax_a;ll2++) {
      for(ll3=lstart;ll3<=cw->lmax_a;ll3++) {
	int jj,l1,lmin_here,lmax_here;
	int lmin_here_00=0,lmax_here_00=2*(cw->lmax_a+1)+1;
	flouble xi_1122=0,xi_1221=0;

	drc3jj(ll2,ll3,0,0,&lmin_here_00,&lmax_here_00,wigner_00,2*(cw->lmax_a+1));

	lmin_here=lmin_here_00;
	lmax_here=lmax_here_00;

	for(l1=lmin_here;l1<=lmax_here;l1++) {
	  if(l1<=cw->lmax_a) {
	    flouble wfac_1122,wfac_1221;
	    int j00=l1-lmin_here_00;

	    wfac_1122=cl_mask_1122[l1]*wigner_00[j00]*wigner_00[j00];
	    wfac_1221=cl_mask_1221[l1]*wigner_00[j00]*wigner_00[j00];

	    xi_1122+=wfac_1122;
	    xi_1221+=wfac_1221;
	  }
	}

	cw->xi_1122[ll2+0][ll3+0]=xi_1122;
	cw->xi_1221[ll2+0][ll3+0]=xi_1221;
      }
    } //end omp for
    free(wigner_00);
  } //end omp parallel

  free(cl_mask_1122);
  free(cl_mask_1221);
  return cw;
}

void nmt_covar_workspace_free(nmt_covar_workspace *cw)
{
  int ii;
  free(cw->cs);
  for(ii=0;ii<cw->ncls_a*(cw->lmax_a+1);ii++) {
    free(cw->xi_1122[ii]);
    free(cw->xi_1221[ii]);
  }
  nmt_bins_free(cw->bin_a);
  nmt_bins_free(cw->bin_b);
  free(cw);
}

void  nmt_compute_gaussian_covariance(nmt_covar_workspace *cw,nmt_workspace *wa,nmt_workspace *wb,
				      flouble *cla1b1,flouble *cla1b2,flouble *cla2b1,flouble *cla2b2,
				      flouble *covar_out)
{
  int icl_a;
  gsl_matrix *covar_binned=gsl_matrix_alloc(cw->ncls_a*cw->bin_a->n_bands,cw->ncls_b*cw->bin_b->n_bands);
  for(icl_a=0;icl_a<cw->ncls_a;icl_a++) {
    int icl_b;
    for(icl_b=0;icl_b<cw->ncls_b;icl_b++) {
      int iba;
      for(iba=0;iba<cw->bin_a->n_bands;iba++) {
	int ibb;
	for(ibb=0;ibb<cw->bin_b->n_bands;ibb++) {
	  double cbinned=0;
	  int ila;
	  for(ila=0;ila<cw->bin_a->nell_list[iba];ila++) {
	    int ilb;
	    int la=cw->bin_a->ell_list[iba][ila];
	    for(ilb=0;ilb<cw->bin_b->nell_list[ibb];ilb++) {
	      int lb=cw->bin_b->ell_list[ibb][ilb];
	      double xi_1122=cw->xi_1122[cw->ncls_a*la+icl_a][cw->ncls_b*lb+icl_b];
	      double xi_1221=cw->xi_1221[cw->ncls_a*la+icl_a][cw->ncls_b*lb+icl_b];
	      double fac_1122=0.5*(cla1b1[la]*cla2b2[lb]+cla1b1[lb]*cla2b2[la]);
	      double fac_1221=0.5*(cla1b2[la]*cla2b1[lb]+cla1b2[lb]*cla2b1[la]);

	      cbinned+=(xi_1122*fac_1122+xi_1221*fac_1221)*
		cw->bin_a->w_list[iba][ila]*cw->bin_a->f_ell[iba][ila]*
		cw->bin_b->w_list[ibb][ilb]*cw->bin_b->f_ell[ibb][ilb];
	    }
	  }
	  gsl_matrix_set(covar_binned,cw->ncls_a*iba+icl_a,cw->ncls_b*ibb+icl_b,cbinned);
	}
      }
    }
  }

  gsl_matrix *covar_out_g =gsl_matrix_alloc(cw->ncls_a*cw->bin_a->n_bands,cw->ncls_b*cw->bin_b->n_bands);
  gsl_matrix *mat_tmp     =gsl_matrix_alloc(cw->ncls_a*cw->bin_a->n_bands,cw->ncls_b*cw->bin_b->n_bands);
  gsl_matrix *inverse_a   =gsl_matrix_alloc(cw->ncls_a*cw->bin_a->n_bands,cw->ncls_a*cw->bin_a->n_bands);
  gsl_matrix *inverse_b   =gsl_matrix_alloc(cw->ncls_b*cw->bin_b->n_bands,cw->ncls_b*cw->bin_b->n_bands);
  gsl_linalg_LU_invert(wb->coupling_matrix_binned,wb->coupling_matrix_perm,inverse_b); //M_b^-1
  gsl_linalg_LU_invert(wa->coupling_matrix_binned,wa->coupling_matrix_perm,inverse_a); //M_a^-1
  gsl_blas_dgemm(CblasNoTrans,CblasTrans  ,1,covar_binned,inverse_b,0,mat_tmp    ); //tmp = C * M_b^-1^T
  gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1,inverse_a   ,mat_tmp  ,0,covar_out_g); //C' = M_a^-1 * C * M_b^-1^T

  int ii;
  long elem=0;
  for(ii=0;ii<cw->ncls_a*cw->bin_a->n_bands;ii++) {
    int jj;
    for(jj=0;jj<cw->ncls_b*cw->bin_b->n_bands;jj++) {
      covar_out[elem]=gsl_matrix_get(covar_out_g,ii,jj);
      elem++;
    }
  }

  gsl_matrix_free(covar_binned);
  gsl_matrix_free(mat_tmp);
  gsl_matrix_free(inverse_a);
  gsl_matrix_free(inverse_b);
  gsl_matrix_free(covar_out_g);
}

void nmt_covar_workspace_write(nmt_covar_workspace *cw,char *fname)
{
  int ii;
  FILE *fo=my_fopen(fname,"wb");

  my_fwrite(&(cw->lmax_a),sizeof(int),1,fo);
  my_fwrite(&(cw->lmax_b),sizeof(int),1,fo);
  my_fwrite(cw->cs,sizeof(nmt_curvedsky_info),1,fo);
  my_fwrite(&(cw->ncls_a),sizeof(int),1,fo);
  my_fwrite(&(cw->ncls_b),sizeof(int),1,fo);
  for(ii=0;ii<cw->ncls_a*(cw->lmax_a+1);ii++)
    my_fwrite(cw->xi_1122[ii],sizeof(flouble),cw->ncls_b*(cw->lmax_b+1),fo);
  for(ii=0;ii<cw->ncls_a*(cw->lmax_a+1);ii++)
    my_fwrite(cw->xi_1221[ii],sizeof(flouble),cw->ncls_b*(cw->lmax_b+1),fo);

  my_fwrite(&(cw->bin_a->n_bands),sizeof(int),1,fo);
  my_fwrite(cw->bin_a->nell_list,sizeof(int),cw->bin_a->n_bands,fo);
  for(ii=0;ii<cw->bin_a->n_bands;ii++) {
    my_fwrite(cw->bin_a->ell_list[ii],sizeof(int),cw->bin_a->nell_list[ii],fo);
    my_fwrite(cw->bin_a->w_list[ii],sizeof(flouble),cw->bin_a->nell_list[ii],fo);
    my_fwrite(cw->bin_a->f_ell[ii],sizeof(flouble),cw->bin_a->nell_list[ii],fo);
  }

  my_fwrite(&(cw->bin_b->n_bands),sizeof(int),1,fo);
  my_fwrite(cw->bin_b->nell_list,sizeof(int),cw->bin_b->n_bands,fo);
  for(ii=0;ii<cw->bin_b->n_bands;ii++) {
    my_fwrite(cw->bin_b->ell_list[ii],sizeof(int),cw->bin_b->nell_list[ii],fo);
    my_fwrite(cw->bin_b->w_list[ii],sizeof(flouble),cw->bin_b->nell_list[ii],fo);
    my_fwrite(cw->bin_b->f_ell[ii],sizeof(flouble),cw->bin_b->nell_list[ii],fo);
  }

  fclose(fo);
}

nmt_covar_workspace *nmt_covar_workspace_read(char *fname)
{
  int ii;
  nmt_covar_workspace *cw=my_malloc(sizeof(nmt_covar_workspace));
  FILE *fi=my_fopen(fname,"rb");
  cw->cs=my_malloc(sizeof(nmt_curvedsky_info));

  my_fread(&(cw->lmax_a),sizeof(int),1,fi);
  my_fread(&(cw->lmax_b),sizeof(int),1,fi);
  my_fread(cw->cs,sizeof(nmt_curvedsky_info),1,fi);
  my_fread(&(cw->ncls_a),sizeof(int),1,fi);
  my_fread(&(cw->ncls_b),sizeof(int),1,fi);

  cw->xi_1122=my_malloc(cw->ncls_a*(cw->lmax_a+1)*sizeof(flouble *));
  for(ii=0;ii<cw->ncls_a*(cw->lmax_a+1);ii++) {
    cw->xi_1122[ii]=my_malloc(cw->ncls_b*(cw->lmax_b+1)*sizeof(flouble));
    my_fread(cw->xi_1122[ii],sizeof(flouble),cw->ncls_b*(cw->lmax_b+1),fi);
  }
  cw->xi_1221=my_malloc(cw->ncls_a*(cw->lmax_a+1)*sizeof(flouble *));
  for(ii=0;ii<cw->ncls_a*(cw->lmax_a+1);ii++) {
    cw->xi_1221[ii]=my_malloc(cw->ncls_b*(cw->lmax_b+1)*sizeof(flouble));
    my_fread(cw->xi_1221[ii],sizeof(flouble),cw->ncls_b*(cw->lmax_b+1),fi);
  }

  cw->bin_a=my_malloc(sizeof(nmt_binning_scheme));
  my_fread(&(cw->bin_a->n_bands),sizeof(int),1,fi);
  cw->bin_a->nell_list=my_malloc(cw->bin_a->n_bands*sizeof(int));
  cw->bin_a->ell_list=my_malloc(cw->bin_a->n_bands*sizeof(int *));
  cw->bin_a->w_list=my_malloc(cw->bin_a->n_bands*sizeof(flouble *));
  cw->bin_a->f_ell=my_malloc(cw->bin_a->n_bands*sizeof(flouble *));
  my_fread(cw->bin_a->nell_list,sizeof(int),cw->bin_a->n_bands,fi);
  for(ii=0;ii<cw->bin_a->n_bands;ii++) {
    cw->bin_a->ell_list[ii]=my_malloc(cw->bin_a->nell_list[ii]*sizeof(int));
    cw->bin_a->w_list[ii]=my_malloc(cw->bin_a->nell_list[ii]*sizeof(flouble));
    cw->bin_a->f_ell[ii]=my_malloc(cw->bin_a->nell_list[ii]*sizeof(flouble));
    my_fread(cw->bin_a->ell_list[ii],sizeof(int),cw->bin_a->nell_list[ii],fi);
    my_fread(cw->bin_a->w_list[ii],sizeof(flouble),cw->bin_a->nell_list[ii],fi);
    my_fread(cw->bin_a->f_ell[ii],sizeof(flouble),cw->bin_a->nell_list[ii],fi);
  }

  cw->bin_b=my_malloc(sizeof(nmt_binning_scheme));
  my_fread(&(cw->bin_b->n_bands),sizeof(int),1,fi);
  cw->bin_b->nell_list=my_malloc(cw->bin_b->n_bands*sizeof(int));
  cw->bin_b->ell_list=my_malloc(cw->bin_b->n_bands*sizeof(int *));
  cw->bin_b->w_list=my_malloc(cw->bin_b->n_bands*sizeof(flouble *));
  cw->bin_b->f_ell=my_malloc(cw->bin_b->n_bands*sizeof(flouble *));
  my_fread(cw->bin_b->nell_list,sizeof(int),cw->bin_b->n_bands,fi);
  for(ii=0;ii<cw->bin_b->n_bands;ii++) {
    cw->bin_b->ell_list[ii]=my_malloc(cw->bin_b->nell_list[ii]*sizeof(int));
    cw->bin_b->w_list[ii]=my_malloc(cw->bin_b->nell_list[ii]*sizeof(flouble));
    cw->bin_b->f_ell[ii]=my_malloc(cw->bin_b->nell_list[ii]*sizeof(flouble));
    my_fread(cw->bin_b->ell_list[ii],sizeof(int),cw->bin_b->nell_list[ii],fi);
    my_fread(cw->bin_b->w_list[ii],sizeof(flouble),cw->bin_b->nell_list[ii],fi);
    my_fread(cw->bin_b->f_ell[ii],sizeof(flouble),cw->bin_b->nell_list[ii],fi);
  }

  fclose(fi);

  return cw;
}
