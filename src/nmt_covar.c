#include "utils.h"

nmt_covar_workspace *nmt_covar_workspace_init(nmt_field *fla1,nmt_field *fla2,
					      nmt_field *flb1,nmt_field *flb2,
					      int lmax,int niter)
					      
{
  if(!(nmt_diff_curvedsky_info(fla1->cs,fla2->cs)) || !(nmt_diff_curvedsky_info(fla1->cs,flb1->cs)) ||
     !(nmt_diff_curvedsky_info(fla1->cs,flb2->cs)))
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

  cw->lmax=lmax;
  flouble *cl_mask_1122=my_malloc((cw->lmax+1)*sizeof(flouble));
  flouble *cl_mask_1221=my_malloc((cw->lmax+1)*sizeof(flouble));
  cw->xi00_1122=my_malloc((cw->lmax+1)*sizeof(flouble *));
  cw->xi00_1221=my_malloc((cw->lmax+1)*sizeof(flouble *));
  cw->xi02_1122=my_malloc((cw->lmax+1)*sizeof(flouble *));
  cw->xi02_1221=my_malloc((cw->lmax+1)*sizeof(flouble *));
  cw->xi22p_1122=my_malloc((cw->lmax+1)*sizeof(flouble *));
  cw->xi22p_1221=my_malloc((cw->lmax+1)*sizeof(flouble *));
  cw->xi22m_1122=my_malloc((cw->lmax+1)*sizeof(flouble *));
  cw->xi22m_1221=my_malloc((cw->lmax+1)*sizeof(flouble *));
  for(ii=0;ii<(cw->lmax+1);ii++) {
    cw->xi00_1122[ii]=my_malloc((cw->lmax+1)*sizeof(flouble));
    cw->xi00_1221[ii]=my_malloc((cw->lmax+1)*sizeof(flouble));
    cw->xi02_1122[ii]=my_malloc((cw->lmax+1)*sizeof(flouble));
    cw->xi02_1221[ii]=my_malloc((cw->lmax+1)*sizeof(flouble));
    cw->xi22p_1122[ii]=my_malloc((cw->lmax+1)*sizeof(flouble));
    cw->xi22p_1221[ii]=my_malloc((cw->lmax+1)*sizeof(flouble));
    cw->xi22m_1122[ii]=my_malloc((cw->lmax+1)*sizeof(flouble));
    cw->xi22m_1221[ii]=my_malloc((cw->lmax+1)*sizeof(flouble));
  }
  
  he_map_product(fla1->cs,fla1->mask,flb1->mask,mask_a1b1);
  he_map_product(fla1->cs,fla1->mask,flb2->mask,mask_a1b2);
  he_map_product(fla1->cs,fla2->mask,flb1->mask,mask_a2b1);
  he_map_product(fla1->cs,fla2->mask,flb2->mask,mask_a2b2);
  he_anafast(&mask_a1b1,&mask_a2b2,0,0,&cl_mask_1122,fla1->cs,cw->lmax,niter);
  he_anafast(&mask_a1b2,&mask_a2b1,0,0,&cl_mask_1221,fla1->cs,cw->lmax,niter);
  free(mask_a1b1); free(mask_a1b2); free(mask_a2b1); free(mask_a2b2);
  for(ii=0;ii<=cw->lmax;ii++) {
    cl_mask_1122[ii]*=(ii+0.5)/(2*M_PI);
    cl_mask_1221[ii]*=(ii+0.5)/(2*M_PI);
  }

#pragma omp parallel default(none)		\
  shared(cw,cl_mask_1122,cl_mask_1221)
  {
    int ll2,ll3;
    double *wigner_00=NULL,*wigner_22=NULL;
    int lstart=0;
    
    wigner_00=my_malloc(2*(cw->lmax+1)*sizeof(double));
    wigner_22=my_malloc(2*(cw->lmax+1)*sizeof(double));
    
    //if(cw->ncls_a>1)
    //      lstart=2;

#pragma omp for schedule(dynamic)
    for(ll2=lstart;ll2<=cw->lmax;ll2++) {
      for(ll3=lstart;ll3<=cw->lmax;ll3++) {
	int jj,l1,lmin_here,lmax_here;
	int lmin_here_00=0,lmax_here_00=2*(cw->lmax+1)+1;
	int lmin_here_22=0,lmax_here_22=2*(cw->lmax+1)+1;
	flouble xi00_1122=0,xi00_1221=0;
	flouble xi02_1122=0,xi02_1221=0;
	flouble xi22p_1122=0,xi22p_1221=0;
	flouble xi22m_1122=0,xi22m_1221=0;

	drc3jj(ll2,ll3,0,0,&lmin_here_00,&lmax_here_00,wigner_00,2*(cw->lmax+1));
	drc3jj(ll2,ll3,2,-2,&lmin_here_22,&lmax_here_22,wigner_22,2*(cw->lmax+1));

	lmin_here=NMT_MIN(lmin_here_00,lmin_here_22);
	lmax_here=NMT_MAX(lmax_here_00,lmax_here_22);
	for(l1=lmin_here;l1<=lmax_here;l1++) {
	  if(l1<=cw->lmax) {
	    flouble wfacs[4];
	    int suml=l1+ll2+ll3;
	    int j00=l1-lmin_here_00;
	    int j22=l1-lmin_here_22;
	    int prefac_m=suml & 1;
	    int prefac_p=!prefac_m;
	    wfacs[0]=wigner_00[j00]*wigner_00[j00];
	    wfacs[1]=wigner_00[j00]*wigner_22[j22];
	    wfacs[2]=wigner_22[j22]*wigner_22[j22];
	    wfacs[3]=prefac_m*wfacs[2];
	    wfacs[2]*=prefac_p;

	    xi00_1122+=cl_mask_1122[l1]*wfacs[0];
	    xi00_1221+=cl_mask_1221[l1]*wfacs[0];
	    xi02_1122+=cl_mask_1122[l1]*wfacs[1];
	    xi02_1221+=cl_mask_1221[l1]*wfacs[1];
	    xi22p_1122+=cl_mask_1122[l1]*wfacs[2];
	    xi22p_1221+=cl_mask_1221[l1]*wfacs[2];
	    xi22m_1122+=cl_mask_1122[l1]*wfacs[3];
	    xi22m_1221+=cl_mask_1221[l1]*wfacs[3];
	  }
	}

	cw->xi00_1122[ll2][ll3]=xi00_1122;
	cw->xi00_1221[ll2][ll3]=xi00_1221;
	cw->xi02_1122[ll2][ll3]=xi02_1122;
	cw->xi02_1221[ll2][ll3]=xi02_1221;
	cw->xi22p_1122[ll2][ll3]=xi22p_1122;
	cw->xi22p_1221[ll2][ll3]=xi22p_1221;
	cw->xi22m_1122[ll2][ll3]=xi22m_1122;
	cw->xi22m_1221[ll2][ll3]=xi22m_1221;
      }
    } //end omp for
    free(wigner_00);
    free(wigner_22);
  } //end omp parallel

  free(cl_mask_1122);
  free(cl_mask_1221);
  return cw;
}

void nmt_covar_workspace_free(nmt_covar_workspace *cw)
{
  int ii;
  for(ii=0;ii<=cw->lmax;ii++) {
    free(cw->xi00_1122[ii]);
    free(cw->xi00_1221[ii]);
    free(cw->xi02_1122[ii]);
    free(cw->xi02_1221[ii]);
    free(cw->xi22p_1122[ii]);
    free(cw->xi22p_1221[ii]);
    free(cw->xi22m_1122[ii]);
    free(cw->xi22m_1221[ii]);
  }
  free(cw);
}

void  nmt_compute_gaussian_covariance(nmt_covar_workspace *cw,nmt_workspace *wa,nmt_workspace *wb,
				      flouble *cla1b1,flouble *cla1b2,flouble *cla2b1,flouble *cla2b2,
				      flouble *covar_out)
{
  if((cw->lmax<wa->bin->ell_max) || (cw->lmax<wb->bin->ell_max))
    report_error(NMT_ERROR_COVAR,"Coupling coefficients only computed up to l=%d, but you require"
		 "lmax=%d. Recompute this workspace with a larger lmax\n",cw->lmax,wa->bin->ell_max);
  
  int icl_a;
  gsl_matrix *covar_binned=gsl_matrix_alloc(wa->ncls*wa->bin->n_bands,wb->ncls*wb->bin->n_bands);
  for(icl_a=0;icl_a<wa->ncls;icl_a++) {
    int icl_b;
    for(icl_b=0;icl_b<wb->ncls;icl_b++) {
      int iba;
      for(iba=0;iba<wa->bin->n_bands;iba++) {
	int ibb;
	for(ibb=0;ibb<wb->bin->n_bands;ibb++) {
	  double cbinned=0;
	  int ila;
	  for(ila=0;ila<wa->bin->nell_list[iba];ila++) {
	    int ilb;
	    int la=wa->bin->ell_list[iba][ila];
	    for(ilb=0;ilb<wb->bin->nell_list[ibb];ilb++) {
	      int lb=wb->bin->ell_list[ibb][ilb];
	      double xi00_1122=cw->xi00_1122[wa->ncls*la+icl_a][wb->ncls*lb+icl_b];
	      double xi00_1221=cw->xi00_1221[wa->ncls*la+icl_a][wb->ncls*lb+icl_b];
	      double fac_1122=0.5*(cla1b1[la]*cla2b2[lb]+cla1b1[lb]*cla2b2[la]);
	      double fac_1221=0.5*(cla1b2[la]*cla2b1[lb]+cla1b2[lb]*cla2b1[la]);

	      cbinned+=(xi00_1122*fac_1122+xi00_1221*fac_1221)*
		wa->bin->w_list[iba][ila]*wa->bin->f_ell[iba][ila]*
		wb->bin->w_list[ibb][ilb]*wb->bin->f_ell[ibb][ilb];
	    }
	  }
	  gsl_matrix_set(covar_binned,wa->ncls*iba+icl_a,wa->ncls*ibb+icl_b,cbinned);
	}
      }
    }
  }

  gsl_matrix *covar_out_g =gsl_matrix_alloc(wa->ncls*wa->bin->n_bands,wb->ncls*wb->bin->n_bands);
  gsl_matrix *mat_tmp     =gsl_matrix_alloc(wa->ncls*wa->bin->n_bands,wb->ncls*wb->bin->n_bands);
  gsl_matrix *inverse_a   =gsl_matrix_alloc(wa->ncls*wa->bin->n_bands,wa->ncls*wa->bin->n_bands);
  gsl_matrix *inverse_b   =gsl_matrix_alloc(wb->ncls*wb->bin->n_bands,wb->ncls*wb->bin->n_bands);
  gsl_linalg_LU_invert(wb->coupling_matrix_binned,wb->coupling_matrix_perm,inverse_b); //M_b^-1
  gsl_linalg_LU_invert(wa->coupling_matrix_binned,wa->coupling_matrix_perm,inverse_a); //M_a^-1
  gsl_blas_dgemm(CblasNoTrans,CblasTrans  ,1,covar_binned,inverse_b,0,mat_tmp    ); //tmp = C * M_b^-1^T
  gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1,inverse_a   ,mat_tmp  ,0,covar_out_g); //C' = M_a^-1 * C * M_b^-1^T

  int ii;
  long elem=0;
  for(ii=0;ii<wa->ncls*wa->bin->n_bands;ii++) {
    int jj;
    for(jj=0;jj<wb->ncls*wb->bin->n_bands;jj++) {
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

  my_fwrite(&(cw->lmax),sizeof(int),1,fo);
  //00
  for(ii=0;ii<=cw->lmax;ii++)
    my_fwrite(cw->xi00_1122[ii],sizeof(flouble),cw->lmax+1,fo);
  for(ii=0;ii<=cw->lmax;ii++)
    my_fwrite(cw->xi00_1221[ii],sizeof(flouble),cw->lmax+1,fo);
  //02
  for(ii=0;ii<=cw->lmax;ii++)
    my_fwrite(cw->xi02_1122[ii],sizeof(flouble),cw->lmax+1,fo);
  for(ii=0;ii<=cw->lmax;ii++)
    my_fwrite(cw->xi02_1221[ii],sizeof(flouble),cw->lmax+1,fo);
  //22+
  for(ii=0;ii<=cw->lmax;ii++)
    my_fwrite(cw->xi22p_1122[ii],sizeof(flouble),cw->lmax+1,fo);
  for(ii=0;ii<=cw->lmax;ii++)
    my_fwrite(cw->xi22p_1221[ii],sizeof(flouble),cw->lmax+1,fo);
  //22-
  for(ii=0;ii<=cw->lmax;ii++)
    my_fwrite(cw->xi22m_1122[ii],sizeof(flouble),cw->lmax+1,fo);
  for(ii=0;ii<=cw->lmax;ii++)
    my_fwrite(cw->xi22m_1221[ii],sizeof(flouble),cw->lmax+1,fo);

  fclose(fo);
}

nmt_covar_workspace *nmt_covar_workspace_read(char *fname)
{
  int ii;
  nmt_covar_workspace *cw=my_malloc(sizeof(nmt_covar_workspace));
  FILE *fi=my_fopen(fname,"rb");

  my_fread(&(cw->lmax),sizeof(int),1,fi);

  //00
  cw->xi00_1122=my_malloc((cw->lmax+1)*sizeof(flouble *));
  for(ii=0;ii<=cw->lmax;ii++) {
    cw->xi00_1122[ii]=my_malloc((cw->lmax+1)*sizeof(flouble));
    my_fread(cw->xi00_1122[ii],sizeof(flouble),cw->lmax+1,fi);
  }
  cw->xi00_1221=my_malloc((cw->lmax+1)*sizeof(flouble *));
  for(ii=0;ii<=cw->lmax;ii++) {
    cw->xi00_1221[ii]=my_malloc((cw->lmax+1)*sizeof(flouble));
    my_fread(cw->xi00_1221[ii],sizeof(flouble),cw->lmax+1,fi);
  }
  //02
  cw->xi02_1122=my_malloc((cw->lmax+1)*sizeof(flouble *));
  for(ii=0;ii<=cw->lmax;ii++) {
    cw->xi02_1122[ii]=my_malloc((cw->lmax+1)*sizeof(flouble));
    my_fread(cw->xi02_1122[ii],sizeof(flouble),cw->lmax+1,fi);
  }
  cw->xi02_1221=my_malloc((cw->lmax+1)*sizeof(flouble *));
  for(ii=0;ii<=cw->lmax;ii++) {
    cw->xi02_1221[ii]=my_malloc((cw->lmax+1)*sizeof(flouble));
    my_fread(cw->xi02_1221[ii],sizeof(flouble),cw->lmax+1,fi);
  }
  //22+
  cw->xi22p_1122=my_malloc((cw->lmax+1)*sizeof(flouble *));
  for(ii=0;ii<=cw->lmax;ii++) {
    cw->xi22p_1122[ii]=my_malloc((cw->lmax+1)*sizeof(flouble));
    my_fread(cw->xi22p_1122[ii],sizeof(flouble),cw->lmax+1,fi);
  }
  cw->xi22p_1221=my_malloc((cw->lmax+1)*sizeof(flouble *));
  for(ii=0;ii<=cw->lmax;ii++) {
    cw->xi22p_1221[ii]=my_malloc((cw->lmax+1)*sizeof(flouble));
    my_fread(cw->xi22p_1221[ii],sizeof(flouble),cw->lmax+1,fi);
  }
  //22-
  cw->xi22m_1122=my_malloc((cw->lmax+1)*sizeof(flouble *));
  for(ii=0;ii<=cw->lmax;ii++) {
    cw->xi22m_1122[ii]=my_malloc((cw->lmax+1)*sizeof(flouble));
    my_fread(cw->xi22m_1122[ii],sizeof(flouble),cw->lmax+1,fi);
  }
  cw->xi22m_1221=my_malloc((cw->lmax+1)*sizeof(flouble *));
  for(ii=0;ii<=cw->lmax;ii++) {
    cw->xi22m_1221[ii]=my_malloc((cw->lmax+1)*sizeof(flouble));
    my_fread(cw->xi22m_1221[ii],sizeof(flouble),cw->lmax+1,fi);
  }

  fclose(fi);

  return cw;
}
