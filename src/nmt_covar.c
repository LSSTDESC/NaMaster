#include "config.h"
#include "utils.h"

nmt_covar_workspace *nmt_covar_workspace_init(nmt_field *fla1,nmt_field *fla2,
					      nmt_field *flb1,nmt_field *flb2,
					      int lmax,int niter)
					      
{
  if(!(nmt_diff_curvedsky_info(fla1->cs,fla2->cs)) || !(nmt_diff_curvedsky_info(fla1->cs,flb1->cs)) ||
     !(nmt_diff_curvedsky_info(fla1->cs,flb2->cs)))
    report_error(NMT_ERROR_COVAR,"Can't compute covariance for fields with different resolutions\n");
  
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
    
    wigner_00=my_malloc(2*(cw->lmax+1)*sizeof(double));
    wigner_22=my_malloc(2*(cw->lmax+1)*sizeof(double));

#pragma omp for schedule(dynamic)
    for(ll2=0;ll2<=cw->lmax;ll2++) {
      for(ll3=0;ll3<=cw->lmax;ll3++) {
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

void  nmt_compute_gaussian_covariance(nmt_covar_workspace *cw,
				      int pol_a,int pol_b,int pol_c,int pol_d,
				      nmt_workspace *wa,nmt_workspace *wb,
				      flouble **clac,flouble **clad,
				      flouble **clbc,flouble **clbd,
				      flouble *covar_out)
{
  if((cw->lmax<wa->bin->ell_max) || (cw->lmax<wb->bin->ell_max))
    report_error(NMT_ERROR_COVAR,"Coupling coefficients only computed up to l=%d, but you require"
		 "lmax=%d. Recompute this workspace with a larger lmax\n",cw->lmax,wa->bin->ell_max);

  int nmaps_a=pol_a ? 2 : 1;
  int nmaps_b=pol_b ? 2 : 1;
  int nmaps_c=pol_c ? 2 : 1;
  int nmaps_d=pol_d ? 2 : 1;
  if((wa->ncls!=nmaps_a*nmaps_b) || (wb->ncls!=nmaps_c*nmaps_d))
    report_error(NMT_ERROR_COVAR,"Input spins don't match input workspaces\n");

  gsl_matrix *covar_binned=gsl_matrix_alloc(wa->ncls*wa->bin->n_bands,wb->ncls*wb->bin->n_bands);

#pragma omp parallel default(none)			\
  shared(cw,pol_a,pol_b,pol_c,pol_d)			\
  shared(wa,wb,clac,clad,clbc,clbd)			\
  shared(nmaps_a,nmaps_b,nmaps_c,nmaps_d,covar_binned)
  {
    int band_a;

#pragma omp for
    for(band_a=0;band_a<wa->bin->n_bands;band_a++) {
      int band_b;
      for(band_b=0;band_b<wb->bin->n_bands;band_b++) {
	int ia;
	for(ia=0;ia<nmaps_a;ia++) {
	  int ib;
	  for(ib=0;ib<nmaps_b;ib++) {
	    int ic;
	    int icl_a=ib+nmaps_b*ia;
	    for(ic=0;ic<nmaps_c;ic++) {
	      int id;
	      for(id=0;id<nmaps_d;id++) {
		int ila;
		int icl_b=id+nmaps_d*ic;
		double cbinned=0;
		for(ila=0;ila<wa->bin->nell_list[band_a];ila++) {
		  int ilb;
		  int la=wa->bin->ell_list[band_a][ila];
		  for(ilb=0;ilb<wb->bin->nell_list[band_b];ilb++) {
		    int iap;
		    int lb=wb->bin->ell_list[band_b][ilb];
		    double xis_1122[6]={cw->xi00_1122[la][lb],cw->xi02_1122[la][lb],cw->xi22p_1122[la][lb],
					cw->xi22m_1122[la][lb],-cw->xi22m_1122[la][lb],0};
		    double xis_1221[6]={cw->xi00_1221[la][lb],cw->xi02_1221[la][lb],cw->xi22p_1221[la][lb],
					cw->xi22m_1221[la][lb],-cw->xi22m_1221[la][lb],0};
		    double prefac_ell=wa->bin->w_list[band_a][ila]*wa->bin->f_ell[band_a][ila]*
		      wb->bin->w_list[band_b][ilb]*wb->bin->f_ell[band_b][ilb];
		    for(iap=0;iap<nmaps_a;iap++) {
		      int ibp;
		      for(ibp=0;ibp<nmaps_b;ibp++) {
			int icp;
			for(icp=0;icp<nmaps_c;icp++) {
			  int idp;
			  for(idp=0;idp<nmaps_d;idp++) {
			    double *cl_ac=clac[icp+nmaps_c*iap];
			    double *cl_ad=clad[idp+nmaps_d*iap];
			    double *cl_bc=clbc[icp+nmaps_c*ibp];
			    double *cl_bd=clbd[idp+nmaps_d*ibp];
			    double fac_1122=0.5*(cl_ac[la]*cl_bd[lb]+cl_ac[lb]*cl_bd[la]);
			    double fac_1221=0.5*(cl_ad[la]*cl_bc[lb]+cl_ad[lb]*cl_bc[la]);
			    int ind_1122=cov_get_coupling_pair_index(nmaps_a,nmaps_c,nmaps_b,nmaps_d,
								     ia,iap,ic,icp,ib,ibp,id,idp);
			    int ind_1221=cov_get_coupling_pair_index(nmaps_a,nmaps_d,nmaps_b,nmaps_c,
								     ia,iap,id,idp,ib,ibp,ic,icp);
			    
			    cbinned+=(xis_1122[ind_1122]*fac_1122+xis_1221[ind_1221]*fac_1221)*prefac_ell;
			  }
			}
		      }
		    }
		  }
		}
		gsl_matrix_set(covar_binned,wa->ncls*band_a+icl_a,wb->ncls*band_b+icl_b,cbinned);
	      }
	    }
	  }
	}
      }
    } //end omp for
  } //end omp parallel

  //Sandwich with inverse MCM
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
