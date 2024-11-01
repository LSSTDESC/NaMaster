#include "config.h"
#include "utils.h"

nmt_covar_workspace *nmt_covar_workspace_init(flouble *cl_masks_11_22,
					      flouble *cl_masks_12_21,
					      int lmax, int lmax_mask,
					      int l_toeplitz,
					      int l_exact,int dl_band,
					      int spin0_only)
{
  int ii;
  nmt_covar_workspace *cw=my_malloc(sizeof(nmt_covar_workspace));
  cw->lmax=lmax;
  cw->lmax_mask=lmax_mask;
  cw->spin0_only=spin0_only;

  flouble **cl_masks=my_malloc(2*sizeof(flouble));
  cl_masks[0]=my_malloc((cw->lmax_mask+1)*sizeof(flouble));
  cl_masks[1]=my_malloc((cw->lmax_mask+1)*sizeof(flouble));
  for(ii=0;ii<=cw->lmax_mask;ii++) {
    cl_masks[0][ii]=cl_masks_11_22[ii]*(ii+0.5)/(2*M_PI);
    cl_masks[1][ii]=cl_masks_12_21[ii]*(ii+0.5)/(2*M_PI);
  }

  nmt_master_calculator *c;
  if(cw->spin0_only) {
    c=nmt_compute_master_coefficients(cw->lmax, cw->lmax_mask,
				      2, cl_masks,
				      0, 0, 0, 0, 0, 0, 0,
				      l_toeplitz,l_exact,dl_band);
  }
  else {
    c=nmt_compute_master_coefficients(cw->lmax, cw->lmax_mask,
				      2, cl_masks,
				      0, 2, 0, 0, 0, 0, 1,
				      l_toeplitz,l_exact,dl_band);
  }
  cw->xi00_1122=c->xi_00[0];
  cw->xi00_1221=c->xi_00[1];
  if(cw->spin0_only) {
    cw->xi02_1122=c->xi_00[0];
    cw->xi02_1221=c->xi_00[1];
    cw->xi22p_1122=c->xi_00[0];
    cw->xi22p_1221=c->xi_00[1];
    cw->xi22m_1122=c->xi_00[0];
    cw->xi22m_1221=c->xi_00[1];
  }
  else {
    cw->xi02_1122=c->xi_0s[0][0];
    cw->xi02_1221=c->xi_0s[1][0];
    cw->xi22p_1122=c->xi_pp[0][0];
    cw->xi22p_1221=c->xi_pp[1][0];
    cw->xi22m_1122=c->xi_mm[0][0];
    cw->xi22m_1221=c->xi_mm[1][0];
    free(c->xi_0s[0]);
    free(c->xi_0s[1]);
    free(c->xi_0s);
    free(c->xi_pp[0]);
    free(c->xi_pp[1]);
    free(c->xi_pp);
    free(c->xi_mm[0]);
    free(c->xi_mm[1]);
    free(c->xi_mm);
  }
  free(c->xi_00);
  free(c);
  free(cl_masks[0]);
  free(cl_masks[1]);
  free(cl_masks);
  return cw;
}

void nmt_covar_workspace_free(nmt_covar_workspace *cw)
{
  int ii;
  for(ii=0;ii<=cw->lmax;ii++) {
    free(cw->xi00_1122[ii]);
    free(cw->xi00_1221[ii]);
    if(cw->spin0_only==0) {
      free(cw->xi02_1122[ii]);
      free(cw->xi02_1221[ii]);
      free(cw->xi22p_1122[ii]);
      free(cw->xi22p_1221[ii]);
      free(cw->xi22m_1122[ii]);
      free(cw->xi22m_1221[ii]);
    }
  }
  free(cw);
}

void  nmt_compute_gaussian_covariance_coupled(nmt_covar_workspace *cw,
                                              int spin_a,int spin_b,int spin_c,int spin_d,
                                              nmt_workspace *wa,nmt_workspace *wb,
                                              flouble **clac,flouble **clad,
                                              flouble **clbc,flouble **clbd,
                                              flouble *covar_out)
{
  if((cw->lmax<wa->bin->ell_max) || (cw->lmax<wb->bin->ell_max))
    report_error(NMT_ERROR_COVAR,"Coupling coefficients only computed up to l=%d, but you require "
		 "lmax=%d. Recompute this workspace with a larger lmax\n",cw->lmax,wa->bin->ell_max);
  if((cw->spin0_only) && (spin_a+spin_b+spin_c+spin_d != 0))
    report_error(NMT_ERROR_COVAR,"Coupling coefficients only computed for spin=0\n");

  int nmaps_a=spin_a ? 2 : 1;
  int nmaps_b=spin_b ? 2 : 1;
  int nmaps_c=spin_c ? 2 : 1;
  int nmaps_d=spin_d ? 2 : 1;
  if((wa->ncls!=nmaps_a*nmaps_b) || (wb->ncls!=nmaps_c*nmaps_d))
    report_error(NMT_ERROR_COVAR,"Input spins don't match input workspaces\n");

#pragma omp parallel default(none)			\
  shared(cw,spin_a,spin_b,spin_c,spin_d)                \
  shared(wa,wb,clac,clad,clbc,clbd)			\
  shared(nmaps_a,nmaps_b,nmaps_c,nmaps_d,covar_out)
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
		for(ila=0;ila<wa->bin->nell_list[band_a];ila++) {
		  int ilb;
		  int la=wa->bin->ell_list[band_a][ila];
		  for(ilb=0;ilb<wb->bin->nell_list[band_b];ilb++) {
		    int iap;
		    int lb=wb->bin->ell_list[band_b][ilb];
		    double xis_1122[6]={cw->xi00_1122[la][lb],
                                        cw->xi02_1122[la][lb],
                                        cw->xi22p_1122[la][lb],
					cw->xi22m_1122[la][lb],
                                        -cw->xi22m_1122[la][lb],0};
		    double xis_1221[6]={cw->xi00_1221[la][lb],
                                        cw->xi02_1221[la][lb],
                                        cw->xi22p_1221[la][lb],
					cw->xi22m_1221[la][lb],
                                        -cw->xi22m_1221[la][lb],0};
		    double prefac_ell=wa->bin->f_ell[band_a][ila]*wb->bin->f_ell[band_b][ilb];
                    double cov_element=0;
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
			    
                            cov_element+=(xis_1122[ind_1122]*fac_1122+xis_1221[ind_1221]*fac_1221)*prefac_ell;
			  }
			}
		      }
		    }
                    covar_out[((wa->ncls*la+icl_a)*(cw->lmax+1)+lb)*wb->ncls+icl_b]=cov_element;
		  }
		}
	      }
	    }
	  }
	}
      }
    } //end omp for
  } //end omp parallel
}

void  nmt_compute_gaussian_covariance(nmt_covar_workspace *cw,
				      int spin_a,int spin_b,int spin_c,int spin_d,
				      nmt_workspace *wa,nmt_workspace *wb,
				      flouble **clac,flouble **clad,
				      flouble **clbc,flouble **clbd,
				      flouble *covar_out)
{
  if((cw->lmax<wa->bin->ell_max) || (cw->lmax<wb->bin->ell_max))
    report_error(NMT_ERROR_COVAR,"Coupling coefficients only computed up to l=%d, but you require "
		 "lmax=%d. Recompute this workspace with a larger lmax\n",cw->lmax,wa->bin->ell_max);
  if((cw->spin0_only) && (spin_a+spin_b+spin_c+spin_d != 0))
    report_error(NMT_ERROR_COVAR,"Coupling coefficients only computed for spin=0\n");

  int nmaps_a=spin_a ? 2 : 1;
  int nmaps_b=spin_b ? 2 : 1;
  int nmaps_c=spin_c ? 2 : 1;
  int nmaps_d=spin_d ? 2 : 1;
  if((wa->ncls!=nmaps_a*nmaps_b) || (wb->ncls!=nmaps_c*nmaps_d))
    report_error(NMT_ERROR_COVAR,"Input spins don't match input workspaces\n");

  gsl_matrix *covar_binned=gsl_matrix_alloc(wa->ncls*wa->bin->n_bands,wb->ncls*wb->bin->n_bands);

#pragma omp parallel default(none)			\
  shared(cw,spin_a,spin_b,spin_c,spin_d)                \
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
