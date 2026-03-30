#include "config.h"
#include "utils.h"

nmt_covar_workspace *nmt_covar_workspace_init(int spin_a1, int spin_a2,
					      int spin_b1, int spin_b2,
					      int all_spins, int auto_any,
					      int has_1122, int has_1221,
					      flouble *cl_masks_11_22,
					      flouble *cl_masks_12_21,
					      int lmax, int lmax_mask,
					      int l_toeplitz,
					      int l_exact,int dl_band)
{
  int ii;
  nmt_covar_workspace *cw=my_malloc(sizeof(nmt_covar_workspace));
  cw->lmax=lmax;
  cw->lmax_mask=lmax_mask;
  cw->all_spins=all_spins;
  cw->spin_a1=spin_a1;
  cw->spin_a2=spin_a2;
  cw->spin_b1=spin_b1;
  cw->spin_b2=spin_b2;
  cw->xi00_1122=NULL;
  cw->xi00_1221=NULL;
  cw->xi02_1122=NULL;
  cw->xi02_1221=NULL;
  cw->xi22p_1122=NULL;
  cw->xi22p_1221=NULL;
  cw->xi22m_1122=NULL;
  cw->xi22m_1221=NULL;

  flouble **cl_masks=my_malloc(2*sizeof(flouble));
  cl_masks[0]=my_malloc((cw->lmax_mask+1)*sizeof(flouble));
  cl_masks[1]=my_malloc((cw->lmax_mask+1)*sizeof(flouble));
  for(ii=0;ii<=cw->lmax_mask;ii++) {
    int iadd=0;
    if(has_1122) {
      cl_masks[iadd][ii]=cl_masks_11_22[ii]*(ii+0.5)/(2*M_PI);
      iadd++;
    }
    if(has_1221)
      cl_masks[iadd][ii]=cl_masks_12_21[ii]*(ii+0.5)/(2*M_PI);
  }
  int n_cl_masks=has_1122+has_1221;

  int is_00_a = (spin_a1==0) && (spin_a2==0);
  int is_00_b = (spin_b1==0) && (spin_b2==0);
  int is_0s_a = ((spin_a1==0) || (spin_a2==0)) && (spin_a1!=spin_a2);
  int is_0s_b = ((spin_b1==0) || (spin_b2==0)) && (spin_b1!=spin_b2);
  int is_ss_a = (spin_a1!=0) && (spin_a2!=0);
  int is_ss_b = (spin_b1!=0) && (spin_b2!=0);
  int is_00_00 = is_00_a && is_00_b;
  int is_00_0s = (is_00_a && is_0s_b) || (is_0s_a && is_00_b);
  int is_00_ss = (is_00_a && is_ss_b) || (is_ss_a && is_00_b);
  int is_0s_0s = is_0s_a && is_0s_b;
  int is_0s_ss = (is_0s_a && is_ss_b) || (is_ss_a && is_0s_b);
  int is_ss_ss = is_ss_a && is_ss_b;

  nmt_master_calculator *c;
  int ncls=auto_any ? 1 : 2;
  if(n_cl_masks==1)
    ncls=1;

  if(all_spins) {
    int iadd=0;
    c=nmt_compute_master_coefficients(cw->lmax, cw->lmax_mask,
				      n_cl_masks, cl_masks,
				      0, 2, 0, 0, 0, 0, 1,
				      l_toeplitz,l_exact,dl_band);
    if(has_1122) {
      cw->xi00_1122=c->xi_00[iadd];
      cw->xi02_1122=c->xi_0s[iadd][0];
      cw->xi22p_1122=c->xi_pp[iadd][0];
      cw->xi22m_1122=c->xi_mm[iadd][0];
      free(c->xi_0s[iadd]);
      free(c->xi_pp[iadd]);
      free(c->xi_mm[iadd]);
      iadd++;
    }
    if(has_1221) {
      cw->xi00_1221=c->xi_00[iadd];
      cw->xi02_1221=c->xi_0s[iadd][0];
      cw->xi22p_1221=c->xi_pp[iadd][0];
      cw->xi22m_1221=c->xi_mm[iadd][0];
      free(c->xi_0s[iadd]);
      free(c->xi_pp[iadd]);
      free(c->xi_mm[iadd]);
    }
    free(c->xi_0s);
    free(c->xi_pp);
    free(c->xi_mm);
    free(c->xi_00);
    free(c->lfac);
    free(c);
  }
  else if(is_0s_0s) {  // For the  (0,s)-(0,s) covariance we need both Xi00 and Xi0s
    int i00ss, i0s0s;

    if(spin_a1 == spin_b1) {  // Combination is 0s, 0s, so 1122 = 00,22 and 1221 is 02,02
      if(has_1122) {
	if(has_1221) {  // has 1122 and 1221
	  i00ss=0; i0s0s=1;
	} else { // only has 1122
	  i00ss=0; i0s0s=-1;
	}
      } else {
	if(has_1221) { // only has 1221
	  i00ss=-1;
	  i0s0s=0;
	} else {  // has neither 1122 nor 1221 (this shouldn't happen)
	  i00ss=-1;
	  i0s0s=-1;
	}
      }
    } else { // Combination is 0s, s0 (or vice-versa), so 1122 = 02,02 and 1221 is 00,22
      if(has_1122) {
	if(has_1221) {  // has 1122 and 1221
	  i0s0s=0; i00ss=1;
	} else { // only has 1122
	  i0s0s=0; i00ss=-1;
	}
      } else {
	if(has_1221) { // only has 1221
	  i0s0s=-1; i00ss=0;
	} else { // has neither 1122 nor 1221 (this shouldn't happen)
	  i0s0s=-1; i00ss=-1;
	}
      }
    }

    if(i00ss != -1) { // Compute Xi_02
      c=nmt_compute_master_coefficients(cw->lmax, cw->lmax_mask,
					1, &(cl_masks[i00ss]),
					0, 2,
					0, 0, 0, 0,
					0,
					l_toeplitz,l_exact,dl_band);
      if(spin_a1 == spin_b1)  // we're in the 0s, 0s case
	cw->xi02_1122=c->xi_0s[0][0];
      else  // we're in the 0s, s0 case
	cw->xi02_1221=c->xi_0s[0][0];
      free(c->xi_0s[0]);
      free(c->xi_0s);
      free(c->lfac);
      free(c);
    }
    if(i0s0s != -1) { // Compute Xi_00
      c=nmt_compute_master_coefficients(cw->lmax, cw->lmax_mask,
					1, &(cl_masks[i0s0s]),
					0, 0,
					0, 0, 0, 0,
					0,
					l_toeplitz,l_exact,dl_band);
      if(spin_a1 != spin_b1)  // we're in the 0s, s0 case
	cw->xi00_1122=c->xi_00[0];
      else // we're in the 0s, 0s case
	cw->xi00_1221=c->xi_00[0];
      free(c->xi_00);
      free(c->lfac);
      free(c);
    }
  }
  else if(is_ss_ss) {  // Case (s,s)-(s,s)
    c=nmt_compute_master_coefficients(cw->lmax, cw->lmax_mask,
				      ncls, cl_masks,
				      2, 2,
				      0, 0, 0, 0,
				      0,
				      l_toeplitz,l_exact,dl_band);
    int iadd=0;
    if(has_1122) {
      cw->xi22p_1122=c->xi_pp[iadd][0];
      cw->xi22m_1122=c->xi_mm[iadd][0];
      free(c->xi_pp[iadd]);
      free(c->xi_mm[iadd]);
      iadd++;
    }
    if(has_1221) {
      if(auto_any && has_1122) {
	cw->xi22p_1221=cw->xi22p_1122;
	cw->xi22m_1221=cw->xi22m_1122;
      }
      else {
	cw->xi22p_1221=c->xi_pp[iadd][0];
	cw->xi22m_1221=c->xi_mm[iadd][0];
	free(c->xi_pp[iadd]);
	free(c->xi_mm[iadd]);
      }
    }
    free(c->xi_pp);
    free(c->xi_mm);
    free(c->lfac);
    free(c);
  }
  else if(is_0s_ss) { // Case (0,s)-(s,s)
    c=nmt_compute_master_coefficients(cw->lmax, cw->lmax_mask,
				      ncls, cl_masks,
				      0, 2,
				      0, 0, 0, 0,
				      0,
				      l_toeplitz,l_exact,dl_band);
    int iadd=0;
    if(has_1122) {
      cw->xi02_1122=c->xi_0s[0][0];
      free(c->xi_0s[iadd]);
      iadd++;
    }
    if(has_1221) {
      if(auto_any)
	cw->xi02_1221=cw->xi02_1122;
      else {
	cw->xi02_1221=c->xi_0s[iadd][0];
	free(c->xi_0s[iadd]);
      }
    }
    free(c->xi_0s);
    free(c->lfac);
    free(c);
  }
  else { //Cases (0,0)-(0,0), (0,0)-(0,s), (0,0)-(s,s)
    c=nmt_compute_master_coefficients(cw->lmax, cw->lmax_mask,
				      ncls, cl_masks,
				      0, 0,
				      0, 0, 0, 0,
				      0,
				      l_toeplitz,l_exact,dl_band);
    int iadd=0;
    if(has_1122) {
      cw->xi00_1122=c->xi_00[iadd];
      iadd++;
    }
    if(has_1221) {
      if(auto_any)
	cw->xi00_1221=cw->xi00_1122;
      else
	cw->xi00_1221=c->xi_00[iadd];
    }
    free(c->xi_00);
    free(c->lfac);
    free(c);
  }

  free(cl_masks[0]);
  free(cl_masks[1]);
  free(cl_masks);
  return cw;
}

void _free_2d_xi(int lmax, flouble **xi) {
  if(xi==NULL)
    return;

  int ii;
  for(ii=0;ii<=lmax;ii++)
    free(xi[ii]);
  free(xi);
}

void nmt_covar_workspace_free(nmt_covar_workspace *cw)
{
  _free_2d_xi(cw->lmax, cw->xi00_1122);
  cw->xi00_1122=NULL;
  _free_2d_xi(cw->lmax, cw->xi00_1221);
  cw->xi00_1221=NULL;
  _free_2d_xi(cw->lmax, cw->xi02_1122);
  cw->xi02_1122=NULL;
  _free_2d_xi(cw->lmax, cw->xi02_1221);
  cw->xi02_1221=NULL;
  _free_2d_xi(cw->lmax, cw->xi22p_1122);
  cw->xi22p_1122=NULL;
  _free_2d_xi(cw->lmax, cw->xi22p_1221);
  cw->xi22p_1221=NULL;
  _free_2d_xi(cw->lmax, cw->xi22m_1122);
  cw->xi22m_1122=NULL;
  _free_2d_xi(cw->lmax, cw->xi22m_1221);
  cw->xi22m_1221=NULL;
  free(cw);
}

double _pick_xi(nmt_covar_workspace *cw,
		int index, int is_1122, int la, int lb)
{
  if(index == 5)
    return 0.;

  flouble **xi;
  int sign=1;

  if(index == 0)
    xi = is_1122 ? cw->xi00_1122 : cw->xi00_1221;
  else if(index == 1)
    xi = is_1122 ? cw->xi02_1122 : cw->xi02_1221;
  else if(index == 2)
    xi = is_1122 ? cw->xi22p_1122 : cw->xi22p_1221;
  else if(index == 3)
    xi = is_1122 ? cw->xi22m_1122 : cw->xi22m_1221;
  else if(index == 4) {
    xi = is_1122 ? cw->xi22m_1122 : cw->xi22m_1221;
    sign = -1;
  }
  else {
    report_error(NMT_ERROR_COVAR, "Unknown coupling index %d \n",
		 index);
  }

  if(xi==NULL) {
    report_error(NMT_ERROR_COVAR, "You requested coupling coefficients with index "
		 "%d, but this has not been calculated.\n", index);
  }
  return sign*xi[la][lb];
}

void  nmt_compute_gaussian_covariance_coupled(nmt_covar_workspace *cw,
					      int spin_a,int spin_b,int spin_c,int spin_d,
                                              nmt_workspace *wa,nmt_workspace *wb,
					      int has_1122, int has_1221,
                                              flouble **clac,flouble **clad,
                                              flouble **clbc,flouble **clbd,
                                              flouble *covar_out)
{
  if((cw->lmax<wa->bin->ell_max) || (cw->lmax<wb->bin->ell_max))
    report_error(NMT_ERROR_COVAR,"Coupling coefficients only computed up to l=%d, but you require "
		 "lmax=%d. Recompute this workspace with a larger lmax\n",cw->lmax,wa->bin->ell_max);

  int sa, sb, sc, sd;
  if(cw->all_spins) {
    sa=spin_a;
    sb=spin_b;
    sc=spin_c;
    sd=spin_d;
  }
  else {
    sa=cw->spin_a1;
    sb=cw->spin_a2;
    sc=cw->spin_b1;
    sd=cw->spin_b2;
  }
  int nmaps_a=sa ? 2 : 1;
  int nmaps_b=sb ? 2 : 1;
  int nmaps_c=sc ? 2 : 1;
  int nmaps_d=sd ? 2 : 1;

  if((wa->ncls!=nmaps_a*nmaps_b) || (wb->ncls!=nmaps_c*nmaps_d))
    report_error(NMT_ERROR_COVAR,"Input spins don't match input workspaces\n");

#pragma omp parallel default(none)			\
  shared(cw,wa,wb,clac,clad,clbc,clbd)			\
  shared(nmaps_a,nmaps_b,nmaps_c,nmaps_d,covar_out)	\
  shared(has_1122,has_1221)
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
			    if(has_1122) {
			      double fac_1122=0.5*(cl_ac[la]*cl_bd[lb]+cl_ac[lb]*cl_bd[la]);
			      int ind_1122=cov_get_coupling_pair_index(nmaps_a,nmaps_c,nmaps_b,nmaps_d,
								       ia,iap,ic,icp,ib,ibp,id,idp);
			      cov_element+=_pick_xi(cw, ind_1122, 1, la, lb)*fac_1122*prefac_ell;
			    }
			    if(has_1221) {
			      double fac_1221=0.5*(cl_ad[la]*cl_bc[lb]+cl_ad[lb]*cl_bc[la]);
			      int ind_1221=cov_get_coupling_pair_index(nmaps_a,nmaps_d,nmaps_b,nmaps_c,
								       ia,iap,id,idp,ib,ibp,ic,icp);
			      cov_element+=_pick_xi(cw, ind_1221, 0, la, lb)*fac_1221*prefac_ell;
			    }
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
				      int has_1122, int has_1221,
				      flouble **clac,flouble **clad,
				      flouble **clbc,flouble **clbd,
				      flouble *covar_out)
{
  if((cw->lmax<wa->bin->ell_max) || (cw->lmax<wb->bin->ell_max))
    report_error(NMT_ERROR_COVAR,"Coupling coefficients only computed up to l=%d, but you require "
		 "lmax=%d. Recompute this workspace with a larger lmax\n",cw->lmax,wa->bin->ell_max);

  int sa, sb, sc, sd;
  if(cw->all_spins) {
    sa=spin_a;
    sb=spin_b;
    sc=spin_c;
    sd=spin_d;
  }
  else {
    sa=cw->spin_a1;
    sb=cw->spin_a2;
    sc=cw->spin_b1;
    sd=cw->spin_b2;
  }
  int nmaps_a=sa ? 2 : 1;
  int nmaps_b=sb ? 2 : 1;
  int nmaps_c=sc ? 2 : 1;
  int nmaps_d=sd ? 2 : 1;
  if((wa->ncls!=nmaps_a*nmaps_b) || (wb->ncls!=nmaps_c*nmaps_d))
    report_error(NMT_ERROR_COVAR,"Input spins don't match input workspaces\n");

  gsl_matrix *covar_binned=gsl_matrix_alloc(wa->ncls*wa->bin->n_bands,wb->ncls*wb->bin->n_bands);

#pragma omp parallel default(none)			\
  shared(cw,wa,wb,clac,clad,clbc,clbd)			\
  shared(nmaps_a,nmaps_b,nmaps_c,nmaps_d,covar_binned)	\
  shared(has_1122,has_1221)
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
			    if(has_1122) {
			      double fac_1122=0.5*(cl_ac[la]*cl_bd[lb]+cl_ac[lb]*cl_bd[la]);
			      int ind_1122=cov_get_coupling_pair_index(nmaps_a,nmaps_c,nmaps_b,nmaps_d,
								       ia,iap,ic,icp,ib,ibp,id,idp);
			      cbinned+=_pick_xi(cw, ind_1122, 1, la, lb)*fac_1122*prefac_ell;
			    }
			    if(has_1221) {
			      double fac_1221=0.5*(cl_ad[la]*cl_bc[lb]+cl_ad[lb]*cl_bc[la]);
			      int ind_1221=cov_get_coupling_pair_index(nmaps_a,nmaps_d,nmaps_b,nmaps_c,
								       ia,iap,id,idp,ib,ibp,ic,icp);
			      cbinned+=_pick_xi(cw, ind_1221, 0, la, lb)*fac_1221*prefac_ell;
			    }
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
