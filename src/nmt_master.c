#include "config.h"
#include "utils.h"

static void nmt_workspace_store_bins(nmt_workspace *w,
				     nmt_binning_scheme *bin)
{
  int ii;

  w->bin=my_malloc(sizeof(nmt_binning_scheme));
  w->bin->n_bands=bin->n_bands;
  w->bin->nell_list=my_malloc(w->bin->n_bands*sizeof(int));
  memcpy(w->bin->nell_list,bin->nell_list,w->bin->n_bands*sizeof(int));
  w->bin->ell_list=my_malloc(w->bin->n_bands*sizeof(int *));
  w->bin->w_list=my_malloc(w->bin->n_bands*sizeof(flouble *));
  w->bin->f_ell=my_malloc(w->bin->n_bands*sizeof(flouble *));
  for(ii=0;ii<w->bin->n_bands;ii++) {
    w->bin->ell_list[ii]=my_malloc(w->bin->nell_list[ii]*sizeof(int));
    w->bin->w_list[ii]=my_malloc(w->bin->nell_list[ii]*sizeof(flouble));
    w->bin->f_ell[ii]=my_malloc(w->bin->nell_list[ii]*sizeof(flouble));
    memcpy(w->bin->ell_list[ii],bin->ell_list[ii],w->bin->nell_list[ii]*sizeof(int));
    memcpy(w->bin->w_list[ii],bin->w_list[ii],w->bin->nell_list[ii]*sizeof(flouble));
    memcpy(w->bin->f_ell[ii],bin->f_ell[ii],w->bin->nell_list[ii]*sizeof(flouble));
  }
  w->bin->ell_max=bin->ell_max;

}

static nmt_workspace *nmt_workspace_new(int ncls,
					nmt_binning_scheme *bin,int is_teb,
					int lmax_fields,int lmax_mask,
					int norm_type, flouble w2)
{
  int ii;
  nmt_workspace *w=my_malloc(sizeof(nmt_workspace));
  w->lmax=bin->ell_max;
  w->lmax_fields=lmax_fields;
  w->lmax_mask=lmax_mask;
  w->is_teb=is_teb;
  w->ncls=ncls;

  w->pcl_masks=my_malloc((w->lmax_mask+1)*sizeof(flouble));
  w->beam_prod=my_malloc((w->lmax_fields+1)*sizeof(flouble));
  w->norm_type=norm_type;
  w->w2=w2;

  w->coupling_matrix_unbinned=my_malloc(w->ncls*(w->lmax+1)*sizeof(flouble *));
  for(ii=0;ii<w->ncls*(w->lmax+1);ii++)
    w->coupling_matrix_unbinned[ii]=my_calloc(w->ncls*(w->lmax+1),sizeof(flouble));

  nmt_workspace_store_bins(w,bin);

  w->coupling_matrix_binned=gsl_matrix_alloc(w->ncls*w->bin->n_bands,w->ncls*w->bin->n_bands);
  w->coupling_matrix_perm=gsl_permutation_alloc(w->ncls*w->bin->n_bands);

  return w;
}

void nmt_workspace_free(nmt_workspace *w)
{
  int ii;
  gsl_permutation_free(w->coupling_matrix_perm);
  gsl_matrix_free(w->coupling_matrix_binned);
  nmt_bins_free(w->bin);
  if(w->coupling_matrix_unbinned!=NULL) {
    for(ii=0;ii<w->ncls*(w->lmax+1);ii++)
      free(w->coupling_matrix_unbinned[ii]);
    free(w->coupling_matrix_unbinned);
  }
  free(w->beam_prod);
  free(w->pcl_masks);
  free(w);
}

static void bin_coupling_matrix(nmt_workspace *w)
{
  int icl_a,icl_b,ib2,ib3,l2,l3,i2,i3,sig;

  for(icl_a=0;icl_a<w->ncls;icl_a++) {
    for(icl_b=0;icl_b<w->ncls;icl_b++) {
      for(ib2=0;ib2<w->bin->n_bands;ib2++) {
	for(ib3=0;ib3<w->bin->n_bands;ib3++) {
	  double coupling_b=0;
	  if(w->norm_type==0) { //Usual normalisation
	    for(i2=0;i2<w->bin->nell_list[ib2];i2++) {
	      l2=w->bin->ell_list[ib2][i2];
	      for(i3=0;i3<w->bin->nell_list[ib3];i3++) {
		l3=w->bin->ell_list[ib3][i3];
		coupling_b+=w->coupling_matrix_unbinned[w->ncls*l2+icl_a][w->ncls*l3+icl_b]*
		  w->beam_prod[l3]*w->bin->w_list[ib2][i2]*w->bin->f_ell[ib2][i2]/w->bin->f_ell[ib3][i3];
	      }
	    }
	  }
	  else { //FKP normalisation
	    if((w->ncls*ib2+icl_a) == (w->ncls*ib3+icl_b))
	      coupling_b=w->w2;
	  }
	  gsl_matrix_set(w->coupling_matrix_binned,w->ncls*ib2+icl_a,w->ncls*ib3+icl_b,coupling_b);
	}
      }
    }
  }

  gsl_linalg_LU_decomp(w->coupling_matrix_binned,w->coupling_matrix_perm,&sig);
}

void nmt_update_coupling_matrix(nmt_workspace *w,int n_rows,double *new_matrix)
{
  int ii;

  if(n_rows!=w->ncls*(w->lmax+1)) {
    report_error(NMT_ERROR_INCONSISTENT,"Input matrix has the wrong size. Expected %d, got %d\n",
		 w->ncls*(w->lmax+1),n_rows);
  }

  for(ii=0;ii<n_rows;ii++)
    memcpy(w->coupling_matrix_unbinned[ii],&(new_matrix[ii*n_rows]),n_rows*sizeof(flouble));
  bin_coupling_matrix(w);
}

void nmt_workspace_update_beams(nmt_workspace *w,
				int nl1,double *b1,
				int nl2,double *b2)
{
  if((nl1<=w->lmax_fields) || (nl2<=w->lmax_fields)) {
    report_error(NMT_ERROR_INCONSISTENT,
		 "New beams are not large enough\n");
  }

  int ii;
  for(ii=0;ii<=w->lmax_fields;ii++)
    w->beam_prod[ii]=b1[ii]*b2[ii];

  //Recompute the binned coupling matrix
  bin_coupling_matrix(w);
}

void nmt_workspace_update_binning(nmt_workspace *w,
				  nmt_binning_scheme *bin)
{
  if(bin->ell_max!=w->bin->ell_max) {
    report_error(NMT_ERROR_INCONSISTENT,
		 "New bins must have the same ell_max\n");
  }

  //Store new bins
  nmt_bins_free(w->bin);
  nmt_workspace_store_bins(w,bin);

  //Rebin matrix
  gsl_matrix_free(w->coupling_matrix_binned);
  gsl_permutation_free(w->coupling_matrix_perm);
  w->coupling_matrix_binned=gsl_matrix_alloc(w->ncls*w->bin->n_bands,w->ncls*w->bin->n_bands);
  w->coupling_matrix_perm=gsl_permutation_alloc(w->ncls*w->bin->n_bands);
  bin_coupling_matrix(w);
}

static int toeplitz_wrap(int il, int lmaxp1)
{
  if(il<0)
    return il+lmaxp1;
  else if(il>=lmaxp1)
    return il-lmaxp1;
  else
    return il;
}

static int lend_toeplitz(int l2, int l_toeplitz, int l_exact, int dl_band, int lmax)
{
  int l_end;

  if(l_toeplitz > 0) {
    if(l2<=l_exact)
      l_end = lmax;
    else if(l2<=l_toeplitz)
      l_end = l2+dl_band;
    else
      l_end = l2;
  }
  else
    l_end = lmax;

  return fmin(l_end, lmax);
}

static void populate_toeplitz(nmt_master_calculator *c, flouble **pcl_masks, int lt)
{
  int ic,l2;
  double ***tplz_00, ***tplz_0s, ***tplz_pp, ***tplz_mm;
  if(c->has_00) {
    tplz_00=my_malloc(c->npcl*sizeof(flouble **));
    for(ic=0;ic<c->npcl;ic++) {
      tplz_00[ic]=my_malloc(2*sizeof(flouble *));
      tplz_00[ic][0]=my_calloc((c->lmax+1),sizeof(flouble));
      tplz_00[ic][1]=my_calloc((c->lmax+1),sizeof(flouble));
    }
  }
  if(c->has_0s) {
    tplz_0s=my_malloc(c->npcl*sizeof(flouble **));
    for(ic=0;ic<c->npcl;ic++) {
      tplz_0s[ic]=my_malloc(2*sizeof(flouble *));
      tplz_0s[ic][0]=my_calloc((c->lmax+1),sizeof(flouble));
      tplz_0s[ic][1]=my_calloc((c->lmax+1),sizeof(flouble));
    }
  }
  if(c->has_ss) {
    tplz_pp=my_malloc(c->npcl*sizeof(flouble **));
    tplz_mm=my_malloc(c->npcl*sizeof(flouble **));
    for(ic=0;ic<c->npcl;ic++) {
      tplz_pp[ic]=my_malloc(2*sizeof(flouble *));
      tplz_pp[ic][0]=my_calloc((c->lmax+1),sizeof(flouble));
      tplz_pp[ic][1]=my_calloc((c->lmax+1),sizeof(flouble));
      tplz_mm[ic]=my_malloc(2*sizeof(flouble *));
      tplz_mm[ic][0]=my_calloc((c->lmax+1),sizeof(flouble));
      tplz_mm[ic][1]=my_calloc((c->lmax+1),sizeof(flouble));
    }
  }

  int lstart=0;
  int max_spin=NMT_MAX(c->s1, c->s2);
  int has_ss2=(c->s1!=0) && (c->s2!=0) && (c->s1!=c->s2);
  if(!(c->has_00))
    lstart=max_spin;

#pragma omp parallel default(none)              \
  shared(c, lstart, pcl_masks, lt, has_ss2)     \
  shared(tplz_00,tplz_0s,tplz_pp,tplz_mm)
  {
    int il3,ll2,icc;
    int l3_list[2];
    double *wigner_00=NULL,*wigner_ss1=NULL,*wigner_ss2=NULL;
    if(c->has_00 || c->has_0s)
      wigner_00=my_malloc(2*(c->lmax_mask+1)*sizeof(double));
    if(c->has_0s || c->has_ss)
      wigner_ss1=my_malloc(2*(c->lmax_mask+1)*sizeof(double));
    if(has_ss2)
      wigner_ss2=my_malloc(2*(c->lmax_mask+1)*sizeof(double));
    else
      wigner_ss2=wigner_ss1;

    l3_list[0]=0;
    l3_list[1]=lt;

#pragma omp for schedule(dynamic)
    for(ll2=lstart;ll2<=c->lmax;ll2++) {
      l3_list[0]=ll2;  //Diagonal first, then column
      for(il3=0;il3<2;il3++) {
        int ll3=l3_list[il3];
        int l1,lmin_here,lmax_here;
	int lmin_00=0,lmax_00=2*(c->lmax_mask+1)+1;
	int lmin_ss1=0,lmax_ss1=2*(c->lmax_mask+1)+1;
	int lmin_ss2=0,lmax_ss2=2*(c->lmax_mask+1)+1;
        lmin_here=abs(ll2-ll3);
        lmax_here=ll2+ll3;

	if(c->has_00 || c->has_0s)
	  drc3jj_000(ll2,ll3,&lmin_00,&lmax_00,c->lfac,wigner_00,2*(c->lmax_mask+1));
	if(c->has_0s || c->has_ss)
	  drc3jj(ll2,ll3,c->s1,-c->s1,&lmin_ss1,&lmax_ss1,wigner_ss1,2*(c->lmax_mask+1));
        if(has_ss2)
          drc3jj(ll2,ll3,c->s2,-c->s2,&lmin_ss2,&lmax_ss2,wigner_ss2,2*(c->lmax_mask+1));
        else {
          lmin_ss2=lmin_ss1;
          lmax_ss2=lmax_ss1;
        }

	for(l1=lmin_here;l1<=lmax_here;l1++) {
          if(l1<=c->lmax_mask) {
            flouble wfac;
            flouble w00=0,wss1=0,wss2=0;
            int j00=l1-lmin_00;
            int jss1=l1-lmin_ss1;
            int jss2=l1-lmin_ss2;
            if(c->has_00 || c->has_0s)
              w00=j00 < 0 ? 0 : wigner_00[j00];
            if(c->has_ss || c->has_0s) {
              wss1=jss1 < 0 ? 0 : wigner_ss1[jss1];
              wss2=jss2 < 0 ? 0 : wigner_ss2[jss2];
            }

            for(icc=0;icc<c->npcl;icc++) {
              double *pcl=pcl_masks[icc];
              if(c->has_00) {
                wfac=pcl[l1]*w00*w00;
                tplz_00[icc][il3][ll2]+=wfac;
              }
              if(c->has_0s) {
                wfac=pcl[l1]*wss1*w00;
                tplz_0s[icc][il3][ll2]+=wfac;
              }
              if(c->has_ss) {
                int suml=l1+ll2+ll3;
                wfac=pcl[l1]*wss1*wss2;

                if(suml & 1) //Odd sum
                  tplz_mm[icc][il3][ll2]+=wfac;
                else
                  tplz_pp[icc][il3][ll2]+=wfac;
              }
            }
          }
        }
      }
    } //end omp for
    free(wigner_00);
    free(wigner_ss1);
    if(has_ss2)
      free(wigner_ss2);
  } //end omp parallel

  for(ic=0;ic<c->npcl;ic++) {
    //Take absolute value of the diagonal to avoid sqrt(-1) later
    for(l2=0;l2<=c->lmax;l2++) {
      if(c->has_00)
        tplz_00[ic][0][l2]=fabs(tplz_00[ic][0][l2]);
      if(c->has_0s)
        tplz_0s[ic][0][l2]=fabs(tplz_0s[ic][0][l2]);
      if(c->has_ss) {
        tplz_pp[ic][0][l2]=fabs(tplz_pp[ic][0][l2]);
        tplz_mm[ic][0][l2]=fabs(tplz_mm[ic][0][l2]);
      }
    }

    //Compute column correlation coefficient
    for(l2=0;l2<=c->lmax;l2++) {
      double d1,d2;
      if(c->has_00) {
        d1=tplz_00[ic][0][l2];
        d2=tplz_00[ic][0][lt];
        if((d1>0) && (d2>0))
          tplz_00[ic][1][l2]=tplz_00[ic][1][l2]/sqrt(d1*d2);
        else
          tplz_00[ic][1][l2]=0;
      }
      if(c->has_0s) {
        d1=tplz_0s[ic][0][l2];
        d2=tplz_0s[ic][0][lt];
        if((d1>0) && (d2>0))
          tplz_0s[ic][1][l2]=tplz_0s[ic][1][l2]/sqrt(d1*d2);
        else
          tplz_0s[ic][1][l2]=0;
      }
      if(c->has_ss) {
        d1=tplz_pp[ic][0][l2];
        d2=tplz_pp[ic][0][lt];
        if((d1>0) && (d2>0))
          tplz_pp[ic][1][l2]=tplz_pp[ic][1][l2]/sqrt(d1*d2);
        else
          tplz_pp[ic][1][l2]=0;
        d1=tplz_mm[ic][0][l2];
        d2=tplz_mm[ic][0][lt];
        if((d1>0) && (d2>0))
          tplz_mm[ic][1][l2]=tplz_mm[ic][1][l2]/sqrt(d1*d2);
        else
          tplz_mm[ic][1][l2]=0;
      }
    }

    //Populate matrices
#pragma omp parallel default(none)                      \
  shared(c, ic, lt, tplz_00, tplz_0s, tplz_pp, tplz_mm)
    {
      int ll2, ll3;

#pragma omp for schedule(dynamic)
      for(ll2=0;ll2<=c->lmax;ll2++) {
        for(ll3=0;ll3<=ll2;ll3++) {
          int il=toeplitz_wrap(ll2+lt-ll3,c->lmax+1);
          if(c->has_00)
            c->xi_00[ic][ll2][ll3]=tplz_00[ic][1][il]*sqrt(tplz_00[ic][0][ll2]*tplz_00[ic][0][ll3]);
          if(c->has_0s)
            c->xi_0s[ic][0][ll2][ll3]=tplz_0s[ic][1][il]*sqrt(tplz_0s[ic][0][ll2]*tplz_0s[ic][0][ll3]);
          if(c->has_ss) {
            c->xi_pp[ic][0][ll2][ll3]=tplz_pp[ic][1][il]*sqrt(tplz_pp[ic][0][ll2]*tplz_pp[ic][0][ll3]);
            c->xi_mm[ic][0][ll2][ll3]=tplz_mm[ic][1][il]*sqrt(tplz_mm[ic][0][ll2]*tplz_mm[ic][0][ll3]);
          }
          if(ll3!=ll2) {
            if(c->has_00)
              c->xi_00[ic][ll3][ll2]=c->xi_00[ic][ll2][ll3];
            if(c->has_0s)
              c->xi_0s[ic][0][ll3][ll2]=c->xi_0s[ic][0][ll2][ll3];
            if(c->has_ss) {
              c->xi_pp[ic][0][ll3][ll2]=c->xi_pp[ic][0][ll2][ll3];
              c->xi_mm[ic][0][ll3][ll2]=c->xi_mm[ic][0][ll2][ll3];
            }
          }
        }
      } //end omp for
    } //end omp parallel
  }

  if(c->has_ss) {
    for(ic=0;ic<c->npcl;ic++) {
      free(tplz_pp[ic][0]);
      free(tplz_pp[ic][1]);
      free(tplz_pp[ic]);
      free(tplz_mm[ic][0]);
      free(tplz_mm[ic][1]);
      free(tplz_mm[ic]);
    }
    free(tplz_pp);
    free(tplz_mm);
  }
  if(c->has_0s) {
    for(ic=0;ic<c->npcl;ic++) {
      free(tplz_0s[ic][0]);
      free(tplz_0s[ic][1]);
      free(tplz_0s[ic]);
    }
    free(tplz_0s);
  }
  if(c->has_00) {
    for(ic=0;ic<c->npcl;ic++) {
      free(tplz_00[ic][0]);
      free(tplz_00[ic][1]);
      free(tplz_00[ic]);
    }
    free(tplz_00);
  }
}

nmt_workspace *nmt_compute_coupling_matrix_anisotropic(int spin1, int spin2,
						       int mask_aniso_1, int mask_aniso_2,
						       int lmax, int lmax_mask,
						       flouble *pcl_masks_00,
						       flouble *pcl_masks_0e,
						       flouble *pcl_masks_e0,
						       flouble *pcl_masks_0b,
						       flouble *pcl_masks_b0,
						       flouble *pcl_masks_ee,
						       flouble *pcl_masks_eb,
						       flouble *pcl_masks_be,
						       flouble *pcl_masks_bb,
						       flouble *beam1,flouble *beam2,
						       nmt_binning_scheme *bin,
						       int norm_type,flouble w2)
{
  int l2,lmax_large,lmax_fields;
  int n_cl,nmaps1=1,nmaps2=1;
  nmt_workspace *w;
  if(spin1) nmaps1=2;
  if(spin2) nmaps2=2;
  n_cl=nmaps1*nmaps2;

  if(bin->ell_max>lmax)
    report_error(NMT_ERROR_CONSISTENT_RESO,
		 "Requesting bandpowers for too high a "
		 "multipole given map resolution\n");
  lmax_fields=lmax; // ell_max for the maps
  lmax_large=lmax_mask; // ell_max for the masks


  w=nmt_workspace_new(n_cl,bin,0,
		      lmax_fields,lmax_large,norm_type,w2);

  for(l2=0;l2<=w->lmax_fields;l2++)
    w->beam_prod[l2]=beam1[l2]*beam2[l2];

  for(l2=0;l2<=w->lmax_mask;l2++)
    w->pcl_masks[l2]=pcl_masks_00[l2]*(2*l2+1.)/(4*M_PI);

  int i_00=0, i_0e=1, i_0b=2, i_e0=3, i_b0=4, i_ee=5, i_eb=6, i_be=7, i_bb=8;
  flouble **pcl_masks=my_malloc(9*sizeof(flouble *));
  for(l2=0;l2<9;l2++)
    pcl_masks[l2]=my_calloc(lmax_mask+1,sizeof(flouble));
  memcpy(pcl_masks[i_00], pcl_masks_00, (lmax_mask+1)*sizeof(flouble));
  if(mask_aniso_2) {
    memcpy(pcl_masks[i_0e], pcl_masks_0e, (lmax_mask+1)*sizeof(flouble));
    memcpy(pcl_masks[i_0b], pcl_masks_0b, (lmax_mask+1)*sizeof(flouble));
  }
  if(mask_aniso_1) {
    memcpy(pcl_masks[i_e0], pcl_masks_e0, (lmax_mask+1)*sizeof(flouble));
    memcpy(pcl_masks[i_b0], pcl_masks_b0, (lmax_mask+1)*sizeof(flouble));
    if(mask_aniso_2) {
      memcpy(pcl_masks[i_ee], pcl_masks_ee, (lmax_mask+1)*sizeof(flouble));
      memcpy(pcl_masks[i_eb], pcl_masks_eb, (lmax_mask+1)*sizeof(flouble));
      memcpy(pcl_masks[i_be], pcl_masks_be, (lmax_mask+1)*sizeof(flouble));
      memcpy(pcl_masks[i_bb], pcl_masks_bb, (lmax_mask+1)*sizeof(flouble));
    }
  }

  int sign_00 = 1;
  int sign_0P = (spin2 & 1) ? -1 : 1;
  int sign_P0 = (spin1 & 1) ? -1 : 1;
  int sign_PP = ((spin1+spin2) & 1) ? -1 : 1;
  for(l2=0;l2<=lmax_mask;l2++) {
    flouble fac=(2*l2+1)/(4*M_PI);
    pcl_masks[i_00][l2] *= sign_00*fac;
    pcl_masks[i_0e][l2] *= sign_0P*fac;
    pcl_masks[i_0b][l2] *= sign_0P*fac;
    pcl_masks[i_e0][l2] *= sign_P0*fac;
    pcl_masks[i_b0][l2] *= sign_P0*fac;
    pcl_masks[i_ee][l2] *= sign_PP*fac;
    pcl_masks[i_eb][l2] *= sign_PP*fac;
    pcl_masks[i_be][l2] *= sign_PP*fac;
    pcl_masks[i_bb][l2] *= sign_PP*fac;
  }

  int max_spin=NMT_MAX(spin1, spin2);
  int lstart=max_spin;
  int reuse_ss1=(spin1==spin2);
  int is_0s=(spin1==0) && (spin2!=0);
  int is_s0=(spin1!=0) && (spin2==0);
  int is_ss=(spin1!=0) && (spin2!=0);

#pragma omp parallel default(none)				\
  shared(pcl_masks, lstart, reuse_ss1, lmax, lmax_mask)		\
  shared(mask_aniso_1, mask_aniso_2, is_0s, is_s0, is_ss)	\
  shared(spin1, spin2, i_00, i_0e, i_0b, i_e0, i_b0)		\
  shared(i_ee, i_eb, i_be, i_bb, w)
  {
    int ll2,ll3;
    double *wigner_ss1=NULL, *wigner_ss2=NULL, *wigner_ss1a=NULL, *wigner_ss2a=NULL;
    // s -s 0 terms
    wigner_ss1=my_malloc(2*(lmax_mask+1)*sizeof(double));
    if(reuse_ss1)
      wigner_ss2=wigner_ss1;
    else
      wigner_ss2=my_malloc(2*(lmax_mask+1)*sizeof(double));

    // s s -2s terms
    if(mask_aniso_1)
      wigner_ss1a=my_malloc(2*(lmax_mask+1)*sizeof(double));
    if(mask_aniso_2) {
      if((!reuse_ss1) || (mask_aniso_1 == 0))
	wigner_ss2a=my_malloc(2*(lmax_mask+1)*sizeof(double));
      else
	wigner_ss2a=wigner_ss1a;
    }

#pragma omp for schedule(dynamic)
    for(ll2=lstart;ll2<=lmax;ll2++) {
      for(ll3=lstart;ll3<=lmax;ll3++) {  //TODO: investigate if there are ell-ell' symmetries to exploit
	int l1,lmin_here,lmax_here;
	int l3fac=2*ll3+1;
	int lmin_ss1=0,lmax_ss1=2*(lmax_mask+1)+1;
	int lmin_ss2=0,lmax_ss2=2*(lmax_mask+1)+1;
	int lmin_ss1a=0,lmax_ss1a=2*(lmax_mask+1)+1;
	int lmin_ss2a=0,lmax_ss2a=2*(lmax_mask+1)+1;
	lmin_here=abs(ll2-ll3);
	lmax_here=ll2+ll3;

	// s -s 0 terms
	drc3jj(ll2,ll3,spin1,-spin1,&lmin_ss1,&lmax_ss1,wigner_ss1,2*(lmax_mask+1));
	if(reuse_ss1) {
	  lmin_ss2=lmin_ss1;
	  lmax_ss2=lmax_ss1;
	}
	else
	  drc3jj(ll2,ll3,spin2,-spin2,&lmin_ss2,&lmax_ss2,wigner_ss2,2*(lmax_mask+1));

	// s s -2s terms
	if(mask_aniso_1)
	  drc3jj(ll2,ll3,spin1,spin1,&lmin_ss1a,&lmax_ss1a,wigner_ss1a,2*(lmax_mask+1));
	if(mask_aniso_2) {
	  if((!reuse_ss1) || (mask_aniso_1 == 0))
	    drc3jj(ll2,ll3,spin2,spin2,&lmin_ss2a,&lmax_ss2a,wigner_ss2a,2*(lmax_mask+1));
	  else {
	    lmin_ss2a=lmin_ss1a;
	    lmax_ss2a=lmax_ss1a;
	  }
	}

	for(l1=lmin_here;l1<=lmax_here;l1++) {
	  if(l1<=lmax_mask) {
	    flouble m00=0,m0e=0,m0b=0,me0=0,mb0=0,mee=0,meb=0,mbe=0,mbb=0;
	    int suml=l1+ll2+ll3;
	    flouble wss1=0,wss2=0,wss1a=0,wss2a=0;
	    int jss1=l1-lmin_ss1;
	    int jss2=l1-lmin_ss2;
	    int jss1a=l1-lmin_ss1a;
	    int jss2a=l1-lmin_ss2a;
	    int splus= (suml & 1) ? 0 : 1;
	    int sminus= (suml & 1) ? 1 : 0;
	    wss1=jss1 < 0 ? 0 : wigner_ss1[jss1];
	    wss2=jss2 < 0 ? 0 : wigner_ss2[jss2];
	    if(mask_aniso_1)
	      wss1a=jss1a < 0 ? 0 : wigner_ss1a[jss1a];
	    if(mask_aniso_2)
	      wss2a=jss2a < 0 ? 0 : wigner_ss2a[jss2a];

	    m00 = wss1*wss2*pcl_masks[i_00][l1];
	    if(mask_aniso_2) {
	      m0e = wss1*wss2a*pcl_masks[i_0e][l1];
	      m0b = wss1*wss2a*pcl_masks[i_0b][l1];
	    }
	    if(mask_aniso_1) {
	      me0 = wss1a*wss2*pcl_masks[i_e0][l1];
	      mb0 = wss1a*wss2*pcl_masks[i_b0][l1];
	      if(mask_aniso_2) {
		mee = wss1a*wss2a*pcl_masks[i_ee][l1];
		meb = wss1a*wss2a*pcl_masks[i_eb][l1];
		mbe = wss1a*wss2a*pcl_masks[i_be][l1];
		mbb = wss1a*wss2a*pcl_masks[i_bb][l1];
	      }
	    }
	    if(is_0s) {
	      w->coupling_matrix_unbinned[2*ll2+0][2*ll3+0] += l3fac*(m00-m0e); //0E,0E
	      w->coupling_matrix_unbinned[2*ll2+0][2*ll3+1] += l3fac*(-m0b);    //0E,0B
	      w->coupling_matrix_unbinned[2*ll2+1][2*ll3+0] += l3fac*(-m0b);    //0B,0E
	      w->coupling_matrix_unbinned[2*ll2+1][2*ll3+1] += l3fac*(m00+m0e); //0B,0B
	    }
	    if(is_s0) {
	      w->coupling_matrix_unbinned[2*ll2+0][2*ll3+0] += l3fac*(m00-me0); //E0,E0
	      w->coupling_matrix_unbinned[2*ll2+0][2*ll3+1] += l3fac*(-mb0);    //E0,B0
	      w->coupling_matrix_unbinned[2*ll2+1][2*ll3+0] += l3fac*(-mb0);    //B0,E0
	      w->coupling_matrix_unbinned[2*ll2+1][2*ll3+1] += l3fac*(m00+me0); //B0,B0
	    }
	    if(is_ss) {
	      w->coupling_matrix_unbinned[4*ll2+0][4*ll3+0] += l3fac*(splus * (m00-m0e-me0+mee) + sminus * mbb);                //EE,EE
	      w->coupling_matrix_unbinned[4*ll2+0][4*ll3+1] += l3fac*(splus * (-m0b+meb)        + sminus * (-mb0-mbe));         //EE,EB
	      w->coupling_matrix_unbinned[4*ll2+0][4*ll3+2] += l3fac*(splus * (-mb0+mbe)        + sminus * (-m0b-meb));         //EE,BE
	      w->coupling_matrix_unbinned[4*ll2+0][4*ll3+3] += l3fac*(splus * mbb               + sminus * (m00+m0e+me0+mee));  //EE,BB
	      w->coupling_matrix_unbinned[4*ll2+1][4*ll3+0] += l3fac*(splus * (-m0b+meb)        + sminus * (mb0-mbe));          //EB,EE
	      w->coupling_matrix_unbinned[4*ll2+1][4*ll3+1] += l3fac*(splus * (m00+m0e-me0-mee) + sminus * (-mbb));             //EB,EB
	      w->coupling_matrix_unbinned[4*ll2+1][4*ll3+2] += l3fac*(splus * mbb               + sminus * (-m00+m0e-me0+mee)); //EB,BE
	      w->coupling_matrix_unbinned[4*ll2+1][4*ll3+3] += l3fac*(splus * (-mb0-mbe)        + sminus * (m0b+meb));          //EB,BB
	      w->coupling_matrix_unbinned[4*ll2+2][4*ll3+0] += l3fac*(splus * (-mb0+mbe)        + sminus * (m0b-meb));          //BE,EE
	      w->coupling_matrix_unbinned[4*ll2+2][4*ll3+1] += l3fac*(splus * mbb               + sminus * (-m00-m0e+me0+mee)); //BE,EB
	      w->coupling_matrix_unbinned[4*ll2+2][4*ll3+2] += l3fac*(splus * (m00-m0e+me0-mee) + sminus * (-mbb));             //BE,BE
	      w->coupling_matrix_unbinned[4*ll2+2][4*ll3+3] += l3fac*(splus * (-m0b-meb)        + sminus * (mb0+mbe));          //BE,BB
	      w->coupling_matrix_unbinned[4*ll2+3][4*ll3+0] += l3fac*(splus * mbb               + sminus * (m00-m0e-me0+mee));  //BB,EE
	      w->coupling_matrix_unbinned[4*ll2+3][4*ll3+1] += l3fac*(splus * (-mb0-mbe)        + sminus * (-m0b+meb));         //BB,EB
	      w->coupling_matrix_unbinned[4*ll2+3][4*ll3+2] += l3fac*(splus * (-m0b-meb)        + sminus * (-mb0+mbe));         //BB,BE
	      w->coupling_matrix_unbinned[4*ll2+3][4*ll3+3] += l3fac*(splus * (m00+m0e+me0+mee) + sminus * mbb);                //BB,BB
	    }
	  }
	}
      }
    } //end omp for

    // s -s 0 terms
    free(wigner_ss1);
    if(!reuse_ss1)
      free(wigner_ss2);
    // s s -2s terms
    if(mask_aniso_1)
      free(wigner_ss1a);
    if(mask_aniso_2) {
      if((!reuse_ss1) || (mask_aniso_1 == 0))
	free(wigner_ss2a);
    }
  } //end omp parallel

  for(l2=0;l2<9;l2++)
    free(pcl_masks[l2]);
  free(pcl_masks);

  bin_coupling_matrix(w);

  return w;
}

nmt_master_calculator *nmt_compute_master_coefficients(int lmax, int lmax_mask,
                                                       int npcl, flouble **pcl_masks,
                                                       int s1, int s2,
                                                       int pure_e1, int pure_b1,
                                                       int pure_e2, int pure_b2,
                                                       int do_teb, int l_toeplitz,
                                                       int l_exact, int dl_band)
{
  int ic, ip, ii;
  nmt_master_calculator *c=my_malloc(sizeof(nmt_master_calculator));
  c->pure_any=pure_e1 || pure_b1 || pure_e2 || pure_b2;
  c->npcl=npcl;
  c->lmax=lmax;
  c->lmax_mask=lmax_mask;
  c->pure_e1=pure_e1;
  c->pure_b1=pure_b1;
  c->pure_e2=pure_e2;
  c->pure_b2=pure_b2;
  c->has_00=0;
  c->has_0s=0;
  c->has_ss=0;
  c->xi_00=NULL;
  c->xi_0s=NULL;
  c->xi_pp=NULL;
  c->xi_mm=NULL;
  if(c->pure_any) {
    c->npure_0s=2;
    c->npure_ss=3;
  }
  else {
    c->npure_0s=1;
    c->npure_ss=1;
  }

  if(s1==0) {
    if(s2==0) {
      c->s1=0; c->s2=0;
    }
    else {
      c->s1=s2; c->s2=0;
    }
  }
  else {
    c->s1=s1;
    c->s2=s2;
  }

  if(do_teb) {
    c->has_00=1;
    c->has_0s=1;
    c->has_ss=1;
  }
  else {
    c->has_00 = (c->s1==0) && (c->s2==0);
    c->has_0s = ((c->s1==0) && (c->s2!=0)) || ((c->s1!=0) && (c->s2==0));
    c->has_ss = (c->s1!=0) && (c->s2!=0);
  }

  if(c->has_00) {
    c->xi_00=my_malloc(c->npcl*sizeof(flouble **));
    for(ic=0;ic<c->npcl;ic++) {
        c->xi_00[ic]=my_malloc((c->lmax+1)*sizeof(flouble *));
        for(ii=0;ii<=c->lmax;ii++)
          c->xi_00[ic][ii]=my_calloc((c->lmax+1),sizeof(flouble));
    }
  }
  if(c->has_0s) {
    c->xi_0s=my_malloc(c->npcl*sizeof(flouble ***));
    for(ic=0;ic<c->npcl;ic++) {
      c->xi_0s[ic]=my_malloc(c->npure_0s*sizeof(flouble **));
      for(ip=0;ip<c->npure_0s;ip++) {
        c->xi_0s[ic][ip]=my_malloc((c->lmax+1)*sizeof(flouble *));
        for(ii=0;ii<=c->lmax;ii++)
          c->xi_0s[ic][ip][ii]=my_calloc((c->lmax+1),sizeof(flouble));
      }
    }
  }
  if(c->has_ss) {
    c->xi_pp=my_malloc(c->npcl*sizeof(flouble ***));
    c->xi_mm=my_malloc(c->npcl*sizeof(flouble ***));
    for(ic=0;ic<c->npcl;ic++) {
      c->xi_pp[ic]=my_malloc(c->npure_ss*sizeof(flouble **));
      c->xi_mm[ic]=my_malloc(c->npure_ss*sizeof(flouble **));
      for(ip=0;ip<c->npure_ss;ip++) {
        c->xi_pp[ic][ip]=my_malloc((c->lmax+1)*sizeof(flouble *));
        c->xi_mm[ic][ip]=my_malloc((c->lmax+1)*sizeof(flouble *));
        for(ii=0;ii<=c->lmax;ii++) {
          c->xi_pp[ic][ip][ii]=my_calloc((c->lmax+1),sizeof(flouble));
          c->xi_mm[ic][ip][ii]=my_calloc((c->lmax+1),sizeof(flouble));
        }
      }
    }
  }

  int lmax_max = NMT_MAX(c->lmax, c->lmax_mask);
  c->lfac = my_malloc(3*(lmax_max+1)*sizeof(double));
  // Precompute log-factorial
  c->lfac[0] = 0.0;
  c->lfac[1] = 0.0;
  for(ii=2;ii<3*(lmax_max+1);ii++)
    c->lfac[ii] = c->lfac[ii-1]+log((double)(ii));

  if(l_toeplitz>0)
    populate_toeplitz(c, pcl_masks, l_toeplitz);

  int lstart=0;
  int max_spin=NMT_MAX(c->s1, c->s2);
  int has_ss2=(c->s1!=0) && (c->s2!=0) && (!do_teb) && (c->s1!=c->s2);
  if(!(c->has_00))
    lstart=max_spin;

#pragma omp parallel default(none)              \
  shared(c, lstart, do_teb, pcl_masks, has_ss2) \
  shared(l_toeplitz, l_exact, dl_band)
  {
    int ll2,ll3,icc;
    double *wigner_00=NULL,*wigner_ss1=NULL,*wigner_12=NULL,*wigner_02=NULL,*wigner_ss2=NULL;
    if(c->has_00 || c->has_0s)
      wigner_00=my_malloc(2*(c->lmax_mask+1)*sizeof(double));
    if(c->has_0s || c->has_ss)
      wigner_ss1=my_malloc(2*(c->lmax_mask+1)*sizeof(double));
    if(has_ss2)
      wigner_ss2=my_malloc(2*(c->lmax_mask+1)*sizeof(double));
    else
      wigner_ss2=wigner_ss1;
    if(c->pure_any) {
      wigner_12=my_malloc(2*(c->lmax_mask+1)*sizeof(double));
      wigner_02=my_malloc(2*(c->lmax_mask+1)*sizeof(double));
    }

#pragma omp for schedule(dynamic)
    for(ll2=lstart;ll2<=c->lmax;ll2++) {
      int l3_end=lend_toeplitz(ll2, l_toeplitz, l_exact, dl_band, c->lmax);
      int l3_start=lstart;
      if(!(c->pure_any)) //We can use symmetry
        l3_start=ll2;
      for(ll3=l3_start;ll3<=l3_end;ll3++) {
        int l1,lmin_here,lmax_here;
        int lmin_00=0,lmax_00=2*(c->lmax_mask+1)+1;
        int lmin_ss1=0,lmax_ss1=2*(c->lmax_mask+1)+1;
        int lmin_ss2=0,lmax_ss2=2*(c->lmax_mask+1)+1;
        int lmin_12=0,lmax_12=2*(c->lmax_mask+1)+1;
        int lmin_02=0,lmax_02=2*(c->lmax_mask+1)+1;
        lmin_here=abs(ll2-ll3);
        lmax_here=ll2+ll3;

        if(l_toeplitz > 0) {
          //Set all elements that will be recomputed to zero
          for(icc=0;icc<c->npcl;icc++) {
            if(c->has_00)
              c->xi_00[icc][ll2][ll3]=0;
            if(c->has_0s)
              c->xi_0s[icc][0][ll2][ll3]=0;
            if(c->has_ss) {
              c->xi_pp[icc][0][ll2][ll3]=0;
              c->xi_mm[icc][0][ll2][ll3]=0;
            }
          }
        }

        if(c->has_00 || c->has_0s)
	  drc3jj_000(ll2,ll3,&lmin_00,&lmax_00,c->lfac,wigner_00,2*(c->lmax_mask+1));
        if(c->has_0s || c->has_ss)
          drc3jj(ll2,ll3,c->s1,-c->s1,&lmin_ss1,&lmax_ss1,wigner_ss1,2*(c->lmax_mask+1));
        if(has_ss2)
          drc3jj(ll2,ll3,c->s2,-c->s2,&lmin_ss2,&lmax_ss2,wigner_ss2,2*(c->lmax_mask+1));
        else {
          lmin_ss2=lmin_ss1;
          lmax_ss2=lmax_ss1;
        }
	if(c->pure_any) {
	  drc3jj(ll2,ll3,1,-2,&lmin_12,&lmax_12,wigner_12,2*(c->lmax_mask+1));
	  drc3jj(ll2,ll3,0,-2,&lmin_02,&lmax_02,wigner_02,2*(c->lmax_mask+1));
	}

        for(l1=lmin_here;l1<=lmax_here;l1++) {
          int ipp;
          if(l1<=c->lmax_mask) {
            flouble wfac,fac_12=0,fac_02=0;
            flouble w00=0,wss1=0,wss2=0,w12=0,w02=0;
            int j02,j12;
            int j00=l1-lmin_00;
            int jss1=l1-lmin_ss1;
            int jss2=l1-lmin_ss2;
            if(c->has_00 || c->has_0s)
              w00=j00 < 0 ? 0 : wigner_00[j00];
            if(c->has_ss || c->has_0s) {
              wss1=jss1 < 0 ? 0 : wigner_ss1[jss1];
              wss2=jss2 < 0 ? 0 : wigner_ss2[jss2];
            }

            if(c->pure_any) {
	      j12=l1-lmin_12;
	      j02=l1-lmin_02;
	      if(ll2>1.) {
		fac_12=2*sqrt((l1+1.)*(l1+0.)/((ll2+2)*(ll2-1.)));
		if(l1>1.)
		  fac_02=sqrt((l1+2.)*(l1+1.)*(l1+0.)*(l1-1.)/((ll2+2.)*(ll2+1.)*(ll2+0.)*(ll2-1.)));
		else
		  fac_02=0;
	      }
	      else {
		fac_12=0;
		fac_02=0;
	      }
	      if(j12<0) { //If out of range, w12 is just 0
		fac_12=0;
		j12=0;
	      }
	      if(j02<0) { //if out of range, w02 is just 0
		fac_02=0;
		j02=0;
	      }
              w12=j12 < 0 ? 0 : wigner_12[j12];
              w02=j02 < 0 ? 0 : wigner_02[j02];
	    }

            for(icc=0;icc<c->npcl;icc++) {
              double *pcl=pcl_masks[icc];
              if(c->has_00) {
                wfac=pcl[l1]*w00*w00;
                c->xi_00[icc][ll2][ll3]+=wfac;
              }
              if(c->has_0s) {
                double wfac_ispure[2];
                wfac_ispure[0]=wss1;
                wfac_ispure[0]*=pcl[l1]*w00;
                if(c->pure_any) {
                  wfac_ispure[1]=wss1+fac_12*w12+fac_02*w02;
                  wfac_ispure[1]*=pcl[l1]*w00;
                }
                for(ipp=0;ipp<c->npure_0s;ipp++)
                  c->xi_0s[icc][ipp][ll2][ll3]+=wfac_ispure[ipp];
              }
              if(c->has_ss) {
                double wfac_ispure[3];
                int suml=l1+ll2+ll3;
                wfac_ispure[0]=wss1;
                wfac_ispure[0]*=wss2*pcl[l1];
                if(c->pure_any) {
                  wfac_ispure[1]=wss1+fac_12*w12+fac_02*w02;
                  wfac_ispure[2]=wfac_ispure[1]*wfac_ispure[1]*pcl[l1];
                  wfac_ispure[1]*=wss2*pcl[l1];
                }

                if(suml & 1) { //Odd sum
                  for(ipp=0;ipp<c->npure_ss;ipp++)
                    c->xi_mm[icc][ipp][ll2][ll3]+=wfac_ispure[ipp];
                }
                else {
                  for(ipp=0;ipp<c->npure_ss;ipp++)
                    c->xi_pp[icc][ipp][ll2][ll3]+=wfac_ispure[ipp];
                }
              }
	    }
	  }
	}

        if((!(c->pure_any)) && (ll2 != ll3)) { //Can use symmetry
          for(icc=0;icc<c->npcl;icc++) {
            if(c->has_00)
              c->xi_00[icc][ll3][ll2]=c->xi_00[icc][ll2][ll3];
            if(c->has_0s)
              c->xi_0s[icc][0][ll3][ll2]=c->xi_0s[icc][0][ll2][ll3];
            if(c->has_ss) {
              c->xi_pp[icc][0][ll3][ll2]=c->xi_pp[icc][0][ll2][ll3];
              c->xi_mm[icc][0][ll3][ll2]=c->xi_mm[icc][0][ll2][ll3];
            }
          }
        }
      }
    } //end omp for
    free(wigner_00);
    free(wigner_ss1);
    if(has_ss2)
      free(wigner_ss2);
    free(wigner_12);
    free(wigner_02);
  } //end omp parallel

  // Fill out lower triangle
  if(l_toeplitz > 0) {
    int l2, l3;
    for(l2=c->lmax+l_exact-l_toeplitz;l2<=c->lmax;l2++) {
      for(l3=l_exact;l3<=l2+l_toeplitz-c->lmax;l3++){
        flouble **mat;
        flouble m;
	int is_den_zero;
        int lx=l_exact+l2-l3;
        for(ic=0;ic<c->npcl;ic++) {
          if(c->has_00) {
            mat = c->xi_00[ic];
	    is_den_zero=((mat[lx][lx] == 0) || (mat[l_exact][l_exact] == 0));
	    if(is_den_zero)
	      m=0;
	    else {
	      m=mat[lx][l_exact]*sqrt(fabs(mat[l2][l2]*mat[l3][l3]/
					   (mat[lx][lx]*mat[l_exact][l_exact])));
	    }
            mat[l2][l3]=m;
            mat[l3][l2]=m;
          }
          if(c->has_0s) {
            mat = c->xi_0s[ic][0];
	    is_den_zero=((mat[lx][lx] == 0) || (mat[l_exact][l_exact] == 0));
	    if(is_den_zero)
	      m=0;
	    else {
	      m=mat[lx][l_exact]*sqrt(fabs(mat[l2][l2]*mat[l3][l3]/
					   (mat[lx][lx]*mat[l_exact][l_exact])));
	    }
            mat[l2][l3]=m;
            mat[l3][l2]=m;
          }
          if(c->has_ss) {
            mat = c->xi_pp[ic][0];
	    is_den_zero=((mat[lx][lx] == 0) || (mat[l_exact][l_exact] == 0));
	    if(is_den_zero)
	      m=0;
	    else {
	      m=mat[lx][l_exact]*sqrt(fabs(mat[l2][l2]*mat[l3][l3]/
					   (mat[lx][lx]*mat[l_exact][l_exact])));
	    }
            mat[l2][l3]=m;
            mat[l3][l2]=m;
            mat = c->xi_mm[ic][0];
	    is_den_zero=((mat[lx][lx] == 0) || (mat[l_exact][l_exact] == 0));
	    if(is_den_zero)
	      m=0;
	    else {
	      m=mat[lx][l_exact]*sqrt(fabs(mat[l2][l2]*mat[l3][l3]/
					   (mat[lx][lx]*mat[l_exact][l_exact])));
	    }
            mat[l2][l3]=m;
            mat[l3][l2]=m;
          }
        }
      }
    }
  }

  return c;
}

void nmt_compute_general_coupling_matrix(int lmax,
					 flouble *pcl_mask,
					 int s1, int s2,
					 int n1, int n2,
					 flouble *xi_out)
{
  int nls=lmax+1;
  int sign = ((n1+n2) & 1) ? -1 : 1;

#pragma omp parallel default(none)				\
  shared(lmax, pcl_mask, s1, s2, n1, n2, xi_out, nls, sign)
  {
    int same_sn=(s1 == s2) && (n1 == n2);
    int ll2,ll3,icc;
    int lstart=NMT_MAX(s1, s2);
    flouble *wl_mask=my_malloc((lmax+1)*sizeof(flouble));
    double *wigner_sn1=NULL,*wigner_sn2=NULL;
    wigner_sn1=my_malloc(2*(lmax+1)*sizeof(double));
    if(same_sn)
      wigner_sn2=wigner_sn1;
    else
      wigner_sn2=my_malloc(2*(lmax+1)*sizeof(double));

    for(ll2=0;ll2<=lmax;ll2++)
      wl_mask[ll2]=pcl_mask[ll2]*(2*ll2+1)/(4*M_PI);

#pragma omp for schedule(dynamic)
    for(ll2=lstart;ll2<=lmax;ll2++) {
      for(ll3=lstart;ll3<=lmax;ll3++) {
        int l1,lmin_here,lmax_here;
	int lmin_sn1=0,lmax_sn1=2*(lmax+1)+1;
	int lmin_sn2=0,lmax_sn2=2*(lmax+1)+1;
	int index=ll3+(lmax+1)*ll2;
        lmin_here=abs(ll2-ll3);
        lmax_here=ll2+ll3;


	drc3jj(ll2,ll3,n1,-s1,&lmin_sn1,&lmax_sn1,wigner_sn1,2*(lmax+1));
	if(same_sn) {
	  wigner_sn2=wigner_sn1;
	  lmin_sn2=lmin_sn1;
	  lmax_sn2=lmax_sn1;
	}
	else
	  drc3jj(ll2,ll3,n2,-s2,&lmin_sn2,&lmax_sn2,wigner_sn2,2*(lmax+1));

        for(l1=lmin_here;l1<=lmax_here;l1++) {
          if(l1<=lmax) {
	    int jsn1=l1-lmin_sn1;
	    int jsn2=l1-lmin_sn2;
	    flouble wsn1=0,wsn2=0;
	    wsn1=jsn1 < 0 ? 0 : wigner_sn1[jsn1];
	    wsn2=jsn2 < 0 ? 0 : wigner_sn2[jsn2];
	    //if(!((l1+ll2+ll3) & 1)) //Even sum
	    //if((l1+ll2+ll3) & 1) //Even sum
	    xi_out[index] += wl_mask[l1]*wsn1*wsn2;
	  }
	}
	xi_out[index] *= (2*ll3+1.0); //*sign
      }
    } //end omp for
    free(wl_mask);
    free(wigner_sn1);
    if(!same_sn)
      free(wigner_sn2);
  } //end omp parallel
}

void nmt_master_calculator_free(nmt_master_calculator *c)
{
  int ii, ip, ic;

  if(c->has_00) {
    for(ic=0;ic<c->npcl;ic++) {
      for(ii=0;ii<=c->lmax;ii++)
        free(c->xi_00[ic][ii]);
      free(c->xi_00[ic]);
    }
    free(c->xi_00);
  }
  if(c->has_0s) {
    for(ic=0;ic<c->npcl;ic++) {
      for(ip=0;ip<c->npure_0s;ip++) {
        for(ii=0;ii<=c->lmax;ii++)
          free(c->xi_0s[ic][ip][ii]);
        free(c->xi_0s[ic][ip]);
      }
      free(c->xi_0s[ic]);
    }
    free(c->xi_0s);
  }
  if(c->has_ss) {
    for(ic=0;ic<c->npcl;ic++) {
      for(ip=0;ip<c->npure_ss;ip++) {
        for(ii=0;ii<=c->lmax;ii++) {
          free(c->xi_pp[ic][ip][ii]);
          free(c->xi_mm[ic][ip][ii]);
        }
        free(c->xi_pp[ic][ip]);
        free(c->xi_mm[ic][ip]);
      }
      free(c->xi_pp[ic]);
      free(c->xi_mm[ic]);
    }
    free(c->xi_pp);
    free(c->xi_mm);
  }
  free(c->lfac);
  free(c);
}

//Computes binned coupling matrix
// fl1,fl2 (in) : fields we're correlating
// coupling_matrix_out (out) : unbinned coupling matrix
nmt_workspace *nmt_compute_coupling_matrix(int spin1,int spin2,
					   int lmax, int lmax_mask,
					   int pure_e1,int pure_b1,
					   int pure_e2,int pure_b2,
					   flouble *pcl_masks,
					   flouble *beam1,flouble *beam2,
					   nmt_binning_scheme *bin,int is_teb,
                                           int l_toeplitz,int l_exact,int dl_band,
					   int norm_type,flouble w2)
{
  int l2,lmax_large,lmax_fields;
  nmt_workspace *w;
  int n_cl,nmaps1=1,nmaps2=1;
  if(spin1) nmaps1=2;
  if(spin2) nmaps2=2;
  n_cl=nmaps1*nmaps2;
  if(is_teb) {
    if(!((spin1==0) && (spin2!=0)))
      report_error(NMT_ERROR_INCONSISTENT,"For T-E-B MCM the first input field must be spin-0 and the second spin-!=0\n");
    n_cl=7;
  }

  if(bin->ell_max>lmax)
    report_error(NMT_ERROR_CONSISTENT_RESO,
		 "Requesting bandpowers for too high a "
		 "multipole given map resolution\n");
  lmax_fields=lmax; // ell_max for the maps
  lmax_large=lmax_mask; // ell_max for the masks
  w=nmt_workspace_new(n_cl,bin,is_teb,
		      lmax_fields,lmax_large,norm_type,w2);

  for(l2=0;l2<=w->lmax_fields;l2++)
    w->beam_prod[l2]=beam1[l2]*beam2[l2];

  for(l2=0;l2<=w->lmax_mask;l2++)
    w->pcl_masks[l2]=pcl_masks[l2]*(2*l2+1.)/(4*M_PI);

  // Compute coupling coefficients
  nmt_master_calculator *c=nmt_compute_master_coefficients(w->lmax, w->lmax_mask,
                                                           1, &(w->pcl_masks),
                                                           spin1, spin2,
                                                           pure_e1,pure_b1,
                                                           pure_e2,pure_b2,
                                                           is_teb, l_toeplitz, l_exact, dl_band);

  // Apply coupling coefficients
#pragma omp parallel default(none)				\
  shared(w,c,pure_e1,pure_e2,pure_b1,pure_b2,spin1,spin2)
  {
    int ll2,ll3;
    int pe1=pure_e1,pe2=pure_e2,pb1=pure_b1,pb2=pure_b2;
    int sign_overall=1;
    if((spin1+spin2) & 1)
      sign_overall=-1;

#pragma omp for schedule(dynamic)
    for(ll2=0;ll2<=w->lmax;ll2++) {
      for(ll3=0;ll3<=w->lmax;ll3++) {
        double fac=(2*ll3+1.)*sign_overall;
        if(w->ncls==1)
          w->coupling_matrix_unbinned[1*ll2+0][1*ll3+0]=fac*c->xi_00[0][ll2][ll3]; //TT,TT
        if(w->ncls==2) {
          w->coupling_matrix_unbinned[2*ll2+0][2*ll3+0]=fac*c->xi_0s[0][pe1+pe2][ll2][ll3]; //TE,TE
          w->coupling_matrix_unbinned[2*ll2+1][2*ll3+1]=fac*c->xi_0s[0][pb1+pb2][ll2][ll3]; //TB,TB
        }
        if(w->ncls==4) {
          w->coupling_matrix_unbinned[4*ll2+0][4*ll3+3]=fac*c->xi_mm[0][pe1+pe2][ll2][ll3]; //EE,BB
          w->coupling_matrix_unbinned[4*ll2+1][4*ll3+2]=-fac*c->xi_mm[0][pe1+pb2][ll2][ll3]; //EB,BE
          w->coupling_matrix_unbinned[4*ll2+2][4*ll3+1]=-fac*c->xi_mm[0][pb1+pe2][ll2][ll3]; //BE,EB
          w->coupling_matrix_unbinned[4*ll2+3][4*ll3+0]=fac*c->xi_mm[0][pb1+pb2][ll2][ll3]; //BB,EE
          w->coupling_matrix_unbinned[4*ll2+0][4*ll3+0]=fac*c->xi_pp[0][pe1+pe2][ll2][ll3]; //EE,EE
          w->coupling_matrix_unbinned[4*ll2+1][4*ll3+1]=fac*c->xi_pp[0][pe1+pb2][ll2][ll3]; //EB,EB
          w->coupling_matrix_unbinned[4*ll2+2][4*ll3+2]=fac*c->xi_pp[0][pb1+pe2][ll2][ll3]; //BE,BE
          w->coupling_matrix_unbinned[4*ll2+3][4*ll3+3]=fac*c->xi_pp[0][pb1+pb2][ll2][ll3]; //BB,BB
        }
        if(w->ncls==7) {
          w->coupling_matrix_unbinned[7*ll2+0][7*ll3+0]=fac*c->xi_00[0][ll2][ll3]; //TT,TT
          w->coupling_matrix_unbinned[7*ll2+1][7*ll3+1]=fac*c->xi_0s[0][pe2][ll2][ll3]; //TE,TE
          w->coupling_matrix_unbinned[7*ll2+2][7*ll3+2]=fac*c->xi_0s[0][pb2][ll2][ll3]; //TB,TB
          w->coupling_matrix_unbinned[7*ll2+3][7*ll3+6]=fac*c->xi_mm[0][pe2+pe2][ll2][ll3]; //EE,BB
          w->coupling_matrix_unbinned[7*ll2+4][7*ll3+5]=-fac*c->xi_mm[0][pe2+pb2][ll2][ll3]; //EB,BE
          w->coupling_matrix_unbinned[7*ll2+5][7*ll3+4]=-fac*c->xi_mm[0][pb2+pe2][ll2][ll3]; //BE,EB
          w->coupling_matrix_unbinned[7*ll2+6][7*ll3+3]=fac*c->xi_mm[0][pb2+pb2][ll2][ll3]; //BB,EE
          w->coupling_matrix_unbinned[7*ll2+3][7*ll3+3]=fac*c->xi_pp[0][pe2+pe2][ll2][ll3]; //EE,EE
          w->coupling_matrix_unbinned[7*ll2+4][7*ll3+4]=fac*c->xi_pp[0][pe2+pb2][ll2][ll3]; //EB,EB
          w->coupling_matrix_unbinned[7*ll2+5][7*ll3+5]=fac*c->xi_pp[0][pb2+pe2][ll2][ll3]; //BE,BE
          w->coupling_matrix_unbinned[7*ll2+6][7*ll3+6]=fac*c->xi_pp[0][pb2+pb2][ll2][ll3]; //BB,BB
        }
      }
    } //end omp for
  } //end omp parallel
 
  nmt_master_calculator_free(c);
  bin_coupling_matrix(w);

  return w;
}

void nmt_couple_cl_l(nmt_workspace *w,flouble **cl_in,flouble **cl_out)
{
  int l1;
  for(l1=0;l1<=w->lmax;l1++) {
    int icl1=0;
    for(icl1=0;icl1<w->ncls;icl1++) {
      int l2;
      flouble cl=0;
      flouble *mrow=w->coupling_matrix_unbinned[w->ncls*l1+icl1];
      for(l2=0;l2<=w->lmax;l2++) {
	int icl2=0;
	flouble beamprod=w->beam_prod[l2];
	for(icl2=0;icl2<w->ncls;icl2++)
	  cl+=mrow[w->ncls*l2+icl2]*beamprod*cl_in[icl2][l2];
      }
      cl_out[icl1][l1]=cl;
    }
  }
}

void nmt_compute_bandpower_windows(nmt_workspace *w,double *bpw_win_out)
{
  // Bin mode-coupling matrix
  gsl_matrix *mat_coupled_bin=gsl_matrix_calloc(w->ncls*w->bin->n_bands,
						w->ncls*(w->lmax+1));

  int icl1;
  for(icl1=0;icl1<w->ncls;icl1++) {
    int ib1;
    for(ib1=0;ib1<w->bin->n_bands;ib1++) {
      int i1;
      int index_b1=w->ncls*ib1+icl1;
      for(i1=0;i1<w->bin->nell_list[ib1];i1++) {
	int icl2;
	int l1=w->bin->ell_list[ib1][i1];
	int index_1=w->ncls*l1+icl1;
	double wf=w->bin->f_ell[ib1][i1]*w->bin->w_list[ib1][i1];
	double *matrix_row=w->coupling_matrix_unbinned[index_1];
	for(icl2=0;icl2<w->ncls;icl2++) {
	  int l2;
	  for(l2=0;l2<=w->lmax;l2++) {
	    int index_2=w->ncls*l2+icl2;
	    double beamprod=w->beam_prod[l2];
	    double m0=gsl_matrix_get(mat_coupled_bin,
				     index_b1,index_2);
	    gsl_matrix_set(mat_coupled_bin,index_b1,index_2,
			   m0+matrix_row[index_2]*beamprod*wf);
	  }
	}
      }
    }
  }

  gsl_matrix *inv_mcm=gsl_matrix_alloc(w->ncls*w->bin->n_bands,
				       w->ncls*w->bin->n_bands);
  gsl_matrix *bpw_win=gsl_matrix_calloc(w->ncls*w->bin->n_bands,
					w->ncls*(w->lmax+1));
  //Inverse binned MCM
  gsl_linalg_LU_invert(w->coupling_matrix_binned,
		       w->coupling_matrix_perm,
		       inv_mcm);
  //M^-1 * M
  gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1,inv_mcm,mat_coupled_bin,0,bpw_win);

  for(icl1=0;icl1<w->ncls;icl1++) {
    int ib1;
    for(ib1=0;ib1<w->bin->n_bands;ib1++) {
      int icl2;
      int index_1=w->ncls*ib1+icl1;
      for(icl2=0;icl2<w->ncls;icl2++) {
	int l2;
	for(l2=0;l2<=w->lmax;l2++) {
	  int index_2=w->ncls*l2+icl2;
	  int index=index_1*w->ncls*(w->lmax+1)+index_2;
	  bpw_win_out[index]=gsl_matrix_get(bpw_win,index_1,index_2);
	}
      }
    }
  }

  gsl_matrix_free(bpw_win);
  gsl_matrix_free(inv_mcm);
  gsl_matrix_free(mat_coupled_bin);
}

void nmt_decouple_cl_l(nmt_workspace *w,flouble **cl_in,flouble **cl_noise_in,
		       flouble **cl_bias,flouble **cl_out)
{
  int icl,ib2,l2;
  gsl_vector *dl_map_bad_b=gsl_vector_alloc(w->ncls*w->bin->n_bands);
  gsl_vector *dl_map_good_b=gsl_vector_alloc(w->ncls*w->bin->n_bands);

  //Bin coupled power spectrum
  for(icl=0;icl<w->ncls;icl++) {
    for(ib2=0;ib2<w->bin->n_bands;ib2++) {
      int i2;
      double dl_b=0;
      for(i2=0;i2<w->bin->nell_list[ib2];i2++) {
	l2=w->bin->ell_list[ib2][i2];
	dl_b+=(cl_in[icl][l2]-cl_noise_in[icl][l2]-cl_bias[icl][l2])*w->bin->f_ell[ib2][i2]*w->bin->w_list[ib2][i2];
      }
      gsl_vector_set(dl_map_bad_b,w->ncls*ib2+icl,dl_b);
    }
  }

  gsl_linalg_LU_solve(w->coupling_matrix_binned,w->coupling_matrix_perm,dl_map_bad_b,dl_map_good_b);
  for(icl=0;icl<w->ncls;icl++) {
    for(ib2=0;ib2<w->bin->n_bands;ib2++)
      cl_out[icl][ib2]=gsl_vector_get(dl_map_good_b,w->ncls*ib2+icl);
  }

  gsl_vector_free(dl_map_bad_b);
  gsl_vector_free(dl_map_good_b);
}
