#include "config.h"
#include "utils.h"

static void purify_generic(nmt_field *fl,flouble *mask,fcomplex **walm0,
			   flouble **maps_in,fcomplex **alms_out,int niter)
{
  if(fl->pure_b || fl->pure_e) {
    nmt_purify(fl,mask,walm0,maps_in,maps_in,alms_out,niter);
  }
  else {
    int im1;
    for(im1=0;im1<fl->nmaps;im1++)
      he_map_product(fl->cs,maps_in[im1],mask,maps_in[im1]);
    he_map2alm(fl->cs,fl->lmax,1,2*fl->pol,maps_in,alms_out,niter);
  }
}

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

static nmt_workspace *nmt_workspace_new(nmt_curvedsky_info *cs,int ncls,
					nmt_binning_scheme *bin,int is_teb,
					int lmax_fields,int lmax_mask)
{
  int ii;
  nmt_workspace *w=my_malloc(sizeof(nmt_workspace));
  w->lmax=bin->ell_max;
  w->lmax_fields=lmax_fields;
  w->lmax_mask=lmax_mask;
  w->is_teb=is_teb;
  w->ncls=ncls;

  w->cs=nmt_curvedsky_info_copy(cs);
  w->pcl_masks=my_malloc((w->lmax_mask+1)*sizeof(flouble));
  w->beam_prod=my_malloc((w->lmax_fields+1)*sizeof(flouble));

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
  free(w->cs);
  gsl_permutation_free(w->coupling_matrix_perm);
  gsl_matrix_free(w->coupling_matrix_binned);
  nmt_bins_free(w->bin);
  for(ii=0;ii<w->ncls*(w->lmax+1);ii++)
    free(w->coupling_matrix_unbinned[ii]);
  free(w->coupling_matrix_unbinned);
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
	  for(i2=0;i2<w->bin->nell_list[ib2];i2++) {
	    l2=w->bin->ell_list[ib2][i2];
	    for(i3=0;i3<w->bin->nell_list[ib3];i3++) {
	      l3=w->bin->ell_list[ib3][i3];
	      coupling_b+=w->coupling_matrix_unbinned[w->ncls*l2+icl_a][w->ncls*l3+icl_b]*
		w->beam_prod[l3]*w->bin->w_list[ib2][i2]*w->bin->f_ell[ib2][i2]/w->bin->f_ell[ib3][i3];
	    }
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

//Computes binned coupling matrix
// fl1,fl2 (in) : fields we're correlating
// coupling_matrix_out (out) : unbinned coupling matrix
nmt_workspace *nmt_compute_coupling_matrix(nmt_field *fl1,nmt_field *fl2,
					   nmt_binning_scheme *bin,int is_teb,
					   int niter,int lmax_mask)
{
  int l2,lmax_large,lmax_fields;
  nmt_workspace *w;
  int n_cl=fl1->nmaps*fl2->nmaps;
  if(is_teb) {
    if(!((fl1->pol==0) && (fl2->pol==1)))
      report_error(NMT_ERROR_INCONSISTENT,"For T-E-B MCM the first input field must be spin-0 and the second spin-2\n");
    n_cl=7;
  }

  if(!(nmt_diff_curvedsky_info(fl1->cs,fl2->cs)))
    report_error(NMT_ERROR_CONSISTENT_RESO,
		 "Can't correlate fields with different pixelizations"
		 " or resolutions\n");
  if(bin->ell_max>he_get_lmax(fl1->cs))
    report_error(NMT_ERROR_CONSISTENT_RESO,
		 "Requesting bandpowers for too high a "
		 "multipole given map resolution\n");
  lmax_fields=fl1->lmax; // ell_max for the maps
  lmax_large=lmax_fields; // ell_max for the masks
  if(lmax_mask>lmax_large)
    lmax_large=lmax_mask;
  w=nmt_workspace_new(fl1->cs,n_cl,bin,is_teb,
		      lmax_fields,lmax_large);

  for(l2=0;l2<=w->lmax_fields;l2++)
    w->beam_prod[l2]=fl1->beam[l2]*fl2->beam[l2];

  he_anafast(&(fl1->mask),&(fl2->mask),0,0,&(w->pcl_masks),fl1->cs,w->lmax_mask,niter);
  for(l2=0;l2<=w->lmax_mask;l2++)
    w->pcl_masks[l2]*=(2*l2+1.);

#pragma omp parallel default(none)		\
  shared(w,fl1,fl2)
  {
    int ll2,ll3;
    double *wigner_00=NULL,*wigner_22=NULL,*wigner_12=NULL,*wigner_02=NULL;
    int lstart=0;
    int pe1=fl1->pure_e,pe2=fl2->pure_e,pb1=fl1->pure_b,pb2=fl2->pure_b;
    int pure_any=pe1 || pb1 || pe2 || pb2;

    if((w->ncls==1) || (w->ncls==2) || (w->ncls==7))
      wigner_00=my_malloc(2*(w->lmax_mask+1)*sizeof(double));
    if((w->ncls==2) || (w->ncls==4) || (w->ncls==7))
      wigner_22=my_malloc(2*(w->lmax_mask+1)*sizeof(double));
    if(pure_any) {
      wigner_12=my_malloc(2*(w->lmax_mask+1)*sizeof(double));
      wigner_02=my_malloc(2*(w->lmax_mask+1)*sizeof(double));
    }

    if((w->ncls!=1) && (w->ncls!=7))
      lstart=2;

#pragma omp for schedule(dynamic)
    for(ll2=lstart;ll2<=w->lmax;ll2++) {
      for(ll3=lstart;ll3<=w->lmax;ll3++) {
	int jj,l1,lmin_here,lmax_here;
	int lmin_here_00=0,lmax_here_00=2*(w->lmax_mask+1)+1;
	int lmin_here_22=0,lmax_here_22=2*(w->lmax_mask+1)+1;
	int lmin_here_12=0,lmax_here_12=2*(w->lmax_mask+1)+1;
	int lmin_here_02=0,lmax_here_02=2*(w->lmax_mask+1)+1;

	if((w->ncls==1) || (w->ncls==2) || (w->ncls==7))
	  drc3jj(ll2,ll3,0,0,&lmin_here_00,&lmax_here_00,wigner_00,2*(w->lmax_mask+1));
	if((w->ncls==2) || (w->ncls==4) || (w->ncls==7))
	  drc3jj(ll2,ll3,2,-2,&lmin_here_22,&lmax_here_22,wigner_22,2*(w->lmax_mask+1));
	if(pure_any) {
	  drc3jj(ll2,ll3,1,-2,&lmin_here_12,&lmax_here_12,wigner_12,2*(w->lmax_mask+1));
	  drc3jj(ll2,ll3,0,-2,&lmin_here_02,&lmax_here_02,wigner_02,2*(w->lmax_mask+1));
	}

	if(w->ncls!=7) {
	  lmin_here=NMT_MAX(lmin_here_00,lmin_here_22);
	  lmax_here=NMT_MIN(lmax_here_00,lmax_here_22);
	}
	else {
	  lmin_here=NMT_MIN(lmin_here_00,lmin_here_22);
	  lmax_here=NMT_MAX(lmax_here_00,lmax_here_22);
	}
	if(pure_any) {
	  lmin_here=NMT_MIN(lmin_here,lmin_here_12);
	  lmin_here=NMT_MIN(lmin_here,lmin_here_02);
	  lmax_here=NMT_MAX(lmax_here,lmax_here_12);
	  lmax_here=NMT_MAX(lmax_here,lmax_here_02);
	}
	//All lines regarding lmax are in principle unnecessary, since lmax is just l3+l2

	for(l1=lmin_here;l1<=lmax_here;l1++) {
	  if(l1<=w->lmax_mask) {
	    flouble wfac,fac_12=0,fac_02=0;
	    int j02,j12;
	    int j00=l1-lmin_here_00;
	    int j22=l1-lmin_here_22;
	    if(pure_any) {
	      j12=l1-lmin_here_12;
	      j02=l1-lmin_here_02;
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
	    }

	    if(w->ncls==1) {
	      wfac=w->pcl_masks[l1]*wigner_00[j00]*wigner_00[j00];
	      w->coupling_matrix_unbinned[1*ll2+0][1*ll3+0]+=wfac; //TT,TT
	    }
	    if(w->ncls==2) {
	      double wfac_ispure[2];
	      if(pure_any) {
		wfac_ispure[0]=wigner_22[j22];
		wfac_ispure[1]=wigner_22[j22]+fac_12*wigner_12[j12]+fac_02*wigner_02[j02];
		wfac_ispure[0]*=w->pcl_masks[l1]*wigner_00[j00];
		wfac_ispure[1]*=w->pcl_masks[l1]*wigner_00[j00];
	      }
	      else {
		wfac_ispure[0]=wigner_22[j22]*w->pcl_masks[l1]*wigner_00[j00];
		wfac_ispure[1]=wfac_ispure[0];
	      }
	      w->coupling_matrix_unbinned[2*ll2+0][2*ll3+0]+=wfac_ispure[pe1+pe2]; //TE,TE
	      w->coupling_matrix_unbinned[2*ll2+1][2*ll3+1]+=wfac_ispure[pb1+pb2]; //TB,TB
	    }
	    if(w->ncls==4) {
	      double wfac_ispure[3];
	      int suml=l1+ll2+ll3;
	      if(pure_any) {
		wfac_ispure[0]=wigner_22[j22];
		wfac_ispure[1]=wigner_22[j22]+fac_12*wigner_12[j12]+fac_02*wigner_02[j02];
		wfac_ispure[2]=wfac_ispure[1]*wfac_ispure[1]*w->pcl_masks[l1];
		wfac_ispure[1]*=wigner_22[j22]*w->pcl_masks[l1];
		wfac_ispure[0]*=wigner_22[j22]*w->pcl_masks[l1];
	      }
	      else {
		wfac_ispure[0]=wigner_22[j22]*wigner_22[j22]*w->pcl_masks[l1];
		wfac_ispure[1]=wfac_ispure[0];
		wfac_ispure[2]=wfac_ispure[0];
	      }

	      if(suml & 1) { //Odd sum
		w->coupling_matrix_unbinned[4*ll2+0][4*ll3+3]+=wfac_ispure[pe1+pe2]; //EE,BB
		w->coupling_matrix_unbinned[4*ll2+1][4*ll3+2]-=wfac_ispure[pe1+pb2]; //EB,BE
		w->coupling_matrix_unbinned[4*ll2+2][4*ll3+1]-=wfac_ispure[pb1+pe2]; //BE,EB
		w->coupling_matrix_unbinned[4*ll2+3][4*ll3+0]+=wfac_ispure[pb1+pb2]; //BB,EE
	      }
	      else {
		w->coupling_matrix_unbinned[4*ll2+0][4*ll3+0]+=wfac_ispure[pe1+pe2]; //EE,EE
		w->coupling_matrix_unbinned[4*ll2+1][4*ll3+1]+=wfac_ispure[pe1+pb2]; //EB,EB
		w->coupling_matrix_unbinned[4*ll2+2][4*ll3+2]+=wfac_ispure[pb1+pe2]; //BE,BE
		w->coupling_matrix_unbinned[4*ll2+3][4*ll3+3]+=wfac_ispure[pb1+pb2]; //BB,BB
	      }
	    }
	    if(w->ncls==7) {
	      double wfac_ispure_02[2];
	      double wfac_ispure_22[3];
	      int suml=l1+ll2+ll3;
	      wfac=w->pcl_masks[l1]*wigner_00[j00]*wigner_00[j00];
	      if(pure_any) {
		wfac_ispure_02[0]=wigner_22[j22];
		wfac_ispure_02[1]=wigner_22[j22]+fac_12*wigner_12[j12]+fac_02*wigner_02[j02];
		wfac_ispure_02[0]*=w->pcl_masks[l1]*wigner_00[j00];
		wfac_ispure_02[1]*=w->pcl_masks[l1]*wigner_00[j00];
		wfac_ispure_22[0]=wigner_22[j22];
		wfac_ispure_22[1]=wigner_22[j22]+fac_12*wigner_12[j12]+fac_02*wigner_02[j02];
		wfac_ispure_22[2]=wfac_ispure_22[1]*wfac_ispure_22[1]*w->pcl_masks[l1];
		wfac_ispure_22[1]*=wigner_22[j22]*w->pcl_masks[l1];
		wfac_ispure_22[0]*=wigner_22[j22]*w->pcl_masks[l1];
	      }
	      else {
		wfac_ispure_02[0]=wigner_22[j22]*w->pcl_masks[l1]*wigner_00[j00];
		wfac_ispure_02[1]=wfac_ispure_02[0];
		wfac_ispure_22[0]=wigner_22[j22]*wigner_22[j22]*w->pcl_masks[l1];
		wfac_ispure_22[1]=wfac_ispure_22[0];
		wfac_ispure_22[2]=wfac_ispure_22[0];
	      }
	      w->coupling_matrix_unbinned[7*ll2+0][7*ll3+0]+=wfac; //TT,TT
	      w->coupling_matrix_unbinned[7*ll2+1][7*ll3+1]+=wfac_ispure_02[pe2]; //TE,TE
	      w->coupling_matrix_unbinned[7*ll2+2][7*ll3+2]+=wfac_ispure_02[pb2]; //TB,TB
	      if(suml & 1) { //Odd sum
		w->coupling_matrix_unbinned[7*ll2+3][7*ll3+6]+=wfac_ispure_22[pe2+pe2]; //EE,BB
		w->coupling_matrix_unbinned[7*ll2+4][7*ll3+5]-=wfac_ispure_22[pe2+pb2]; //EB,BE
		w->coupling_matrix_unbinned[7*ll2+5][7*ll3+4]-=wfac_ispure_22[pb2+pe2]; //BE,EB
		w->coupling_matrix_unbinned[7*ll2+6][7*ll3+3]+=wfac_ispure_22[pb2+pb2]; //BB,EE
	      }
	      else {
		w->coupling_matrix_unbinned[7*ll2+3][7*ll3+3]+=wfac_ispure_22[pe2+pe2]; //EE,EE
		w->coupling_matrix_unbinned[7*ll2+4][7*ll3+4]+=wfac_ispure_22[pe2+pb2]; //EB,EB
		w->coupling_matrix_unbinned[7*ll2+5][7*ll3+5]+=wfac_ispure_22[pb2+pe2]; //BE,BE
		w->coupling_matrix_unbinned[7*ll2+6][7*ll3+6]+=wfac_ispure_22[pb2+pb2]; //BB,BB
	      }
	    }
	  }
	}
	for(jj=0;jj<w->ncls;jj++) {
	  int kk;
	  for(kk=0;kk<w->ncls;kk++)
	    w->coupling_matrix_unbinned[w->ncls*ll2+jj][w->ncls*ll3+kk]*=(2*ll3+1.)/(4*M_PI);
	}
      }
    } //end omp for
    if((w->ncls==1) || (w->ncls==2) || (w->ncls==7))
      free(wigner_00);
    if((w->ncls==2) || (w->ncls==4) || (w->ncls==7))
      free(wigner_22);
    if(pure_any) {
      free(wigner_12);
      free(wigner_02);
    }
  } //end omp parallel

  bin_coupling_matrix(w);

  return w;
}

void nmt_compute_uncorr_noise_deprojection_bias(nmt_field *fl1,flouble *map_var,flouble **cl_bias,
						int  niter)
{
  int ii;
  long ip;
  int nspec=fl1->nmaps*fl1->nmaps;
  int lmax=fl1->lmax;

  for(ii=0;ii<nspec;ii++) {
    for(ip=0;ip<=lmax;ip++)
      cl_bias[ii][ip]=0;
  }

  if(fl1->ntemp>0) {
    //Allocate dummy maps and alms
    flouble **map_dum=my_malloc(fl1->nmaps*sizeof(flouble *));
    fcomplex **alm_dum=my_malloc(fl1->nmaps*sizeof(fcomplex *));
    for(ii=0;ii<fl1->nmaps;ii++) {
      map_dum[ii]=my_malloc(fl1->npix*sizeof(flouble));
      alm_dum[ii]=my_malloc(he_nalms(fl1->lmax)*sizeof(fcomplex));
    }

    flouble **cl_dum;
    cl_dum=my_malloc(nspec*sizeof(flouble *));
    for(ii=0;ii<nspec;ii++)
      cl_dum[ii]=my_calloc((lmax+1),sizeof(flouble));

    int iti,itj,itp,itq,im1;
    flouble *mat_prod=my_calloc(fl1->ntemp*fl1->ntemp,sizeof(flouble));
    for(iti=0;iti<fl1->ntemp;iti++) {
      for(itj=0;itj<fl1->ntemp;itj++) {
	double nij=gsl_matrix_get(fl1->matrix_M,iti,itj);
	for(im1=0;im1<fl1->nmaps;im1++) {
	  he_map_product(fl1->cs,fl1->temp[itj][im1],map_var,map_dum[im1]); //sigma^2*f^j
	  he_map_product(fl1->cs,map_dum[im1],fl1->mask,map_dum[im1]); //v*sigma^2*f^j
	  he_map_product(fl1->cs,map_dum[im1],fl1->mask,map_dum[im1]); //v^2*sigma^2*f^j
	}

	//Int[v^2*sigma^2*f^j*f^r]
	for(im1=0;im1<fl1->nmaps;im1++)
	  mat_prod[iti*fl1->ntemp+itj]+=he_map_dot(fl1->cs,map_dum[im1],fl1->temp[iti][im1]);

	//SHT[v^2*sigma^2*f^j]
	he_map2alm(fl1->cs,fl1->lmax,1,2*fl1->pol,map_dum,alm_dum,niter);
	//Sum_m(SHT[v^2*sigma^2*f^j]*f^i)/(2l+1)
	he_alm2cl(alm_dum,fl1->a_temp[iti],fl1->pol,fl1->pol,cl_dum,lmax);
	for(im1=0;im1<nspec;im1++) {
	  for(ip=0;ip<=lmax;ip++)
	    cl_bias[im1][ip]-=2*cl_dum[im1][ip]*nij;
	}
      }
    }

    for(iti=0;iti<fl1->ntemp;iti++) {
      for(itp=0;itp<fl1->ntemp;itp++) {
	//Sum_m(f^i*f^p*)/(2l+1)
	he_alm2cl(fl1->a_temp[iti],fl1->a_temp[itp],fl1->pol,fl1->pol,cl_dum,lmax);
	for(itj=0;itj<fl1->ntemp;itj++) {
	  double mij=gsl_matrix_get(fl1->matrix_M,iti,itj);
	  for(itq=0;itq<fl1->ntemp;itq++) {
	    double npq=gsl_matrix_get(fl1->matrix_M,itp,itq);
	    for(im1=0;im1<nspec;im1++) {
	      for(ip=0;ip<=lmax;ip++)
		cl_bias[im1][ip]+=cl_dum[im1][ip]*mat_prod[itj*fl1->ntemp+itq]*mij*npq;
	    }
	  }
	}
      }
    }

    free(mat_prod);

    for(ii=0;ii<fl1->nmaps;ii++) {
      free(map_dum[ii]);
      free(alm_dum[ii]);
    }
    free(map_dum);
    free(alm_dum);
    for(ii=0;ii<nspec;ii++)
      free(cl_dum[ii]);
    free(cl_dum);
  }
}

void nmt_compute_deprojection_bias(nmt_field *fl1,nmt_field *fl2,
				   flouble **cl_proposal,flouble **cl_bias,int niter)
{
  int ii;
  flouble **cl_dum;
  long ip;
  int nspec=fl1->nmaps*fl2->nmaps;
  int lmax=fl1->lmax;

  if(!(nmt_diff_curvedsky_info(fl1->cs,fl2->cs)))
    report_error(NMT_ERROR_CONSISTENT_RESO,"Can't correlate fields with different pixelizations\n");

  cl_dum=my_malloc(nspec*sizeof(flouble *));
  for(ii=0;ii<nspec;ii++) {
    cl_dum[ii]=my_calloc((lmax+1),sizeof(flouble));
    for(ip=0;ip<=lmax;ip++)
      cl_bias[ii][ip]=0;
  }

  //TODO: some terms (e.g. C^ab*SHT[w*g^j]) could be precomputed
  //TODO: if fl1=fl2 F2=F3
  //Allocate dummy maps and alms
  flouble **map_1_dum=my_malloc(fl1->nmaps*sizeof(flouble *));
  fcomplex **alm_1_dum=my_malloc(fl1->nmaps*sizeof(fcomplex *));
  for(ii=0;ii<fl1->nmaps;ii++) {
    map_1_dum[ii]=my_malloc(fl1->npix*sizeof(flouble));
    alm_1_dum[ii]=my_malloc(he_nalms(fl1->lmax)*sizeof(fcomplex));
  }
  flouble **map_2_dum=my_malloc(fl2->nmaps*sizeof(flouble *));
  fcomplex **alm_2_dum=my_malloc(fl2->nmaps*sizeof(fcomplex *));
  for(ii=0;ii<fl2->nmaps;ii++) {
    map_2_dum[ii]=my_malloc(fl1->npix*sizeof(flouble));
    alm_2_dum[ii]=my_malloc(he_nalms(fl1->lmax)*sizeof(fcomplex));
  }

  if(fl2->ntemp>0) {
    int iti;
    for(iti=0;iti<fl2->ntemp;iti++) {
      int itj;
      for(itj=0;itj<fl2->ntemp;itj++) {
	int im1,im2;
	double nij=gsl_matrix_get(fl2->matrix_M,iti,itj);
	//w*g^j
	for(im2=0;im2<fl2->nmaps;im2++)
	  he_map_product(fl2->cs,fl2->temp[itj][im2],fl2->mask,map_2_dum[im2]);
	//SHT[w*g^j]
	he_map2alm(fl2->cs,fl2->lmax,1,2*fl2->pol,map_2_dum,alm_2_dum,niter);
	//C^ab*SHT[w*g^j]
	for(im1=0;im1<fl1->nmaps;im1++) {
	  he_zero_alm(fl1->lmax,alm_1_dum[im1]);
	  for(im2=0;im2<fl2->nmaps;im2++)
	    he_alter_alm(lmax,-1.,alm_2_dum[im2],alm_1_dum[im1],cl_proposal[im1*fl2->nmaps+im2],1);
	}
	//SHT^-1[C^ab*SHT[w*g^j]]
	he_alm2map(fl1->cs,fl1->lmax,1,2*fl1->pol,map_1_dum,alm_1_dum);
	//SHT[v*SHT^-1[C^ab*SHT[w*g^j]]]
	purify_generic(fl1,fl1->mask,fl1->a_mask,map_1_dum,alm_1_dum,niter);
	//Sum_m(SHT[v*SHT^-1[C^ab*SHT[w*g^j]]]*g^i*)/(2l+1)
	he_alm2cl(alm_1_dum,fl2->a_temp[iti],fl1->pol,fl2->pol,cl_dum,lmax);
	for(im1=0;im1<nspec;im1++) {
	  for(ip=0;ip<=lmax;ip++)
	    cl_bias[im1][ip]-=cl_dum[im1][ip]*nij;
	}
      }
    }
  }

  if(fl1->ntemp>0) {
    int iti;
    for(iti=0;iti<fl1->ntemp;iti++) {
      int itj;
      for(itj=0;itj<fl1->ntemp;itj++) {
	int im1,im2;
	double mij=gsl_matrix_get(fl1->matrix_M,iti,itj);
	//v*f^j
	for(im1=0;im1<fl1->nmaps;im1++)
	  he_map_product(fl1->cs,fl1->temp[itj][im1],fl1->mask,map_1_dum[im1]);
	//SHT[v*f^j]
	he_map2alm(fl1->cs,fl1->lmax,1,2*fl1->pol,map_1_dum,alm_1_dum,niter);
	//C^abT*SHT[v*f^j]
	for(im2=0;im2<fl2->nmaps;im2++) {
	  he_zero_alm(fl2->lmax,alm_2_dum[im2]);
	  for(im1=0;im1<fl1->nmaps;im1++)
	    he_alter_alm(lmax,-1.,alm_1_dum[im1],alm_2_dum[im2],cl_proposal[im1*fl2->nmaps+im2],1);
	}
	//SHT^-1[C^abT*SHT[v*f^j]]
	he_alm2map(fl2->cs,fl2->lmax,1,2*fl2->pol,map_2_dum,alm_2_dum);
	//SHT[w*SHT^-1[C^abT*SHT[v*f^j]]]
	purify_generic(fl2,fl2->mask,fl2->a_mask,map_2_dum,alm_2_dum,niter);
	//Sum_m(f^i*SHT[w*SHT^-1[C^abT*SHT[v*f^j]]]^*)/(2l+1)
	he_alm2cl(fl1->a_temp[iti],alm_2_dum,fl1->pol,fl2->pol,cl_dum,lmax);
	for(im1=0;im1<nspec;im1++) {
	  for(ip=0;ip<=lmax;ip++)
	    cl_bias[im1][ip]-=cl_dum[im1][ip]*mij;
	}
      }
    }
  }

  if((fl1->ntemp>0) && (fl2->ntemp>0)) {
    int iti,itj,itp,itq,im1,im2;
    flouble *mat_prod=my_calloc(fl1->ntemp*fl2->ntemp,sizeof(flouble));
    for(itj=0;itj<fl1->ntemp;itj++) {
      for(itq=0;itq<fl2->ntemp;itq++) {
	//w*g^q
	for(im2=0;im2<fl2->nmaps;im2++)
	  he_map_product(fl2->cs,fl2->temp[itq][im2],fl2->mask,map_2_dum[im2]);
	//SHT[w*g^q]
	he_map2alm(fl2->cs,fl2->lmax,1,2*fl2->pol,map_2_dum,alm_2_dum,niter);
	//C^ab*SHT[w*g^q]
	for(im1=0;im1<fl1->nmaps;im1++) {
	  he_zero_alm(fl1->lmax,alm_1_dum[im1]);
	  for(im2=0;im2<fl2->nmaps;im2++)
	    he_alter_alm(lmax,-1.,alm_2_dum[im2],alm_1_dum[im1],cl_proposal[im1*fl2->nmaps+im2],1);
	}
	//SHT^-1[C^ab*SHT[w*g^q]]
	he_alm2map(fl1->cs,fl1->lmax,1,2*fl1->pol,map_1_dum,alm_1_dum);
	for(im1=0;im1<fl1->nmaps;im1++) {
	  //v*SHT^-1[C^ab*SHT[w*g^q]]
	  he_map_product(fl1->cs,map_1_dum[im1],fl1->mask,map_1_dum[im1]);
	  //Int[f^jT*v*SHT^-1[C^ab*SHT[w*g^q]]]
	  mat_prod[itj*fl2->ntemp+itq]+=he_map_dot(fl1->cs,map_1_dum[im1],fl1->temp[itj][im1]);
	}
      }
    }

    for(iti=0;iti<fl1->ntemp;iti++) {
      for(itp=0;itp<fl2->ntemp;itp++) {
	//Sum_m(f^i*g^p*)/(2l+1)
	he_alm2cl(fl1->a_temp[iti],fl2->a_temp[itp],fl1->pol,fl2->pol,cl_dum,lmax);
	for(itj=0;itj<fl1->ntemp;itj++) {
	  double mij=gsl_matrix_get(fl1->matrix_M,iti,itj);
	  for(itq=0;itq<fl2->ntemp;itq++) {
	    double npq=gsl_matrix_get(fl2->matrix_M,itp,itq);
	    for(im1=0;im1<nspec;im1++) {
	      for(ip=0;ip<=lmax;ip++)
		cl_bias[im1][ip]+=cl_dum[im1][ip]*mat_prod[itj*fl2->ntemp+itq]*mij*npq;
	    }
	  }
	}
      }
    }

    free(mat_prod);
  }

  for(ii=0;ii<fl1->nmaps;ii++) {
    free(map_1_dum[ii]);
    free(alm_1_dum[ii]);
  }
  free(map_1_dum);
  free(alm_1_dum);
  for(ii=0;ii<fl2->nmaps;ii++) {
    free(map_2_dum[ii]);
    free(alm_2_dum[ii]);
  }
  free(map_2_dum);
  free(alm_2_dum);
  for(ii=0;ii<nspec;ii++)
    free(cl_dum[ii]);
  free(cl_dum);
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
  double *bpws=my_malloc(w->ncls*w->bin->n_bands*w->ncls*(w->lmax+1));

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

void nmt_compute_coupled_cell(nmt_field *fl1,nmt_field *fl2,flouble **cl_out)
{
  if(fl1->lmax!=fl2->lmax)
    report_error(NMT_ERROR_CONSISTENT_RESO,"Can't correlate fields with different resolutions\n");

  he_alm2cl(fl1->alms,fl2->alms,fl1->pol,fl2->pol,cl_out,fl1->lmax);
}

nmt_workspace *nmt_compute_power_spectra(nmt_field *fl1,nmt_field *fl2,
					 nmt_binning_scheme *bin,nmt_workspace *w0,
					 flouble **cl_noise,flouble **cl_proposal,flouble **cl_out,
					 int niter,int lmax_mask)
{
  int ii;
  flouble **cl_bias,**cl_data;
  nmt_workspace *w;

  if(w0==NULL)
    w=nmt_compute_coupling_matrix(fl1,fl2,bin,0,niter,lmax_mask);
  else {
    w=w0;
    if(w->lmax>fl1->lmax)
      report_error(NMT_ERROR_CONSISTENT_RESO,"Workspace does not match map resolution\n");
  }

  cl_bias=my_malloc(w->ncls*sizeof(flouble *));
  cl_data=my_malloc(w->ncls*sizeof(flouble *));
  for(ii=0;ii<w->ncls;ii++) {
    cl_bias[ii]=my_calloc((fl1->lmax+1),sizeof(flouble));
    cl_data[ii]=my_calloc((fl1->lmax+1),sizeof(flouble));
  }
  nmt_compute_coupled_cell(fl1,fl2,cl_data);
  nmt_compute_deprojection_bias(fl1,fl2,cl_proposal,cl_bias,niter);
  nmt_decouple_cl_l(w,cl_data,cl_noise,cl_bias,cl_out);
  for(ii=0;ii<w->ncls;ii++) {
    free(cl_bias[ii]);
    free(cl_data[ii]);
  }
  free(cl_bias);
  free(cl_data);

  return w;
}
