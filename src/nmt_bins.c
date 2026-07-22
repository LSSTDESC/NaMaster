#include "config.h"
#include "utils.h"

void nmt_bins_free(nmt_binning_scheme *bins)
{
  int ii;
  if(bins!=NULL) {
    free(bins->nell_list);
    for(ii=0;ii<bins->n_bands;ii++) {
      free(bins->ell_list[ii]);
      free(bins->w_list[ii]);
      free(bins->f_ell[ii]);
    }
    free(bins->ell_list);
    free(bins->w_list);
    free(bins->f_ell);
    free(bins);
  }
}

nmt_binning_scheme *nmt_bins_constant(int nlb,int lmax,int is_l2)
{
  int ii;
  int nband_max=(lmax-1)/nlb;
  flouble w0=1./nlb;

  nmt_binning_scheme *bins=my_malloc(sizeof(nmt_binning_scheme));
  bins->ell_max=lmax;
  bins->n_bands=nband_max;
  bins->nell_list=my_calloc(nband_max,sizeof(int));
  bins->ell_list=my_malloc(nband_max*sizeof(int *));
  bins->w_list=my_malloc(nband_max*sizeof(flouble *));
  bins->f_ell=my_malloc(nband_max*sizeof(flouble *));

  for(ii=0;ii<nband_max;ii++) {
    int jj;
    bins->nell_list[ii]=nlb;
    bins->ell_list[ii]=my_malloc(nlb*sizeof(int));
    bins->w_list[ii]=my_malloc(nlb*sizeof(flouble));
    bins->f_ell[ii]=my_malloc(nlb*sizeof(flouble));
    for(jj=0;jj<nlb;jj++) {
      int ell=2+ii*nlb+jj;
      bins->ell_list[ii][jj]=ell;
      bins->w_list[ii][jj]=w0;
      if(is_l2)
	bins->f_ell[ii][jj]=ell*(ell+1.)/(2*M_PI);
      else
	bins->f_ell[ii][jj]=1;
    }
  }

  return bins;
}

nmt_binning_scheme *nmt_bins_create(int nell,int *bpws,int *ells,flouble *weights,
				    flouble *f_ell,int lmax)
{
  nmt_binning_scheme *bins;
  int ii,nband_max=0;

  for(ii=0;ii<nell;ii++) {
    if(ells[ii]<=lmax) {
      if(bpws[ii]>nband_max)
	nband_max=bpws[ii];
    }
  }
  nband_max++;

  bins=my_malloc(sizeof(nmt_binning_scheme));
  bins->ell_max=lmax;
  bins->n_bands=nband_max;
  bins->nell_list=my_calloc(nband_max,sizeof(int));
  bins->ell_list=my_malloc(nband_max*sizeof(int *));
  bins->w_list=my_malloc(nband_max*sizeof(flouble *));
  bins->f_ell=my_malloc(nband_max*sizeof(flouble *));

  for(ii=0;ii<nell;ii++) {
    if(ells[ii]<=lmax) {
      if(bpws[ii]>=0)
	bins->nell_list[bpws[ii]]++;
    }
  }

  for(ii=0;ii<nband_max;ii++) {
    bins->ell_list[ii]=my_malloc(bins->nell_list[ii]*sizeof(int));
    bins->w_list[ii]=my_malloc(bins->nell_list[ii]*sizeof(flouble));
    bins->f_ell[ii]=my_malloc(bins->nell_list[ii]*sizeof(flouble));
  }

  for(ii=0;ii<nband_max;ii++)
    bins->nell_list[ii]=0;

  for(ii=0;ii<nell;ii++) {
    flouble f;
    int l=ells[ii];
    int b=bpws[ii];
    flouble w=weights[ii];
    if(f_ell==NULL)
      f=1;
    else
      f=f_ell[ii];

    if(l<=lmax) {
      if(b>=0) {
	bins->ell_list[b][bins->nell_list[b]]=l;
	bins->w_list[b][bins->nell_list[b]]=w;
	if(f<=0) //Prevent division by zero later on
	  bins->f_ell[b][bins->nell_list[b]]=1;
	else
	  bins->f_ell[b][bins->nell_list[b]]=f;
	bins->nell_list[b]++;
      }
    }
  }

  for(ii=0;ii<nband_max;ii++) {
    int jj;
    flouble norm=0;
    for(jj=0;jj<bins->nell_list[ii];jj++)
      norm+=bins->w_list[ii][jj];
    if(norm<=0)
      report_error(NMT_ERROR_BWEIGHT,"Weights in band %d are wrong\n",ii);
    for(jj=0;jj<bins->nell_list[ii];jj++)
      bins->w_list[ii][jj]/=norm;
  }

  return bins;
}

void nmt_bin_mcm_oneside(nmt_binning_scheme *bin,
			 int ncls,
			 flouble *mcm_in,
			 flouble *mcm_out,
			 flouble *beam1,
			 flouble *beam2)
{
  memset(mcm_out,0,
	 ncls*ncls*bin->n_bands*(bin->ell_max+1)*sizeof(flouble));

#pragma omp parallel default(none)		\
  shared(bin,ncls,mcm_in,mcm_out,beam1,beam2)
  {
    int l2;
    int nls=bin->ell_max+1;

#pragma omp for schedule(dynamic)
    for(l2=0;l2<=bin->ell_max;l2++) {
      int icl2;
      flouble beams=beam1[l2]*beam2[l2];
      for(icl2=0;icl2<ncls;icl2++) {
	int ib1;
	for(ib1=0;ib1<bin->n_bands;ib1++) {
	  int i1;
	  for(i1=0;i1<bin->nell_list[ib1];i1++) {
	    int icl1;
	    int l1=bin->ell_list[ib1][i1];
	    for(icl1=0;icl1<ncls;icl1++) {
	      int index_out=((ncls*ib1+icl1)*nls+l2)*ncls+icl2;
	      int index_in=((ncls*l1+icl1)*nls+l2)*ncls+icl2;
	      flouble mcmin=mcm_in[index_in];
	      flouble wf=bin->f_ell[ib1][i1]*bin->w_list[ib1][i1];
	      mcm_out[index_out]+=mcmin*beams*wf;
	    }
	  }
	}
      }
    }
  } // end omp for
} // end omp parallel

void nmt_bin_mcm(nmt_binning_scheme *bin,
		 int ncls,
		 flouble *mcm_in,
		 flouble *mcm_out,
		 int norm_type,
		 flouble w2,
		 flouble *beam1,
		 flouble *beam2)
{
  memset(mcm_out,0,
	 ncls*ncls*bin->n_bands*bin->n_bands*sizeof(flouble));
#pragma omp parallel default(none)		\
  shared(bin,ncls,mcm_in,mcm_out)		\
  shared(norm_type,w2,beam1,beam2)
  {
    int icl_a,icl_b,ib2,ib3,l2,l3,i2,i3,sig;
    int nls=bin->ell_max+1;

#pragma omp for schedule(dynamic)
    for(ib2=0;ib2<bin->n_bands;ib2++) {
      for(ib3=0;ib3<bin->n_bands;ib3++) {
	for(icl_a=0;icl_a<ncls;icl_a++) {
	  for(icl_b=0;icl_b<ncls;icl_b++) {
	    double coupling_b=0;
	    if(norm_type==0) { //Usual normalisation
	      for(i2=0;i2<bin->nell_list[ib2];i2++) {
		l2=bin->ell_list[ib2][i2];
		for(i3=0;i3<bin->nell_list[ib3];i3++) {
		  l3=bin->ell_list[ib3][i3];
		  coupling_b+=mcm_in[((l2*ncls+icl_a)*nls+l3)*ncls+icl_b]*
		    beam1[l3]*beam2[l3]*bin->w_list[ib2][i2]*bin->f_ell[ib2][i2]/bin->f_ell[ib3][i3];
		}
	      }
	    }
	    else { //FKP normalisation
	      if((ncls*ib2+icl_a) == (ncls*ib3+icl_b))
		coupling_b=w2;
	    }
	    mcm_out[((ib2*ncls+icl_a)*bin->n_bands+ib3)*ncls+icl_b]=coupling_b;
	  }
	}
      }
    }
  } // end omp parallel
} // end omp for

void nmt_bin_cls(nmt_binning_scheme *bin,int ncls,flouble *cls_in,flouble *cls_out)
{
  memset(cls_out,0,ncls*bin->n_bands*sizeof(flouble));

#pragma omp parallel default(none) \
  shared(bin,ncls,cls_in,cls_out)
  {
    int ib;

#pragma omp for schedule(dynamic)
    for(ib=0;ib<bin->n_bands;ib++) {
      int icl;
      for(icl=0;icl<ncls;icl++) {
	int il;
	int iband=ib*ncls+icl;
	cls_out[iband]=0;
	for(il=0;il<bin->nell_list[ib];il++) {
	  int l=bin->ell_list[ib][il];
	  flouble w=bin->w_list[ib][il];
	  flouble f=bin->f_ell[ib][il];
	  cls_out[iband]+=w*f*cls_in[l*ncls+icl];
	}
      }
    }
  } // end omp for
} // end omp parallel

void nmt_unbin_cls(nmt_binning_scheme *bin,int ncls,flouble *cls_in,flouble *cls_out)
{
  memset(cls_out,0,ncls*(bin->ell_max+1)*sizeof(flouble));

#pragma omp parallel default(none)		\
  shared(bin,ncls,cls_in,cls_out)
  {
    int icl;
    int nls=bin->ell_max+1;

#pragma omp for schedule(dynamic)
    for(icl=0;icl<ncls;icl++) {
      int ib;
      for(ib=0;ib<bin->n_bands;ib++) {
	int il;
	flouble clb=cls_in[ib*ncls+icl];
	for(il=0;il<bin->nell_list[ib];il++) {
	  int l=bin->ell_list[ib][il];
	  cls_out[l*ncls+icl]=clb/bin->f_ell[ib][il];
	}
      }
    }
  } // end omp for
} // end omp parallel

void nmt_ell_eff(nmt_binning_scheme *bin,flouble *larr)
{
#pragma omp parallel default(none) \
  shared(bin,larr)
  {
    int ib;

#pragma omp for schedule(dynamic)
    for(ib=0;ib<bin->n_bands;ib++) {
      int il;
      larr[ib]=0;
      for(il=0;il<bin->nell_list[ib];il++)
	larr[ib]+=bin->ell_list[ib][il]*bin->w_list[ib][il];
    }
  } // end omp for
} // end omp parallel
