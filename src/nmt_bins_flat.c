#include "config.h"
#include "utils.h"

int nmt_bins_flat_search_fast(nmt_binning_scheme_flat *bin,flouble l,int il)
{
  //If last iteration failed, restart from 0
  if(il<0) il=0;

  if(l<bin->ell_0_list[il]) { //Loop backwards
    int ilback=il-1;
    while(ilback>=0) {
      if(l<bin->ell_0_list[ilback])
	ilback--;
      else
	return ilback;
    }
    return -1;
  }
  else if(l>=bin->ell_f_list[il]) { //Loop forwards
    int ilback=il+1;
    while(ilback<bin->n_bands) {
      if(l>=bin->ell_f_list[ilback])
	ilback++;
      else
	return ilback;
    }
    return -1;
  }
  else {
    return il;
  }
}

void nmt_bins_flat_free(nmt_binning_scheme_flat *bin)
{
  free(bin->l_fl);
  free(bin->f_fl);
  nmt_k_function_free(bin->fl_f);
  free(bin->ell_0_list);
  free(bin->ell_f_list);
  free(bin);
}

nmt_binning_scheme_flat *nmt_bins_flat_constant(int nlb, flouble lmax)
{
  int ii;
  int nband_max=(int)((lmax-1)/nlb);
  flouble *l0=my_calloc(nband_max,sizeof(flouble));
  flouble *lf=my_calloc(nband_max,sizeof(flouble));
  for(ii=0;ii<nband_max;ii++) {
    l0[ii]=2+nlb*ii;
    lf[ii]=2+nlb*(ii+1);
  }

  nmt_binning_scheme_flat *bin=nmt_bins_flat_create(nband_max,l0,lf,
                                                    -1,NULL,NULL);
  free(l0);
  free(lf);
  return bin;
}

nmt_binning_scheme_flat *nmt_bins_flat_create(int nell,flouble *l0,flouble *lf,
                                              int nl_fl,flouble *l_fl,flouble *f_fl)
{
  nmt_binning_scheme_flat *bin=my_malloc(sizeof(nmt_binning_scheme_flat));
  bin->n_bands=nell;
  bin->ell_0_list=my_calloc(nell,sizeof(flouble));
  bin->ell_f_list=my_calloc(nell,sizeof(flouble));
  memcpy(bin->ell_0_list,l0,nell*sizeof(flouble));
  memcpy(bin->ell_f_list,lf,nell*sizeof(flouble));

  int is_const=0;
  if(nl_fl<=0)
    bin->nl_fl=2;
  else
    bin->nl_fl=nl_fl;
  bin->l_fl=my_malloc(bin->nl_fl*sizeof(flouble));
  bin->f_fl=my_malloc(bin->nl_fl*sizeof(flouble));
  if(nl_fl<=0) {
    bin->l_fl[0]=l0[0];
    bin->l_fl[1]=lf[nell-1];
    bin->f_fl[0]=1;
    bin->f_fl[0]=1;
    is_const=1;
  }
  else {
    memcpy(bin->l_fl,l_fl,nl_fl*sizeof(flouble));
    memcpy(bin->f_fl,f_fl,nl_fl*sizeof(flouble));
  }
  bin->fl_f=nmt_k_function_alloc(bin->nl_fl, bin->l_fl, bin->f_fl,
                                 bin->f_fl[0], bin->f_fl[bin->nl_fl-1],
                                 is_const);

  return bin;
}

void nmt_bin_cls_flat(nmt_binning_scheme_flat *bin,int nl,flouble *larr,flouble **cls_in,
		      flouble **cls_out,int ncls)
{
  int icl;
  gsl_interp_accel *intacc=gsl_interp_accel_alloc();

  for(icl=0;icl<ncls;icl++) {
    int il;
    nmt_k_function *fcl=nmt_k_function_alloc(nl,larr,cls_in[icl],cls_in[icl][0],cls_in[icl][nl-1],0);
    for(il=0;il<bin->n_bands;il++) {
      flouble ell=0.5*(bin->ell_0_list[il]+bin->ell_f_list[il]);
      flouble cell=nmt_k_function_eval(fcl,ell,intacc);
      flouble fell=nmt_k_function_eval(bin->fl_f,ell,intacc);
      cls_out[icl][il]=cell*fell;
    }
    nmt_k_function_free(fcl);
  }
  
  gsl_interp_accel_free(intacc);
}

void nmt_unbin_cls_flat(nmt_binning_scheme_flat *bin,flouble **cls_in,
			int nl,flouble *larr,flouble **cls_out,int ncls)
{
  int icl;

  for(icl=0;icl<ncls;icl++) {
    int il,ib=0;
    for(il=0;il<nl;il++) {
      flouble fl=nmt_k_function_eval(bin->fl_f,larr[il],NULL);
      ib=nmt_bins_flat_search_fast(bin,larr[il],ib);
      if(ib>=0)
	cls_out[icl][il]=cls_in[icl][ib];
      else
	cls_out[icl][il]=-999;
    }
  }
}

void nmt_ell_eff_flat(nmt_binning_scheme_flat *bin,flouble *larr)
{
  int ib;

  for(ib=0;ib<bin->n_bands;ib++)
    larr[ib]=0.5*(bin->ell_0_list[ib]+bin->ell_f_list[ib]);
}
