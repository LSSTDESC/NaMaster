#include "config.h"
#include "utils.h"

static void apodize_mask_CX(long nside,flouble *mask_in,flouble *mask_out,flouble aposize,char *apotype)
{
  long npix=he_nside2npix(nside);
  double aporad=aposize*M_PI/180;
  double x2_thr=1-cos(aporad);
  double inv_x2_thr=1./x2_thr;
  flouble *vec=my_malloc(3*npix*sizeof(flouble));
  flouble *cthv=my_malloc(npix*sizeof(flouble));
  flouble *phiv=my_malloc(npix*sizeof(flouble));
  int apotyp=0;
  if(!strcmp(apotype,"C1"))
    apotyp=0;
  else if(!strcmp(apotype,"C2"))
    apotyp=1;
  else
    report_error(NMT_ERROR_APO,"Unknown apodization type %s\n",apotype);

  if(mask_out!=mask_in)
    memcpy(mask_out,mask_in,npix*sizeof(flouble));

  //Get coords for each pixel
#pragma omp parallel default(none)		\
  shared(vec,npix,nside,cthv,phiv)
  {
    long ip;
#pragma omp for
    for(ip=0;ip<npix;ip++) {
      flouble *v=vec+3*ip;
      he_pix2vec_ring(nside,ip,v);
      cthv[ip]=v[2];
      phiv[ip]=atan2(v[1],v[0]);
      if(phiv[ip]<0)
	phiv[ip]+=2*M_PI;
    } //end omp for
  }//end omp parallel

  int lenlist0=(int)(4*npix*(1-cos(1.2*aporad)));
  if(lenlist0 < 2)
    report_error(NMT_ERROR_APO,"Your apodization scale is too small for this pixel size\n");
  
#pragma omp parallel default(none)			\
  shared(vec,npix,x2_thr,inv_x2_thr,mask_in,mask_out)	\
  shared(nside,cthv,phiv,aporad,apotyp,lenlist0)
  {
    long ip;
    int *listpix=my_malloc(lenlist0*sizeof(int));

#pragma omp for schedule(dynamic)
    for(ip=0;ip<npix;ip++) {
      if(mask_in[ip]>0) {
	int j;
	int lenlist_half=lenlist0/2;
	flouble *v0=vec+3*ip;
	flouble x2dist=1000;
	he_query_disc(nside,cthv[ip],phiv[ip],1.2*aporad,listpix,&lenlist_half,1);
	for(j=0;j<lenlist_half;j++) {
	  int ip2=listpix[j];
	  if(mask_in[ip2]<=0) {
	    flouble *v1=vec+3*ip2;
	    flouble x2=1-v0[0]*v1[0]-v0[1]*v1[1]-v0[2]*v1[2];
	    if(x2<x2dist) x2dist=x2;
	  }
	}
	if(x2dist<x2_thr) {
	  flouble f,xn;
	  if(x2dist<=0)
	    f=0;
	  else {
	    xn=sqrt(x2dist*inv_x2_thr);
	    if(apotyp==0)
	      f=xn-sin(xn*2*M_PI)/(2*M_PI);
	    else
	      f=0.5*(1-cos(xn*M_PI));
	  }
	  mask_out[ip]*=f;
	}
      }
    } //end omp for
    free(listpix);
  }//end omp parallel

  free(vec);
  free(cthv);
  free(phiv);
}

static void apodize_mask_smooth_binary(long nside,flouble *mask_in,flouble *mask_out,flouble aposize)
{
  long npix=he_nside2npix(nside);
  double aporad=aposize*M_PI/180;
  memcpy(mask_out,mask_in,npix*sizeof(flouble));

  int lenlist0=(int)(4*npix*(1-cos(2.5*aporad)));
  if(lenlist0 < 2)
    report_error(NMT_ERROR_APO,"Your apodization scale is too small for this pixel size\n");

#pragma omp parallel default(none)                      \
  shared(npix,mask_in,mask_out,nside,aporad,lenlist0)
  {
    long ip;
    int *listpix=my_malloc(lenlist0*sizeof(int));

#pragma omp for schedule(dynamic)
    for(ip=0;ip<npix;ip++) {
      if(mask_in[ip]<=0) {
	int j;
	flouble v[3],cthv,phiv;
	int lenlist_half=lenlist0/2;
	he_pix2vec_ring(nside,ip,v);
	cthv=v[2];
	phiv=atan2(v[1],v[0]);
	if(phiv<0)
	  phiv+=2*M_PI;
	he_query_disc(nside,cthv,phiv,2.5*aporad,listpix,&lenlist_half,1);
	for(j=0;j<lenlist_half;j++) {
	  int ip2=listpix[j];
#pragma omp atomic
	  mask_out[ip2]*=0;
	}
      }
    } //end omp for
    free(listpix);
  }//end omp parallel

}


void nmt_apodize_mask(long nside,flouble *mask_in,flouble *mask_out,flouble aposize,char *apotype)
{
  if(aposize<0)
    report_error(NMT_ERROR_APO,"Apodization scale must be a positive number\n");
  else if(aposize==0)
    memcpy(mask_out,mask_in,he_nside2npix(nside)*sizeof(flouble));
  else {
    if((!strcmp(apotype,"C1")) || (!strcmp(apotype,"C2")))
      apodize_mask_CX(nside,mask_in,mask_out,aposize,apotype);
    else if(!strcmp(apotype,"Smooth"))
      apodize_mask_smooth_binary(nside,mask_in,mask_out,aposize);
    else
      report_error(NMT_ERROR_APO,"Unknown apodization type %s. Allowed: \"Smooth\", \"C1\", \"C2\"\n",apotype);
  }
}
