#include "config.h"
#include "utils.h"

static void apodize_mask_CX(nmt_flatsky_info *fs,flouble *mask_in,flouble *mask_out,
			    flouble aposize,char *apotype)
{
  double aporad=aposize*M_PI/180;
  int apotyp=0;
  if(!strcmp(apotype,"C1"))
    apotyp=0;
  else if(!strcmp(apotype,"C2"))
    apotyp=1;
  else
    report_error(NMT_ERROR_APO,"Unknown apodization type %s\n",apotype);

  if(mask_out!=mask_in)
    memcpy(mask_out,mask_in,fs->nx*fs->ny*sizeof(flouble));

#pragma omp parallel default(none)			\
  shared(fs,mask_in,mask_out,aporad,apotyp)
  {
    int iy;
    flouble **rarr;
    double x_thr=aporad;
    double inv_x_thr=1./x_thr;
    flouble dx=fs->lx/fs->nx;
    flouble dy=fs->lx/fs->nx;
    int nx_patch=(int)(1.2*aporad/dx);
    int ny_patch=(int)(1.2*aporad/dy);
    rarr=my_malloc((2*ny_patch+1)*sizeof(flouble *));
    for(iy=0;iy<=2*ny_patch;iy++) {
      int ix;
      flouble y=(iy-ny_patch)*dy;
      rarr[iy]=my_malloc((2*nx_patch+1)*sizeof(flouble));
      for(ix=0;ix<=2*nx_patch;ix++) {
	flouble x=(ix-nx_patch)*dx;
	rarr[iy][ix]=sqrt(x*x+y*y);
      }
    }

#pragma omp for schedule(dynamic)
    for(iy=0;iy<fs->ny;iy++) {
      int ix;
      for(ix=0;ix<fs->nx;ix++) {
	int index=ix+fs->nx*iy;
	flouble rmin=100000;
	if(mask_in[index]>0) {
	  int iyy;
	  for(iyy=0;iyy<=2*ny_patch;iyy++) {
	    int ixx;
	    if(iy+iyy-ny_patch<0) continue;
	    if(iy+iyy-ny_patch>=fs->ny) break;
	    for(ixx=0;ixx<=2*nx_patch;ixx++) {
	      if(ix+ixx-nx_patch<0) continue;
	      if(ix+ixx-nx_patch>=fs->nx) break;
	      int index2=ix+ixx-nx_patch+fs->nx*(iy+iyy-ny_patch);
	      if(mask_in[index2]<=0)
		if(rarr[iyy][ixx]<rmin) rmin=rarr[iyy][ixx];
	    }
	  }
	  if(rmin<x_thr) {
	    flouble f,xn;
	    if(rmin<0)
	      f=0;
	    else {
	      xn=rmin*inv_x_thr;
	      if(apotyp==0)
		f=xn-sin(xn*2*M_PI)/(2*M_PI);
	      else
		f=0.5*(1-cos(xn*M_PI));
	    }
	    mask_out[index]*=f;
	  }
	}
      }
    } //end omp for
    for(iy=0;iy<=2*ny_patch;iy++)
      free(rarr[iy]);
    free(rarr);
  } //end omp parallel
}

static void apodize_mask_smooth(nmt_flatsky_info *fs,flouble *mask_in,flouble *mask_out,flouble aposize)
{
  long npix=fs->nx*fs->ny;
  double aporad=aposize*M_PI/180;
  flouble *mask_dum=my_malloc(npix*sizeof(flouble));
  fcomplex *alms_dum=my_malloc(fs->ny*(fs->nx/2+1)*sizeof(fcomplex));
  memcpy(mask_dum,mask_in,npix*sizeof(flouble));

#pragma omp parallel default(none)		\
  shared(fs,npix,mask_in,mask_dum,aporad)
  {
    int iy;
    flouble **rarr;
    double x_thr=2.5*aporad;
    flouble dx=fs->lx/fs->nx;
    flouble dy=fs->lx/fs->nx;
    int nx_patch=(int)(1.2*x_thr/dx);
    int ny_patch=(int)(1.2*x_thr/dy);
    rarr=my_malloc((2*ny_patch+1)*sizeof(flouble *));
    for(iy=0;iy<=2*ny_patch;iy++) {
      int ix;
      flouble y=(iy-ny_patch)*dy;
      rarr[iy]=my_malloc((2*nx_patch+1)*sizeof(flouble));
      for(ix=0;ix<=2*nx_patch;ix++) {
	flouble x=(ix-nx_patch)*dx;
	rarr[iy][ix]=sqrt(x*x+y*y);
      }
    }

#pragma omp for schedule(dynamic)
    for(iy=0;iy<fs->ny;iy++) {
      int ix;
      for(ix=0;ix<fs->nx;ix++) {
	int index=ix+fs->nx*iy;
	if(mask_in[index]<=0) {
	  int iyy;
	  for(iyy=0;iyy<=2*ny_patch;iyy++) {
	    int ixx;
	    if(iy+iyy-ny_patch<0) continue;
	    if(iy+iyy-ny_patch>=fs->ny) break;
	    for(ixx=0;ixx<=2*nx_patch;ixx++) {
	      if(ix+ixx-nx_patch<0) continue;
	      if(ix+ixx-nx_patch>=fs->nx) break;
	      if(rarr[iyy][ixx]<=x_thr) {
		int index2=ix+ixx-nx_patch+fs->nx*(iy+iyy-ny_patch);
		mask_dum[index2]*=0;
	      }
	    }
	  }
	}
      }
    } //end omp for
    for(iy=0;iy<=2*ny_patch;iy++)
      free(rarr[iy]);
    free(rarr);
  } //end omp parallel

  fs_map2alm(fs,1,0,&mask_dum,&alms_dum);
  fs_alter_alm(fs,aporad*180*60*2.355/M_PI,alms_dum,alms_dum,NULL,0);
  fs_alm2map(fs,1,0,&mask_dum,&alms_dum);
  fs_map_product(fs,mask_in,mask_dum,mask_out);

  free(mask_dum);
  free(alms_dum);
}

void nmt_apodize_mask_flat(int nx,int ny,flouble lx,flouble ly,
			   flouble *mask_in,flouble *mask_out,flouble aposize,char *apotype)
{
  if(aposize<0)
    report_error(NMT_ERROR_APO,"Apodization scale must be a positive number\n");
  else if(aposize==0) {
    int ii;
    for(ii=0;ii<nx*ny;ii++)
      mask_out[ii]=mask_in[ii];
  }
  else {
    nmt_flatsky_info *fs=nmt_flatsky_info_alloc(nx,ny,lx,ly);
    if((!strcmp(apotype,"C1")) || (!strcmp(apotype,"C2"))) {
      apodize_mask_CX(fs,mask_in,mask_out,aposize,apotype);
    }
    else if(!strcmp(apotype,"Smooth")) 
      apodize_mask_smooth(fs,mask_in,mask_out,aposize);
    else {
      nmt_flatsky_info_free(fs);
      report_error(NMT_ERROR_APO,"Unknown apodization type %s. Allowed: \"Smooth\", \"C1\", \"C2\"\n",apotype);
    }
    nmt_flatsky_info_free(fs);
  }
}

