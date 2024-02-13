#include "namaster.h"
#include "ctest.h"
#include "utils.h"
#include "nmt_test_utils.h"
#include <chealpix.h>

CTEST(nmt,mask_flat) {
  int ii;
  double aposize=1.;
  nmt_flatsky_info *fsk=nmt_flatsky_info_alloc(200,200,10*M_PI/180,10*M_PI/180);
  long npix=fsk->npix;
  double *mask=my_calloc(npix,sizeof(double));
  double *mask_C1=my_calloc(npix,sizeof(double));
  double *mask_C2=my_calloc(npix,sizeof(double));
  double *mask_sm=my_calloc(npix,sizeof(double));
  double ioff=(aposize*M_PI/180)/(fsk->lx/fsk->nx);
  double inv_xthr=180/(M_PI*aposize);

  for(ii=0;ii<fsk->ny/2;ii++) {
    int jj;
    for(jj=0;jj<fsk->nx;jj++)
      mask[jj+fsk->nx*ii]=1.;
  }

    //Wrong apotype
  set_error_policy(THROW_ON_ERROR);

  try { nmt_apodize_mask_flat(fsk->nx,fsk->ny,fsk->lx,fsk->ly,mask,mask_C1,aposize,"NONE"); }
  catch(1) {}
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  set_error_policy(EXIT_ON_ERROR);
  nmt_apodize_mask_flat(fsk->nx,fsk->ny,fsk->lx,fsk->ly,mask,mask_C1,aposize,"C1");
  nmt_apodize_mask_flat(fsk->nx,fsk->ny,fsk->lx,fsk->ly,mask,mask_C2,aposize,"C2");
  nmt_apodize_mask_flat(fsk->nx,fsk->ny,fsk->lx,fsk->ly,mask,mask_sm,aposize,"Smooth");

  for(ii=0;ii<fsk->ny;ii++) {
    int jj;
    for(jj=0;jj<fsk->nx;jj++) {
      int index=jj+ii*fsk->nx;
      if(ii>=fsk->ny/2) {
	ASSERT_DBL_NEAR_TOL(0.,mask_C1[index],1E-10);
	ASSERT_DBL_NEAR_TOL(0.,mask_C2[index],1E-10);
	ASSERT_DBL_NEAR_TOL(0.,mask_sm[index],1E-10);
      }
      else if(ii<fsk->ny/2-ioff) { //Can only check for C1 and C2
	ASSERT_DBL_NEAR_TOL(1.,mask_C1[index],1E-10);
	ASSERT_DBL_NEAR_TOL(1.,mask_C2[index],1E-10);
      }
      else { //Can only check for C1 and C2
	double f,xn=inv_xthr*fabs((fsk->ny/2-ii)*fsk->ly/fsk->ny);
	f=xn-sin(xn*2*M_PI)/(2*M_PI);
	ASSERT_DBL_NEAR_TOL(f,mask_C1[index],1E-10);
	f=0.5*(1-cos(xn*M_PI));
	ASSERT_DBL_NEAR_TOL(f,mask_C2[index],1E-10);
      }
    }
  }
  
  nmt_flatsky_info_free(fsk);
  free(mask_C1);
  free(mask_C2);
  free(mask_sm);
  free(mask);
}

CTEST(nmt,mask) {
  int ii;
  long nside=256;
  long npix=he_nside2npix(nside);
  double aposize=2.;
  double th0=M_PI/4; //45-degree cap
  double *mask=my_calloc(npix,sizeof(double));
  double *mask_C1=my_calloc(npix,sizeof(double));
  double *mask_C2=my_calloc(npix,sizeof(double));
  double inv_x2thr=1./(1-cos(aposize*M_PI/180));

  //Add north pole
  for(ii=0;ii<npix;ii++) {
    double th,ph;
    pix2ang_ring(nside,ii,&th,&ph);
    if(th<th0)
      mask[ii]=1.;
  }
  
  //Wrong apotype
  set_error_policy(THROW_ON_ERROR);

  try { nmt_apodize_mask(nside,mask,mask_C1,aposize,"NONE"); }
  catch(1) {}
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  set_error_policy(EXIT_ON_ERROR);

  //Compare with analytical
  nmt_apodize_mask(nside,mask,mask_C1,aposize,"C1");
  nmt_apodize_mask(nside,mask,mask_C2,aposize,"C2");
  for(ii=0;ii<npix;ii++) {
    double th,ph;
    pix2ang_ring(nside,ii,&th,&ph);
    if(mask[ii]==0) {
      ASSERT_DBL_NEAR_TOL(0.,mask_C1[ii],1E-10);
      ASSERT_DBL_NEAR_TOL(0.,mask_C2[ii],1E-10);
    }
    if(th>th0) {
      ASSERT_DBL_NEAR_TOL(0.,mask_C1[ii],1E-10);
      ASSERT_DBL_NEAR_TOL(0.,mask_C2[ii],1E-10);
    }
    else if(th<th0-aposize*M_PI/180) { //Can only check for C1 and C2
      ASSERT_DBL_NEAR_TOL(1.,mask_C1[ii],1E-10);
      ASSERT_DBL_NEAR_TOL(1.,mask_C2[ii],1E-10);
    }
    else { //Can only check for C1 and C2
      double f;
      double x2=1-cos(th-th0);
      double xn=sqrt(x2*inv_x2thr);
      f=xn-sin(xn*2*M_PI)/(2*M_PI);
      ASSERT_DBL_NEAR_TOL(f,mask_C1[ii],2E-2);
      f=0.5*(1-cos(xn*M_PI));
      ASSERT_DBL_NEAR_TOL(f,mask_C2[ii],2E-2);
    }
  }

  free(mask_C1);
  free(mask_C2);
  free(mask);
}

CTEST(nmt,mask_error) {
  int ii;
  long nside=2;
  long npix=he_nside2npix(nside);
  double aposize=2.;
  double th0=M_PI/4; //45-degree cap
  double *mask=my_calloc(npix,sizeof(double));
  double *mask_apo=my_calloc(npix,sizeof(double));

  //Add north pole
  for(ii=0;ii<npix;ii++) {
    double th,ph;
    pix2ang_ring(nside,ii,&th,&ph);
    if(th<th0)
      mask[ii]=1.;
  }
  
  set_error_policy(THROW_ON_ERROR);

  try { nmt_apodize_mask(nside,mask,mask_apo,aposize,"C1"); }
  catch(1) {}
  ASSERT_NOT_EQUAL(0,nmt_exception_status);

  try { nmt_apodize_mask(nside,mask,mask_apo,aposize,"C2"); }
  catch(1) {}
  ASSERT_NOT_EQUAL(0,nmt_exception_status);

  set_error_policy(EXIT_ON_ERROR);

  free(mask_apo);
  free(mask);
}
