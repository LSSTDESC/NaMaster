#include "namaster.h"
#include "ctest.h"
#include "utils.h"
#include "nmt_test_utils.h"
#include <chealpix.h>

CTEST(nmt,fft_malloc) {
  set_error_policy(THROW_ON_ERROR);

  printf("\nError messages expected: \n");

  double *dum=dftw_malloc(10);
  free(dum);
  try{ dftw_malloc(-1); }
  catch(1) {}
  ASSERT_EQUAL(1,exception_status);

  set_error_policy(EXIT_ON_ERROR);
}

CTEST(nmt,fsk_info) {
  nmt_flatsky_info *fsk=nmt_flatsky_info_alloc(100,100,M_PI/180,M_PI/180);
  ASSERT_EQUAL(10000,fsk->npix);
  ASSERT_EQUAL(pow(M_PI/180,2)/10000,fsk->pixsize);
  nmt_flatsky_info_free(fsk);
}

CTEST(nmt,fsk_fft) {
  int ii;
  int nmaps=34;
  nmt_flatsky_info *fsk=nmt_flatsky_info_alloc(141,311,M_PI/180,M_PI/180);
  double **maps=my_malloc(2*nmaps*sizeof(double *));
  fcomplex **alms=my_malloc(2*nmaps*sizeof(fcomplex *));

  for(ii=0;ii<2*nmaps;ii++) {
    maps[ii]=my_calloc(fsk->npix,sizeof(double));
    alms[ii]=dftw_malloc(fsk->ny*(fsk->nx/2+1)*sizeof(fcomplex));
  }

  //Direct FFT
  //Single FFT, spin-0
  fs_map2alm(fsk,1,0,maps,alms);
  //Several FFT, spin-0
  fs_map2alm(fsk,nmaps,0,maps,alms);
  //Single FFT, spin-2
  fs_map2alm(fsk,1,2,maps,alms);
  //Several FFT, spin-2
  fs_map2alm(fsk,nmaps,2,maps,alms);

  //Zero_alm and alter_alm
  fs_zero_alm(fsk,alms[0]);
  fs_zero_alm(fsk,alms[1]);
  nmt_k_function *b=fs_generate_beam_window(10.);
  fs_alter_alm(fsk,10.,alms[0],alms[1],NULL,0);
  fs_alter_alm(fsk,10.,alms[0],alms[1],b,0);
  fs_alter_alm(fsk,10.,alms[0],alms[1],b,1);
  nmt_k_function_free(b);
  
  //Inverse FFT
  //Single FFT, spin-0
  fs_alm2map(fsk,1,0,maps,alms);
  //Several FFT, spin-0
  fs_alm2map(fsk,nmaps,0,maps,alms);
  //Single FFT, spin-2
  fs_alm2map(fsk,1,2,maps,alms);
  //Several FFT, spin-2
  fs_alm2map(fsk,nmaps,2,maps,alms);
  
  //Particular example
  //Spin-0. map = 2*pi/A * Re[exp(i*k0*x)] ->
  //        a(k) = (delta_{k,k0}+delta_{k,-k0})/2
  int i0_x=2,i0_y=3;
  double k0_x=i0_x*2*M_PI/fsk->lx;
  double k0_y=i0_y*2*M_PI/fsk->ly;
  double cphi0=k0_x/sqrt(k0_x*k0_x+k0_y*k0_y);
  double sphi0=k0_y/sqrt(k0_x*k0_x+k0_y*k0_y);
  double c2phi0=cphi0*cphi0-sphi0*sphi0;
  double s2phi0=2*sphi0*cphi0;
  for(ii=0;ii<fsk->ny;ii++) {
    int jj;
    double y=ii*fsk->ly/fsk->ny;
    for(jj=0;jj<fsk->nx;jj++) {
      double x=jj*fsk->lx/fsk->nx;
      double phase=k0_x*x+k0_y*y;
      maps[0][jj+fsk->nx*ii]=2*M_PI*cos(phase)/(fsk->lx*fsk->ly);
    }
  }
  fs_map2alm(fsk,1,0,maps,alms);
  for(ii=0;ii<fsk->ny;ii++) {
    int jj;
    for(jj=0;jj<=fsk->nx/2;jj++) {
      double re=creal(alms[0][jj+(fsk->nx/2+1)*ii]);
      double im=cimag(alms[0][jj+(fsk->nx/2+1)*ii]);
      if((jj==i0_x) && (ii==i0_y)) {
	ASSERT_DBL_NEAR_TOL(0.5,re,1E-5);
	ASSERT_DBL_NEAR_TOL(0.0,im,1E-5);
      }
      else {
	ASSERT_DBL_NEAR_TOL(0.0,re,1E-5);
	ASSERT_DBL_NEAR_TOL(0.0,im,1E-5);
      }
    }
  }
  //Spin-2. map = 2*pi/A * (cos(2*phi_k0),-sin(2*phi_k0)) Re[exp(i*k0*x)] ->
  //        a_E(k) = (delta_{k,k0}+delta_{k,-k0})/2
  //        a_B(k) = 0
  for(ii=0;ii<fsk->ny;ii++) {
    int jj;
    double y=ii*fsk->ly/fsk->ny;
    for(jj=0;jj<fsk->nx;jj++) {
      double x=jj*fsk->lx/fsk->nx;
      double phase=k0_x*x+k0_y*y;
      maps[0][jj+fsk->nx*ii]= 2*M_PI*c2phi0*cos(phase)/(fsk->lx*fsk->ly);
      maps[1][jj+fsk->nx*ii]=-2*M_PI*s2phi0*cos(phase)/(fsk->lx*fsk->ly);
    }
  }
  fs_map2alm(fsk,1,2,maps,alms);
  for(ii=0;ii<fsk->ny;ii++) {
    int jj;
    for(jj=0;jj<=fsk->nx/2;jj++) {
      double re0=creal(alms[0][jj+(fsk->nx/2+1)*ii]);
      double im0=cimag(alms[0][jj+(fsk->nx/2+1)*ii]);
      double re1=creal(alms[1][jj+(fsk->nx/2+1)*ii]);
      double im1=cimag(alms[1][jj+(fsk->nx/2+1)*ii]);
      if((jj==i0_x) && (ii==i0_y)) {
	ASSERT_DBL_NEAR_TOL(0.5,re0,1E-5);
	ASSERT_DBL_NEAR_TOL(0.0,im0,1E-5);
	ASSERT_DBL_NEAR_TOL(0.0,re1,1E-5);
	ASSERT_DBL_NEAR_TOL(0.0,im1,1E-5);
      }
      else {
	ASSERT_DBL_NEAR_TOL(0.0,re0,1E-5);
	ASSERT_DBL_NEAR_TOL(0.0,im0,1E-5);
	ASSERT_DBL_NEAR_TOL(0.0,re1,1E-5);
	ASSERT_DBL_NEAR_TOL(0.0,im1,1E-5);
      }
    }
  }

  for(ii=0;ii<2*nmaps;ii++) {
    dftw_free(maps[ii]);
    dftw_free(alms[ii]);
  }
  free(maps);
  free(alms);
  nmt_flatsky_info_free(fsk);
}
  
CTEST(nmt,fsk_algb) {
  int ii;
  nmt_flatsky_info *fsk=nmt_flatsky_info_alloc(100,100,M_PI/180,M_PI/180);
  double *mp1=my_malloc(fsk->npix*sizeof(double));
  double *mp2=my_malloc(fsk->npix*sizeof(double));
  double *mpr=my_malloc(fsk->npix*sizeof(double));

  for(ii=0;ii<fsk->npix;ii++) {
    mp1[ii]=2.;
    mp2[ii]=0.5;
  }

  double d=fs_map_dot(fsk,mp1,mp2);
  fs_map_product(fsk,mp1,mp2,mpr);
  fs_map_product(fsk,mp1,mp2,mp2);
  for(ii=0;ii<fsk->npix;ii++) {
    ASSERT_DBL_NEAR_TOL(1.,mpr[ii],1E-10);
    ASSERT_DBL_NEAR_TOL(1.,mp2[ii],1E-10);
  }
  ASSERT_DBL_NEAR_TOL(pow(M_PI/180,2),d,1E-5);
  
  free(mp1);
  free(mp2);
  free(mpr);
  nmt_flatsky_info_free(fsk);
}


static double fk(double k)
{
  return 100./(k+100.);
}

CTEST(nmt,fsk_func) {
  int l;
  long lmax=2000;
  double *karr=my_malloc((lmax+1)*sizeof(double));
  double *farr=my_malloc((lmax+1)*sizeof(double));

  for(l=0;l<=lmax;l++) {
    karr[l]=l;
    farr[l]=fk(karr[l]);
  }

  nmt_k_function *kf=nmt_k_function_alloc(lmax+1,karr,farr,1.,0.,0);

  for(l=0;l<lmax;l++) {
    double k=l+0.5;
    double f_int=nmt_k_function_eval(kf,k,NULL);
    double f_exc=fk(k);
    ASSERT_DBL_NEAR_TOL(1.,f_int/f_exc,1E-3);
  }
  
  nmt_k_function_free(kf);

  //Beams
  double sigma=1.*M_PI/180; //Beam sigma in radians
  double fwhm_amin=sigma*180*60/M_PI*2.35482;
  kf=fs_generate_beam_window(fwhm_amin);
  for(l=0;l<100;l++) {
    double ll=(l+0.5)*4.8/(100.*sigma);
    double b=nmt_k_function_eval(kf,ll,NULL);
    double bt=exp(-0.5*ll*ll*sigma*sigma);
    ASSERT_DBL_NEAR_TOL(1.,b/bt,1E-3);
  }
  nmt_k_function_free(kf);
  
  free(karr);
  free(farr);
}
