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

  fs_zero_alm(fsk,alms[0]);
  fs_zero_alm(fsk,alms[1]);
  
  //Inverse FFT
  //Single FFT, spin-0
  fs_alm2map(fsk,1,0,maps,alms);
  //Several FFT, spin-0
  fs_alm2map(fsk,nmaps,0,maps,alms);
  //Single FFT, spin-2
  fs_alm2map(fsk,1,2,maps,alms);
  //Several FFT, spin-2
  fs_alm2map(fsk,nmaps,2,maps,alms);
  
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
