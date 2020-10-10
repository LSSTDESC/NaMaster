#include "namaster.h"
#include "ctest.h"
#include "utils.h"
#include "nmt_test_utils.h"

CTEST(nmt,field_empty) {
  nmt_field *f;
  int ii,jj,nmaps;
  double ntemp=5;
  long nside=128;
  nmt_curvedsky_info *cs=nmt_curvedsky_info_alloc(1,nside,-1,-1,-1,-1,-1,-1,-1);
  long lmax=he_get_lmax(cs);
  long npix=he_nside2npix(nside);
  double *beam=my_malloc((lmax+1)*sizeof(double));
  double *mask=my_malloc(npix*sizeof(double));
  
  for(ii=0;ii<npix;ii++)
    mask[ii]=1.;
  
  for(ii=0;ii<=lmax;ii++)
    beam[ii]=1.;

  ////////
  //Spin-2
  //With purification
  f=nmt_field_alloc_sph(cs,mask,2,NULL,0,NULL,beam,1,1,5,1E-5,HE_NITER_DEFAULT,0,1,1);
  //Sanity checks
  ASSERT_EQUAL(2,f->spin);
  ASSERT_EQUAL(2,f->nmaps);
  nmt_field_free(f);

  free(beam);
  free(mask);
  free(cs);
}

CTEST(nmt,field_lite) {
  nmt_field *f;
  int ii,nmaps;
  double ntemp=5;
  long nside=128;
  nmt_curvedsky_info *cs=nmt_curvedsky_info_alloc(1,nside,-1,-1,-1,-1,-1,-1,-1);
  long lmax=he_get_lmax(cs);
  long npix=he_nside2npix(nside);
  double **maps;
  double ***temp=my_malloc(ntemp*sizeof(double **));
  double *beam=my_malloc((lmax+1)*sizeof(double));
  double *mask=my_malloc(npix*sizeof(double));
  
  for(ii=0;ii<npix;ii++)
    mask[ii]=1.;
  
  for(ii=0;ii<=lmax;ii++)
    beam[ii]=1.;

  ////////
  //Spin-2
  nmaps=2;
  //Create inputs
  maps=test_make_map_analytic(nside,2);

  //With purification
  f=nmt_field_alloc_sph(cs,mask,2,maps,0,NULL,beam,1,1,5,1E-5,HE_NITER_DEFAULT,0,1,0);
  //Sanity checks
  ASSERT_EQUAL(2,f->spin);
  ASSERT_EQUAL(2,f->nmaps);
  //Harmonic transform
  ASSERT_DBL_NEAR_TOL(1.,creal(f->alms[0][he_indexlm(2,0,lmax)]),1E-4);
  ASSERT_DBL_NEAR_TOL(0.,cimag(f->alms[0][he_indexlm(2,0,lmax)]),1E-4);
  ASSERT_DBL_NEAR_TOL(0.,creal(f->alms[1][he_indexlm(2,0,lmax)]),1E-4);
  ASSERT_DBL_NEAR_TOL(0.,cimag(f->alms[1][he_indexlm(2,0,lmax)]),1E-4);
  ASSERT_DBL_NEAR_TOL(0.,creal(f->alms[0][he_indexlm(1,0,lmax)]),1E-4);
  ASSERT_DBL_NEAR_TOL(0.,cimag(f->alms[0][he_indexlm(1,0,lmax)]),1E-4);
  ASSERT_DBL_NEAR_TOL(0.,creal(f->alms[1][he_indexlm(1,0,lmax)]),1E-4);
  ASSERT_DBL_NEAR_TOL(0.,cimag(f->alms[1][he_indexlm(1,0,lmax)]),1E-4);
  ASSERT_DBL_NEAR_TOL(0.,creal(f->alms[0][he_indexlm(3,0,lmax)]),1E-4);
  ASSERT_DBL_NEAR_TOL(0.,cimag(f->alms[0][he_indexlm(3,0,lmax)]),1E-4);
  ASSERT_DBL_NEAR_TOL(2.,creal(f->alms[1][he_indexlm(3,0,lmax)]),1E-4);
  ASSERT_DBL_NEAR_TOL(0.,cimag(f->alms[1][he_indexlm(3,0,lmax)]),1E-4);
  nmt_field_free(f);
  for(ii=0;ii<nmaps;ii++)
    free(maps[ii]);
  free(maps);

  //With templates
  maps=test_make_map_analytic(nside,2);
  for(ii=0;ii<ntemp;ii++)
    temp[ii]=test_make_map_analytic(nside,2);
  f=nmt_field_alloc_sph(cs,mask,2,maps,ntemp,temp,beam,0,0,0,1E-5,HE_NITER_DEFAULT,0,1,0);
  //Since maps and templates are the same, template-deprojected alms should be 0
  ASSERT_DBL_NEAR_TOL(0,creal(f->alms[0][he_indexlm(2,0,lmax)]),1E-5);
  ASSERT_DBL_NEAR_TOL(0.,cimag(f->alms[0][he_indexlm(2,0,lmax)]),1E-5);
  ASSERT_DBL_NEAR_TOL(0.,creal(f->alms[1][he_indexlm(2,0,lmax)]),1E-5);
  ASSERT_DBL_NEAR_TOL(0.,cimag(f->alms[1][he_indexlm(2,0,lmax)]),1E-5);
  ASSERT_DBL_NEAR_TOL(0.,creal(f->alms[0][he_indexlm(1,0,lmax)]),1E-5);
  ASSERT_DBL_NEAR_TOL(0.,cimag(f->alms[0][he_indexlm(1,0,lmax)]),1E-5);
  ASSERT_DBL_NEAR_TOL(0.,creal(f->alms[1][he_indexlm(1,0,lmax)]),1E-5);
  ASSERT_DBL_NEAR_TOL(0.,cimag(f->alms[1][he_indexlm(1,0,lmax)]),1E-5);
  ASSERT_DBL_NEAR_TOL(0.,creal(f->alms[0][he_indexlm(3,0,lmax)]),1E-5);
  ASSERT_DBL_NEAR_TOL(0.,cimag(f->alms[0][he_indexlm(3,0,lmax)]),1E-5);
  ASSERT_DBL_NEAR_TOL(0.,creal(f->alms[1][he_indexlm(3,0,lmax)]),1E-5);
  ASSERT_DBL_NEAR_TOL(0.,cimag(f->alms[1][he_indexlm(3,0,lmax)]),1E-5);
  nmt_field_free(f);
  for(ii=0;ii<ntemp;ii++) {
    int jj;
    for(jj=0;jj<nmaps;jj++)
      free(temp[ii][jj]);
    free(temp[ii]);
  }
  for(ii=0;ii<nmaps;ii++)
    free(maps[ii]);
  free(maps);
  ////////
  
  free(temp);
  free(beam);
  free(mask);
  free(cs);
}

CTEST(nmt,field_alloc) {
  nmt_field *f;
  int ii,jj,nmaps;
  double ntemp=5;
  long nside=128;
  nmt_curvedsky_info *cs=nmt_curvedsky_info_alloc(1,nside,-1,-1,-1,-1,-1,-1,-1);
  long lmax=he_get_lmax(cs);
  long npix=he_nside2npix(nside);
  double **maps;
  double ***temp=my_malloc(ntemp*sizeof(double **));
  double *beam=my_malloc((lmax+1)*sizeof(double));
  double *mask=my_malloc(npix*sizeof(double));
  
  for(ii=0;ii<npix;ii++)
    mask[ii]=1.;
  
  for(ii=0;ii<=lmax;ii++)
    beam[ii]=1.;

  ////////
  //Spin-0
  nmaps=1;
  //Create inputs
  maps=test_make_map_analytic(nside,0);
  for(ii=0;ii<ntemp;ii++)
    temp[ii]=test_make_map_analytic(nside,0);

  //No templates
  f=nmt_field_alloc_sph(cs,mask,0,maps,0,NULL,beam,0,0,0,1E-5,HE_NITER_DEFAULT,0,0,0);
  //Sanity checks
  ASSERT_EQUAL(lmax,f->lmax);
  ASSERT_EQUAL(0,f->pure_e);
  ASSERT_EQUAL(0,f->pure_b);
  ASSERT_EQUAL(npix,f->cs->npix);
  ASSERT_EQUAL(nside,f->cs->n_eq);
  ASSERT_EQUAL(0,f->spin);
  ASSERT_EQUAL(1,f->nmaps);
  //Harmonic transform
  ASSERT_DBL_NEAR_TOL(0.5,creal(f->alms[0][he_indexlm(2,2,lmax)]),1E-5);
  ASSERT_DBL_NEAR_TOL(0.0,cimag(f->alms[0][he_indexlm(2,2,lmax)]),1E-5);
  nmt_field_free(f);
  
  //With templates
  f=nmt_field_alloc_sph(cs,mask,0,maps,ntemp,temp,NULL,0,0,0,1E-5,HE_NITER_DEFAULT,0,0,0);
  //Since maps and templates are the same, template-deprojected map should be 0
  for(ii=0;ii<npix;ii++)
    ASSERT_DBL_NEAR_TOL(0.0,f->maps[0][ii],1E-10);
  for(ii=0;ii<ntemp;ii++) {
    ASSERT_DBL_NEAR_TOL(0.5,creal(f->a_temp[ii][0][he_indexlm(2,2,lmax)]),1E-5);
    ASSERT_DBL_NEAR_TOL(0.0,cimag(f->a_temp[ii][0][he_indexlm(2,2,lmax)]),1E-5);
  }
  nmt_field_free(f);
  
  //Free inputs
  for(ii=0;ii<ntemp;ii++) {
    for(jj=0;jj<nmaps;jj++)
      free(temp[ii][jj]);
  }
  for(ii=0;ii<nmaps;ii++)
    free(maps[ii]);
  free(maps);
  ////////

  ////////
  //Spin-1
  nmaps=2;
  //Create inputs
  maps=test_make_map_analytic(nside,1);

  //No templates
  f=nmt_field_alloc_sph(cs,mask,1,maps,0,NULL,beam,0,0,0,1E-5,HE_NITER_DEFAULT,0,0,0);
  //Sanity checks
  ASSERT_EQUAL(1,f->spin);
  ASSERT_EQUAL(2,f->nmaps);
  //Harmonic transform
  ASSERT_DBL_NEAR_TOL(1.,creal(f->alms[0][he_indexlm(1,0,lmax)]),1E-5);
  ASSERT_DBL_NEAR_TOL(0.,cimag(f->alms[0][he_indexlm(1,0,lmax)]),1E-5);
  ASSERT_DBL_NEAR_TOL(0.,creal(f->alms[0][he_indexlm(1,1,lmax)]),1E-5);
  ASSERT_DBL_NEAR_TOL(0.,cimag(f->alms[0][he_indexlm(1,1,lmax)]),1E-5);
  ASSERT_DBL_NEAR_TOL(3.,creal(f->alms[1][he_indexlm(1,0,lmax)]),1E-5);
  ASSERT_DBL_NEAR_TOL(0.,cimag(f->alms[1][he_indexlm(1,0,lmax)]),1E-5);
  ASSERT_DBL_NEAR_TOL(0.,creal(f->alms[1][he_indexlm(1,1,lmax)]),1E-5);
  ASSERT_DBL_NEAR_TOL(0.,cimag(f->alms[1][he_indexlm(1,1,lmax)]),1E-5);
  nmt_field_free(f);
  for(ii=0;ii<nmaps;ii++)
    free(maps[ii]);
  free(maps);
  ////////

  ////////
  //Spin-2
  nmaps=2;
  //Create inputs
  maps=test_make_map_analytic(nside,2);
  for(ii=0;ii<ntemp;ii++)
    temp[ii]=test_make_map_analytic(nside,2);

  //No templates
  f=nmt_field_alloc_sph(cs,mask,2,maps,0,NULL,beam,0,0,0,1E-5,HE_NITER_DEFAULT,0,0,0);
  //Sanity checks
  ASSERT_EQUAL(2,f->spin);
  ASSERT_EQUAL(2,f->nmaps);
  //Harmonic transform
  ASSERT_DBL_NEAR_TOL(1.,creal(f->alms[0][he_indexlm(2,0,lmax)]),1E-5);
  ASSERT_DBL_NEAR_TOL(0.,cimag(f->alms[0][he_indexlm(2,0,lmax)]),1E-5);
  ASSERT_DBL_NEAR_TOL(0.,creal(f->alms[1][he_indexlm(2,0,lmax)]),1E-5);
  ASSERT_DBL_NEAR_TOL(0.,cimag(f->alms[1][he_indexlm(2,0,lmax)]),1E-5);
  ASSERT_DBL_NEAR_TOL(0.,creal(f->alms[0][he_indexlm(1,0,lmax)]),1E-5);
  ASSERT_DBL_NEAR_TOL(0.,cimag(f->alms[0][he_indexlm(1,0,lmax)]),1E-5);
  ASSERT_DBL_NEAR_TOL(0.,creal(f->alms[1][he_indexlm(1,0,lmax)]),1E-5);
  ASSERT_DBL_NEAR_TOL(0.,cimag(f->alms[1][he_indexlm(1,0,lmax)]),1E-5);
  ASSERT_DBL_NEAR_TOL(0.,creal(f->alms[0][he_indexlm(3,0,lmax)]),1E-5);
  ASSERT_DBL_NEAR_TOL(0.,cimag(f->alms[0][he_indexlm(3,0,lmax)]),1E-5);
  ASSERT_DBL_NEAR_TOL(2.,creal(f->alms[1][he_indexlm(3,0,lmax)]),1E-5);
  ASSERT_DBL_NEAR_TOL(0.,cimag(f->alms[1][he_indexlm(3,0,lmax)]),1E-5);
  nmt_field_free(f);

  //With purification (nothing should change)
  for(ii=0;ii<nmaps;ii++)
    free(maps[ii]);
  maps=test_make_map_analytic(nside,2);
  for(ii=0;ii<ntemp;ii++) {
    for(jj=0;jj<nmaps;jj++)
      free(temp[ii][jj]);
    temp[ii]=test_make_map_analytic(nside,2);
  }
  f=nmt_field_alloc_sph(cs,mask,2,maps,0,NULL,beam,1,1,5,1E-5,HE_NITER_DEFAULT,0,0,0);
  //Sanity checks
  ASSERT_EQUAL(2,f->spin);
  ASSERT_EQUAL(2,f->nmaps);
  //Harmonic transform
  ASSERT_DBL_NEAR_TOL(1.,creal(f->alms[0][he_indexlm(2,0,lmax)]),1E-4);
  ASSERT_DBL_NEAR_TOL(0.,cimag(f->alms[0][he_indexlm(2,0,lmax)]),1E-4);
  ASSERT_DBL_NEAR_TOL(0.,creal(f->alms[1][he_indexlm(2,0,lmax)]),1E-4);
  ASSERT_DBL_NEAR_TOL(0.,cimag(f->alms[1][he_indexlm(2,0,lmax)]),1E-4);
  ASSERT_DBL_NEAR_TOL(0.,creal(f->alms[0][he_indexlm(1,0,lmax)]),1E-4);
  ASSERT_DBL_NEAR_TOL(0.,cimag(f->alms[0][he_indexlm(1,0,lmax)]),1E-4);
  ASSERT_DBL_NEAR_TOL(0.,creal(f->alms[1][he_indexlm(1,0,lmax)]),1E-4);
  ASSERT_DBL_NEAR_TOL(0.,cimag(f->alms[1][he_indexlm(1,0,lmax)]),1E-4);
  ASSERT_DBL_NEAR_TOL(0.,creal(f->alms[0][he_indexlm(3,0,lmax)]),1E-4);
  ASSERT_DBL_NEAR_TOL(0.,cimag(f->alms[0][he_indexlm(3,0,lmax)]),1E-4);
  ASSERT_DBL_NEAR_TOL(2.,creal(f->alms[1][he_indexlm(3,0,lmax)]),1E-4);
  ASSERT_DBL_NEAR_TOL(0.,cimag(f->alms[1][he_indexlm(3,0,lmax)]),1E-4);
  nmt_field_free(f);
  
  //With templates
  for(ii=0;ii<nmaps;ii++)
    free(maps[ii]);
  maps=test_make_map_analytic(nside,2);
  for(ii=0;ii<ntemp;ii++) {
    for(jj=0;jj<nmaps;jj++)
      free(temp[ii][jj]);
    temp[ii]=test_make_map_analytic(nside,2);
  }
  f=nmt_field_alloc_sph(cs,mask,2,maps,ntemp,temp,beam,0,0,0,1E-5,HE_NITER_DEFAULT,0,0,0);
  //Since maps and templates are the same, template-deprojected map should be 0
  for(ii=0;ii<nmaps;ii++) {
    for(jj=0;jj<npix;jj++)
      ASSERT_DBL_NEAR_TOL(0.0,f->maps[ii][jj],1E-10);
  }
  for(ii=0;ii<ntemp;ii++) {
    ASSERT_DBL_NEAR_TOL(1.,creal(f->a_temp[ii][0][he_indexlm(2,0,lmax)]),1E-5);
    ASSERT_DBL_NEAR_TOL(0.,cimag(f->a_temp[ii][0][he_indexlm(2,0,lmax)]),1E-5);
    ASSERT_DBL_NEAR_TOL(0.,creal(f->a_temp[ii][1][he_indexlm(2,0,lmax)]),1E-5);
    ASSERT_DBL_NEAR_TOL(0.,cimag(f->a_temp[ii][1][he_indexlm(2,0,lmax)]),1E-5);
    ASSERT_DBL_NEAR_TOL(0.,creal(f->a_temp[ii][0][he_indexlm(1,0,lmax)]),1E-5);
    ASSERT_DBL_NEAR_TOL(0.,cimag(f->a_temp[ii][0][he_indexlm(1,0,lmax)]),1E-5);
    ASSERT_DBL_NEAR_TOL(0.,creal(f->a_temp[ii][1][he_indexlm(1,0,lmax)]),1E-5);
    ASSERT_DBL_NEAR_TOL(0.,cimag(f->a_temp[ii][1][he_indexlm(1,0,lmax)]),1E-5);
    ASSERT_DBL_NEAR_TOL(0.,creal(f->a_temp[ii][0][he_indexlm(3,0,lmax)]),1E-5);
    ASSERT_DBL_NEAR_TOL(0.,cimag(f->a_temp[ii][0][he_indexlm(3,0,lmax)]),1E-5);
    ASSERT_DBL_NEAR_TOL(2.,creal(f->a_temp[ii][1][he_indexlm(3,0,lmax)]),1E-5);
    ASSERT_DBL_NEAR_TOL(0.,cimag(f->a_temp[ii][1][he_indexlm(3,0,lmax)]),1E-5);
  }
  nmt_field_free(f);
  
  //With templates and purification (nothing should change)
  for(ii=0;ii<nmaps;ii++)
    free(maps[ii]);
  maps=test_make_map_analytic(nside,2);
  for(ii=0;ii<ntemp;ii++) {
    for(jj=0;jj<nmaps;jj++)
      free(temp[ii][jj]);
    temp[ii]=test_make_map_analytic(nside,2);
  }
  f=nmt_field_alloc_sph(cs,mask,2,maps,ntemp,temp,beam,1,1,5,1E-5,HE_NITER_DEFAULT,0,0,0);
  //Since maps and templates are the same, template-deprojected map should be 0
  for(ii=0;ii<nmaps;ii++) {
    for(jj=0;jj<npix;jj++)
      ASSERT_DBL_NEAR_TOL(0.0,f->maps[ii][jj],1E-10);
  }
  for(ii=0;ii<ntemp;ii++) {
    ASSERT_DBL_NEAR_TOL(1.,creal(f->a_temp[ii][0][he_indexlm(2,0,lmax)]),1E-4);
    ASSERT_DBL_NEAR_TOL(0.,cimag(f->a_temp[ii][0][he_indexlm(2,0,lmax)]),1E-4);
    ASSERT_DBL_NEAR_TOL(0.,creal(f->a_temp[ii][1][he_indexlm(2,0,lmax)]),1E-4);
    ASSERT_DBL_NEAR_TOL(0.,cimag(f->a_temp[ii][1][he_indexlm(2,0,lmax)]),1E-4);
    ASSERT_DBL_NEAR_TOL(0.,creal(f->a_temp[ii][0][he_indexlm(1,0,lmax)]),1E-4);
    ASSERT_DBL_NEAR_TOL(0.,cimag(f->a_temp[ii][0][he_indexlm(1,0,lmax)]),1E-4);
    ASSERT_DBL_NEAR_TOL(0.,creal(f->a_temp[ii][1][he_indexlm(1,0,lmax)]),1E-4);
    ASSERT_DBL_NEAR_TOL(0.,cimag(f->a_temp[ii][1][he_indexlm(1,0,lmax)]),1E-4);
    ASSERT_DBL_NEAR_TOL(0.,creal(f->a_temp[ii][0][he_indexlm(3,0,lmax)]),1E-4);
    ASSERT_DBL_NEAR_TOL(0.,cimag(f->a_temp[ii][0][he_indexlm(3,0,lmax)]),1E-4);
    ASSERT_DBL_NEAR_TOL(2.,creal(f->a_temp[ii][1][he_indexlm(3,0,lmax)]),1E-4);
    ASSERT_DBL_NEAR_TOL(0.,cimag(f->a_temp[ii][1][he_indexlm(3,0,lmax)]),1E-4);
  }
  nmt_field_free(f);
  
  //Free inputs
  for(ii=0;ii<ntemp;ii++) {
    for(jj=0;jj<nmaps;jj++)
      free(temp[ii][jj]);
  }
  for(ii=0;ii<nmaps;ii++)
    free(maps[ii]);
  free(maps);
  ////////
  
  for(ii=0;ii<ntemp;ii++)
    free(temp[ii]);
  free(temp);
  free(beam);
  free(mask);
  free(cs);
}

CTEST(nmt,field_read) {
  int ii;
  nmt_field *f;

  //Spin-0, no templates
  f=nmt_field_read(1,"test/mask.fits","test/maps.fits","none","none",0,0,0,3,1E-10,HE_NITER_DEFAULT);
  ASSERT_EQUAL(f->cs->n_eq,256);
  nmt_field_free(f);

  //Spin-0, with templates
  f=nmt_field_read(1,"test/mask.fits","test/maps.fits","test/maps.fits","none",0,0,0,3,1E-10,HE_NITER_DEFAULT);
  ASSERT_EQUAL(f->cs->n_eq,256);
  //Template=map -> map=0
  for(ii=0;ii<f->cs->npix;ii++)
    ASSERT_DBL_NEAR_TOL(0.0,f->maps[0][ii],1E-10);
  nmt_field_free(f);

  //Spin-2, no templates
  f=nmt_field_read(1,"test/mask.fits","test/maps.fits","none","none",1,0,0,3,1E-10,HE_NITER_DEFAULT);
  ASSERT_EQUAL(f->cs->n_eq,256);
  nmt_field_free(f);

  //Spin-2, with templates
  f=NULL;
  //Check that an error is thrown if file is wrong
  set_error_policy(THROW_ON_ERROR);
  try { f=nmt_field_read(1,"test/mask.fits","test/maps.fits","test/maps.fits","none",1,0,0,3,1E-10,HE_NITER_DEFAULT); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(f);
  set_error_policy(EXIT_ON_ERROR);
}

CTEST(nmt,field_synfast) {
  int ii,im1,im2,l,if1,if2;
  long nside=128;
  nmt_curvedsky_info *cs=nmt_curvedsky_info_alloc(1,nside,-1,-1,-1,-1,-1,-1,-1);
  long lmax=he_get_lmax(cs);
  int nfields=3;
  int field_spins[3]={0,1,2};
  int field_nmaps[3]={1,2,2};
  int nmaps=5;
  int ncls_pass=(nmaps*(nmaps+1))/2;
  int ncls=nmaps*nmaps;
  double lpivot=nside/2.;
  double alpha_pivot=1.;
  double **cells_in=my_malloc(ncls*sizeof(double *));
  double **cells_pass=my_malloc(ncls_pass*sizeof(double *));
  double **cells_out=my_malloc(ncls*sizeof(double *));
  double **beam=my_malloc(nfields*sizeof(double *));

  for(ii=0;ii<nfields;ii++) {
    beam[ii]=my_malloc((lmax+1)*sizeof(double));
    for(l=0;l<=lmax;l++)
      beam[ii][l]=1.;
  }

  for(im1=0;im1<nmaps;im1++) {
    for(im2=0;im2<nmaps;im2++) {
      int index=im2+nmaps*im1;
      cells_in[index]=my_malloc((lmax+1)*sizeof(double));
      cells_out[index]=my_malloc((lmax+1)*sizeof(double));
      for(l=0;l<=lmax;l++) {
	if(im1==im2)
	  cells_in[index][l]=pow((2*lpivot)/(l+lpivot),alpha_pivot);
	else
	  cells_in[index][l]=0;
      }
    }
  }
  int icl=0;
  for(im1=0;im1<nmaps;im1++) {
    for(im2=im1;im2<nmaps;im2++) {
      cells_pass[icl]=cells_in[im2+nmaps*im1];
      icl++;
    }
  }

  //Generate maps
  flouble **maps=nmt_synfast_sph(cs,nfields,field_spins,lmax,cells_pass,beam,1234);

  //Compute power spectra
  im1=0;
  for(if1=0;if1<nfields;if1++) {
    im2=0;
    for(if2=0;if2<nfields;if2++) {
      int ncls_here=field_nmaps[if1]*field_nmaps[if2];
      double **cells_here=my_malloc(ncls_here*sizeof(double *));
      for(ii=0;ii<ncls_here;ii++)
	cells_here[ii]=my_malloc((lmax+1)*sizeof(double));
      
      he_anafast(&(maps[im1]),&(maps[im2]),field_spins[if1],field_spins[if2],
		 cells_here,cs,lmax,3);

      int i1;
      for(i1=0;i1<field_nmaps[if1];i1++) {
      	int i2;
      	for(i2=0;i2<field_nmaps[if2];i2++) {
	  int index_here=i2+field_nmaps[if2]*i1;
	  int index_out=im2+i2+nmaps*(im1+i1);
	  for(l=0;l<=lmax;l++)
	    cells_out[index_out][l]=cells_here[index_here][l];
	}
      }

      for(ii=0;ii<ncls_here;ii++)
      	free(cells_here[ii]);
      free(cells_here);
      im2+=field_nmaps[if2];
    }
    im1+=field_nmaps[if1];
  }

  //Compare with input and check for >5-sigma deviations
  for(l=0;l<lmax;l++) {
    for(im1=0;im1<nmaps;im1++) {
      for(im2=0;im2<nmaps;im2++) {
	double c11=cells_in[im1+nmaps*im1][l];
	double c12=cells_in[im2+nmaps*im1][l];
	double c21=cells_in[im1+nmaps*im2][l];
	double c22=cells_in[im2+nmaps*im2][l];
	double sig=sqrt((c11*c22+c12*c21)/(2.*l+1.));
	double diff=fabs(cells_out[im2+nmaps*im1][l]-c12);
	//Check that there are no >5-sigma fluctuations around input power spectrum
	ASSERT_TRUE((int)(diff<5*sig));
      }
    }
  }
  
  for(im1=0;im1<nmaps;im1++)
    free(maps[im1]);
  free(maps);
  
  for(ii=0;ii<nfields;ii++)
    free(beam[ii]);
  free(beam);
  for(ii=0;ii<ncls;ii++) {
    free(cells_in[ii]);
    free(cells_out[ii]);
  }
  free(cells_in);
  free(cells_out);
  free(cells_pass);
  free(cs);
}
