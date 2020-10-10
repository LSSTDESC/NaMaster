#include "namaster.h"
#include "ctest.h"
#include "utils.h"
#include "nmt_test_utils.h"

CTEST(nmt,curvedsky_errors)
{
  int lmax,ii;
  nmt_curvedsky_info *cs;
  long nside=128;
  int nx=128,ny=128;
  double dth=M_PI/(3*nside),dph=M_PI/(3*nside);
  nmt_curvedsky_info *cs_hpx_ref=nmt_curvedsky_info_alloc(1,nside,-1,-1,-1,-1,-1,-1,-1);
  nmt_curvedsky_info *cs_car_ref=nmt_curvedsky_info_alloc(0,-1,-1,nx,ny,dth,dph,0,M_PI);

  set_error_policy(THROW_ON_ERROR);
  cs=NULL;
  try { cs=nmt_curvedsky_info_alloc(1,nside-1,-1,-1,-1,-1,-1,-1,-1); } //Passing incorrect nside
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(cs);
  try { cs=nmt_curvedsky_info_alloc(0,-1,-1,-nx,ny,dth,dph,0,M_PI/2); } //Passing incorrect CAR dimensions
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(cs);
  try { cs=nmt_curvedsky_info_alloc(0,-1,-1,nx,-ny,dth,dph,0,M_PI/2); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(cs);
  try { cs=nmt_curvedsky_info_alloc(0,-1,-1,nx,ny,-dth,dph,0,M_PI/2); } //Wrong pixel sizes
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(cs);
  try { cs=nmt_curvedsky_info_alloc(0,-1,-1,nx,ny,dth,-dph,0,M_PI/2); } //Wrong pixel sizes
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(cs);
  try { cs=nmt_curvedsky_info_alloc(0,-1,-1,nx,ny,dth,dph,4*M_PI,M_PI/2); } //Wrong dimensions
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(cs);
  try { cs=nmt_curvedsky_info_alloc(0,-1,-1,nx,ny,dth,dph,0,-0.1); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(cs);
  try { nmt_curvedsky_info_alloc(0,-1,-1,nx,ny,dth,2*M_PI/(6*nside+0.5),0,M_PI); } //Pixel size is not CC-compliant
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(cs);
  try { nmt_curvedsky_info_alloc(0,-1,-1,nx,ny,M_PI/(3*nside+0.5),dph,0,M_PI); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  ASSERT_NULL(cs);
  set_error_policy(EXIT_ON_ERROR);
  
  //Check lmax calculation
  ASSERT_TRUE(he_get_lmax(cs_hpx_ref)==he_get_lmax(cs_car_ref)-1);

  //Compare infos
  ASSERT_FALSE(nmt_diff_curvedsky_info(cs_hpx_ref,cs_car_ref)); //HPX vs CAR
  cs=nmt_curvedsky_info_alloc(0,-1,-1,nx,ny,dth,dph*2,0,M_PI/2);
  ASSERT_FALSE(nmt_diff_curvedsky_info(cs,cs_car_ref)); //Different pixel sizes
  free(cs);
  cs=nmt_curvedsky_info_alloc(0,-1,-1,nx,ny,dth,dph*(1+5E-10),0,M_PI); //Very similar pixel sizes
  ASSERT_TRUE(nmt_diff_curvedsky_info(cs,cs_car_ref));
  free(cs);

  //Check filling out of azimuth direction (nx = 360/dphi = nside * 360/120 = 3*nside)
  ASSERT_TRUE(cs_car_ref->nx_short==nx);
  ASSERT_TRUE(cs_car_ref->nx==6*nside);
  
  //Extend a map and check it stays the same
  flouble *map_in=my_malloc(nx*ny*sizeof(flouble));
  for(ii=0;ii<nx*ny;ii++)
    map_in[ii]=ii+0.123;
  flouble *map_out=nmt_extend_CAR_map(cs_car_ref,map_in);
  
  for(ii=0;ii<cs_car_ref->nx;ii++) {
    int jj;
    for(jj=0;jj<cs_car_ref->ny;jj++) {
      if(ii>=cs_car_ref->nx_short)
	ASSERT_TRUE(map_out[ii+cs_car_ref->nx*jj]==0);
      else
	ASSERT_TRUE(map_out[ii+cs_car_ref->nx*jj]==map_in[ii+cs_car_ref->nx_short*jj]);
    }
  }
  free(map_in); free(map_out);

  free(cs_car_ref);
  free(cs_hpx_ref);
}

CTEST(nmt,field_car_lite) {
  nmt_field *f;
  int ii,nmaps;
  double ntemp=5;
  int ny=384,nx=2*(ny-1);
  double dtheta=M_PI/(ny-1),dphi=dtheta;
  nmt_curvedsky_info *cs=nmt_curvedsky_info_alloc(0,-1,-1,nx,ny,dtheta,dphi,0.,M_PI);
  long lmax=he_get_lmax(cs);
  long npix_short=nx*ny;
  double **maps;
  double ***temp=my_malloc(ntemp*sizeof(double **));
  double *beam=my_malloc((lmax+1)*sizeof(double));
  double *mask=my_malloc(npix_short*sizeof(double));

  for(ii=0;ii<npix_short;ii++)
    mask[ii]=1.;
  
  for(ii=0;ii<=lmax;ii++)
    beam[ii]=1.;

  ////////
  //Spin-2
  nmaps=2;

  //With purification (nothing should change)
  maps=test_make_map_analytic_car(cs,2);
  f=nmt_field_alloc_sph(cs,mask,2,maps,0,NULL,beam,1,1,5,1E-5,HE_NITER_DEFAULT,0,1,0);
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
  for(ii=0;ii<nmaps;ii++)
    free(maps[ii]);
  free(maps);

  //With templates
  maps=test_make_map_analytic_car(cs,2);
  for(ii=0;ii<ntemp;ii++)
    temp[ii]=test_make_map_analytic_car(cs,2);
  f=nmt_field_alloc_sph(cs,mask,2,maps,ntemp,temp,beam,0,0,0,1E-5,HE_NITER_DEFAULT,0,1,0);
  //Since maps and templates are the same, template-deprojected alms should be 0
  ASSERT_DBL_NEAR_TOL(0.,creal(f->alms[0][he_indexlm(2,0,lmax)]),1E-5);
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

CTEST(nmt,field_car_alloc) {
  nmt_field *f;
  int ii,nmaps;
  double ntemp=5;
  int ny=384,nx=2*(ny-1);
  double dtheta=M_PI/(ny-1),dphi=dtheta;
  nmt_curvedsky_info *cs=nmt_curvedsky_info_alloc(0,-1,-1,nx,ny,dtheta,dphi,0.,M_PI);
  long lmax=he_get_lmax(cs);
  long npix_short=nx*ny;
  double **maps;
  double ***temp=my_malloc(ntemp*sizeof(double **));
  double *beam=my_malloc((lmax+1)*sizeof(double));
  double *mask=my_malloc(npix_short*sizeof(double));

  for(ii=0;ii<npix_short;ii++)
    mask[ii]=1.;
  
  for(ii=0;ii<=lmax;ii++)
    beam[ii]=1.;
  
  ////////
  //Spin-0
  nmaps=1;
  //Create inputs
  maps=test_make_map_analytic_car(cs,0);
  //No templates
  f=nmt_field_alloc_sph(cs,mask,0,maps,0,NULL,beam,0,0,0,1E-5,HE_NITER_DEFAULT,0,0,0);
  //Sanity checks
  ASSERT_EQUAL(lmax,f->lmax);
  ASSERT_EQUAL(0,f->pure_e);
  ASSERT_EQUAL(0,f->pure_b);
  ASSERT_EQUAL(cs->nx*cs->ny,f->cs->npix);
  ASSERT_EQUAL((ny-1)/2,f->cs->n_eq);
  ASSERT_EQUAL(0,f->spin);
  ASSERT_EQUAL(1,f->nmaps);
  //Harmonic transform
  ASSERT_DBL_NEAR_TOL(0.5,creal(f->alms[0][he_indexlm(2,2,lmax)]),1E-5);
  ASSERT_DBL_NEAR_TOL(0.0,cimag(f->alms[0][he_indexlm(2,2,lmax)]),1E-5);
  nmt_field_free(f);
  for(ii=0;ii<nmaps;ii++)
    free(maps[ii]);
  free(maps);
  
  //With templates
  maps=test_make_map_analytic_car(cs,0);
  for(ii=0;ii<ntemp;ii++)
    temp[ii]=test_make_map_analytic_car(cs,0);
  f=nmt_field_alloc_sph(cs,mask,0,maps,ntemp,temp,NULL,0,0,0,1E-5,HE_NITER_DEFAULT,0,0,0);
  //Since maps and templates are the same, template-deprojected map should be 0
  for(ii=0;ii<cs->npix;ii++)
    ASSERT_DBL_NEAR_TOL(0.0,f->maps[0][ii],1E-10);
  for(ii=0;ii<ntemp;ii++) {
    ASSERT_DBL_NEAR_TOL(0.5,creal(f->a_temp[ii][0][he_indexlm(2,2,lmax)]),1E-5);
    ASSERT_DBL_NEAR_TOL(0.0,cimag(f->a_temp[ii][0][he_indexlm(2,2,lmax)]),1E-5);
  }
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
  //Spin-1
  nmaps=2;

  //No templates
  maps=test_make_map_analytic_car(cs,1);
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
  //Spin-2
  nmaps=2;

  //No templates
  maps=test_make_map_analytic_car(cs,2);
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
  for(ii=0;ii<nmaps;ii++)
    free(maps[ii]);
  free(maps);

  //With purification (nothing should change)
  maps=test_make_map_analytic_car(cs,2);
  f=nmt_field_alloc_sph(cs,mask,2,maps,0,NULL,beam,1,1,5,1E-5,HE_NITER_DEFAULT,0,0,0);
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
  for(ii=0;ii<nmaps;ii++)
    free(maps[ii]);
  free(maps);
  
  //With templates
  maps=test_make_map_analytic_car(cs,2);
  for(ii=0;ii<ntemp;ii++)
    temp[ii]=test_make_map_analytic_car(cs,2);
  f=nmt_field_alloc_sph(cs,mask,2,maps,ntemp,temp,beam,0,0,0,1E-5,HE_NITER_DEFAULT,0,0,0);
  //Since maps and templates are the same, template-deprojected map should be 0
  for(ii=0;ii<nmaps;ii++) {
    int jj;
    for(jj=0;jj<cs->npix;jj++)
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
  for(ii=0;ii<ntemp;ii++) {
    int jj;
    for(jj=0;jj<nmaps;jj++)
      free(temp[ii][jj]);
    free(temp[ii]);
  }
  for(ii=0;ii<nmaps;ii++)
    free(maps[ii]);
  free(maps);
  
  //With templates and purification (nothing should change)
  maps=test_make_map_analytic_car(cs,2);
  for(ii=0;ii<ntemp;ii++)
    temp[ii]=test_make_map_analytic_car(cs,2);
  f=nmt_field_alloc_sph(cs,mask,2,maps,ntemp,temp,beam,1,1,5,1E-5,HE_NITER_DEFAULT,0,0,0);
  //Since maps and templates are the same, template-deprojected map should be 0
  for(ii=0;ii<nmaps;ii++) {
    int jj;
    for(jj=0;jj<cs->npix;jj++)
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

CTEST(nmt,field_car_synfast) {
  int ii,im1,im2,l,if1,if2;
  int ny=384,nx=2*(ny-1);
  double dtheta=M_PI/(ny-1),dphi=dtheta;
  nmt_curvedsky_info *cs=nmt_curvedsky_info_alloc(0,-1,-1,nx,ny,dtheta,dphi,0.,M_PI);
  long lmax=he_get_lmax(cs);
  int nfields=3;
  int field_spins[3]={0,2,0};
  int field_nmaps[3]={1,2,1};
  int nmaps=4;
  int ncls_pass=(nmaps*(nmaps+1))/2;
  int ncls=nmaps*nmaps;
  double lpivot=ny/6.;
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
