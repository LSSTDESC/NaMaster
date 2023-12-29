#include "namaster.h"
#include "ctest.h"
#include "utils.h"
#include "nmt_test_utils.h"
#include <chealpix.h>

double **test_make_map_analytic_flat(nmt_flatsky_info *fsk,int spin,int i0_x,int i0_y)
{
  int ii;
  double k0_x=i0_x*2*M_PI/fsk->lx;
  double k0_y=i0_y*2*M_PI/fsk->ly;
  double cphi0=k0_x/sqrt(k0_x*k0_x+k0_y*k0_y);
  double sphi0=k0_y/sqrt(k0_x*k0_x+k0_y*k0_y);
  double c2phi0=cphi0*cphi0-sphi0*sphi0;
  double s2phi0=2*sphi0*cphi0;
  int nmaps=1;
  if(spin) nmaps=2;
  
  double **maps=my_malloc(nmaps*sizeof(double *));
  for(ii=0;ii<nmaps;ii++)
    maps[ii]=dftw_malloc(fsk->npix*sizeof(double));
  
  for(ii=0;ii<fsk->ny;ii++) {
    int jj;
    double y=ii*fsk->ly/fsk->ny;
    for(jj=0;jj<fsk->nx;jj++) {
      double x=jj*fsk->lx/fsk->nx;
      double phase=k0_x*x+k0_y*y;
      if(spin==2) {
	maps[0][jj+fsk->nx*ii]= 2*M_PI*c2phi0*cos(phase)/(fsk->lx*fsk->ly);
	maps[1][jj+fsk->nx*ii]=-2*M_PI*s2phi0*cos(phase)/(fsk->lx*fsk->ly);
      }
      else if(spin==1) {
	maps[0][jj+fsk->nx*ii]=-2*M_PI*cphi0*sin(phase)/(fsk->lx*fsk->ly);
	maps[1][jj+fsk->nx*ii]= 2*M_PI*sphi0*sin(phase)/(fsk->lx*fsk->ly);
      }
      else 
	maps[0][jj+fsk->nx*ii]=2*M_PI*cos(phase)/(fsk->lx*fsk->ly);
    }
  }

  return maps;
}

int *test_get_sequence(int n0,int nf)
{
  int i;
  int *seq=malloc((nf-n0)*sizeof(int));
  ASSERT_NOT_NULL(seq);
  for(i=0;i<(nf-n0);i++)
    seq[i]=n0+i;
  return seq;
}

void test_compare_arrays(int n,double *y,int narr,int iarr,char *fname,double rtol)
{
  int ii;
  FILE *fi=fopen(fname,"r");
  ASSERT_NOT_NULL(fi);
  for(ii=0;ii<n;ii++) {
    int j;
    double xv,yv,dum;
    int stat=fscanf(fi,"%lf",&xv);
    ASSERT_EQUAL(1,stat);
    for(j=0;j<narr;j++) {
      stat=fscanf(fi,"%lE",&dum);
      ASSERT_EQUAL(1,stat);
      if(j==iarr)
	yv=dum;
    }
    ASSERT_DBL_NEAR_TOL(yv,y[ii],fmin(fabs(yv),fabs(y[ii]))*rtol);
  }
  fclose(fi);
}

CTEST(nmt,ut_drc3jj) {
  int sizew=1000;
  double *w3=my_malloc(sizew*sizeof(double));
  int ii,r,l1min,l1max;

  r=drc3jj(2,3,0,0,&l1min,&l1max,w3,sizew);
  ASSERT_EQUAL(r,0);
  ASSERT_EQUAL(l1max,2+3);
  ASSERT_EQUAL(l1min,1);
  ASSERT_DBL_NEAR_TOL(-sqrt(3./35.),w3[0],1E-10);
  ASSERT_DBL_NEAR_TOL(0,w3[1],1E-10);
  ASSERT_DBL_NEAR_TOL(2/sqrt(105.),w3[2],1E-10);
  ASSERT_DBL_NEAR_TOL(0,w3[3],1E-10);
  ASSERT_DBL_NEAR_TOL(-sqrt(10./231.),w3[4],1E-10);

  r=drc3jj(100,200,2,-2,&l1min,&l1max,w3,sizew);
  ASSERT_EQUAL(r,0);
  ASSERT_EQUAL(l1max,100+200);
  ASSERT_EQUAL(l1min,100);
  ASSERT_DBL_NEAR_TOL(0.0139534,w3[0],1E-5);
  ASSERT_DBL_NEAR_TOL(-0.00192083,w3[30],1E-5);
  ASSERT_DBL_NEAR_TOL(0.000639717,w3[54],1E-5);
  ASSERT_DBL_NEAR_TOL(0.000648742,w3[131],1E-5);

  set_error_policy(THROW_ON_ERROR);

  try { r=drc3jj(100,200,2,-2,&l1min,&l1max,w3,100); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  r=drc3jj(2,3,3,0,&l1min,&l1max,w3,sizew);
  for(ii=0;ii<=l1max-l1min;ii++)
    ASSERT_DBL_NEAR_TOL(w3[ii],0.,1E-10);

  set_error_policy(EXIT_ON_ERROR);
  free(w3);
}

CTEST(nmt,ut_errors) {
  set_error_policy(THROW_ON_ERROR);

  //Wrong allocation params
  try { my_malloc(-1); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);

  try { my_calloc(-1,-1); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);

  set_error_policy(EXIT_ON_ERROR);
}
