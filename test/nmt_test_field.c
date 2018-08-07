#include "namaster.h"
#include "ctest.h"
#include "utils.h"
#include "nmt_test_utils.h"

CTEST(nmt,field_create_sp0) {
  int i;
  int spins[1]={0};
  int *ell=test_get_sequence(0,3*NSIDE_TESTS);
  double *cl=malloc(3*NSIDE_TESTS*sizeof(double));
  double *bm=malloc(3*NSIDE_TESTS*sizeof(double));
  for(i=0;i<3*NSIDE_TESTS;i++) {
    cl[i]=pow((ell[i]+1.)/101.,-1.2);
    bm[i]=1.;
  }
  double **map=nmt_synfast_sph(NSIDE_TESTS,1,spins,3*NSIDE_TESTS-1,&cl,&bm,1234);
  double *mask=malloc(12*NSIDE_TESTS*NSIDE_TESTS*sizeof(double));
  for(i=0;i<12*NSIDE_TESTS*NSIDE_TESTS;i++)
    mask[i]=1.;
  nmt_field *fld=nmt_field_alloc_sph(NSIDE_TESTS,mask,0,map,0,NULL,NULL,0,0,3,1E-10);
  ASSERT_EQUAL(fld->nside,NSIDE_TESTS);
  ASSERT_EQUAL(fld->npix,12*NSIDE_TESTS*NSIDE_TESTS);
  nmt_field_free(fld);
  free(mask);
  free(map[0]);
  free(map);
  free(ell);
  free(cl);
}

CTEST(nmt,field_read_sp0) {
  nmt_field *fld=nmt_field_read("test/mask.fits","test/maps.fits","none","none",0,0,0,3,1E-10);
  ASSERT_EQUAL(fld->nside,256);
  nmt_field_free(fld);
}
