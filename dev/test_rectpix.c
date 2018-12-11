#include <stdlib.h>
#include <stdio.h>
#include <namaster.h>
#include <math.h>
#include "utils.h"
#include <fitsio.h>

const int lmax=12000;
// from healpix_extra.c
static void sht_wrapper(int spin,int lmax,int nside,int ntrans,flouble **maps,
  fcomplex **alms,int alm2map);

void test_SHT_wrapper();
void test_map2alm();

int main(int argc,char **argv)
{
  test_map2alm();
  return 0;
}



// copy pasted from healpix_extra.c
static void sht_wrapper(int spin,int lmax,int nside,int ntrans,flouble **maps,fcomplex **alms,int alm2map)
{
  double time=0;
  sharp_alm_info *alm_info;
  sharp_geom_info *geom_info;
#ifdef _SPREC
  int flags=0;
#else //_SPREC
  int flags=SHARP_DP;
#endif //_SPREC

  sharp_make_triangular_alm_info(lmax,lmax,1,&alm_info);
  sharp_make_weighted_healpix_geom_info(nside,1,NULL,&geom_info);
#ifdef _NEW_SHARP
  sharp_execute(alm2map,spin,0,alm,map,geom_info,alm_info,ntrans,flags,0,&time,NULL);
#else //_NEW_SHARP
  sharp_execute(alm2map,spin,alms,maps,geom_info,alm_info,ntrans,flags,&time,NULL);
#endif //_NEW_SHARP
  sharp_destroy_geom_info(geom_info);
  sharp_destroy_alm_info(alm_info);
}


void test_SHT_wrapper()
{
  printf("Testing sht.\n");
  int spin = 0;
  int alm_ind = 10000;

  int i;
  long int nside;
  nmt_curvedsky_info sky_info;
  flouble *CAR_map, *HEAL_map;

  fcomplex **alms = (fcomplex**) malloc(1 * sizeof(fcomplex *));
  flouble **maps = (flouble **) my_malloc(1 * sizeof(flouble *));

// // ---------------------------------------------------------------
  CAR_map = rect_read_CAR_map("cosmojpg_car.fits", &sky_info, 0);
  maps[0] = CAR_map;
  alms[0] = (fcomplex*) malloc(he_nalms(lmax) * sizeof(fcomplex));

  rect_sht_wrapper(spin, lmax, &sky_info, 1, maps, alms, SHARP_MAP2ALM);
  printf("CAR %e + i%e\n", creal(alms[0][alm_ind]), cimag(alms[0][alm_ind]));
// ----------------------------------------------------------------
  HEAL_map = he_read_healpix_map("cosmojpg_heal.fits", &nside, 0);
  maps[0] = HEAL_map;
  alms[0] = (fcomplex*) malloc(he_nalms(lmax) * sizeof(fcomplex));

  sht_wrapper(spin, lmax, nside, 1, maps, alms, SHARP_MAP2ALM);
  printf("hp  %e + i%e\n", creal(alms[0][alm_ind]), cimag(alms[0][alm_ind]));

}


void test_map2alm()
{
  printf("Testing map2alm.\n");
  int spin = 0;
  int alm_ind = 10000;

  int i;
  long int nside;
  nmt_curvedsky_info sky_info;
  flouble *CAR_map, *HEAL_map;

  fcomplex **alms = (fcomplex**) malloc(1 * sizeof(fcomplex *));
  flouble **maps = (flouble **) my_malloc(1 * sizeof(flouble *));

// // ---------------------------------------------------------------
  CAR_map = rect_read_CAR_map("cosmojpg_car.fits", &sky_info, 0);
  maps[0] = CAR_map;
  alms[0] = (fcomplex*) malloc(he_nalms(lmax) * sizeof(fcomplex));

  rect_map2alm(&sky_info, lmax, 1, spin, maps, alms, 1);
  printf("CAR %e + i%e\n", creal(alms[0][alm_ind]), cimag(alms[0][alm_ind]));
// ----------------------------------------------------------------
  HEAL_map = he_read_healpix_map("cosmojpg_heal.fits", &nside, 0);
  maps[0] = HEAL_map;
  alms[0] = (fcomplex*) malloc(he_nalms(lmax) * sizeof(fcomplex));

  he_map2alm(nside, lmax, 1, spin, maps, alms, 1);
  printf("hp  %e + i%e\n", creal(alms[0][alm_ind]), cimag(alms[0][alm_ind]));
}
