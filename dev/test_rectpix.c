
#include <stdlib.h>
#include <stdio.h>
#include <namaster.h>
#include <math.h>
#include "utils.h"
#include <fitsio.h>

int main(int argc,char **argv)
{
  printf("Testing map2alm.\n");
  const int spin = 0;
  const int alm_ind = 5000; // which alm to check
  const int lmax = 1000;
  int i;
  long int nside;
  nmt_curvedsky_info sky_info;
  flouble *CAR_map, *HEAL_map;
  fcomplex **alms = (fcomplex**) malloc(1 * sizeof(fcomplex *));
  flouble **maps = (flouble **) my_malloc(1 * sizeof(flouble *));
  nmt_curvedsky_info *cs=my_malloc(sizeof(nmt_curvedsky_info));

  cs->is_healpix = 0;
  CAR_map = he_read_map("cosmojpg_car.fits", cs, 0);
  maps[0] = CAR_map;
  alms[0] = (fcomplex*) malloc(he_nalms(lmax) * sizeof(fcomplex));
  he_map2alm(cs, lmax, 1, spin, maps, alms, 1);
  printf("CAR  %e + %ei\n", creal(alms[0][alm_ind]), cimag(alms[0][alm_ind]));
  cs->is_healpix = 1;
  HEAL_map = he_read_map("cosmojpg_heal.fits", cs, 0);
  maps[0] = HEAL_map;
  alms[0] = (fcomplex*) malloc(he_nalms(lmax) * sizeof(fcomplex));
  he_map2alm(cs, lmax, 1, spin, maps, alms, 1);
  printf("HPX  %e + %ei\n", creal(alms[0][alm_ind]), cimag(alms[0][alm_ind]));

  return 0;
}
