#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <namaster.h>
#include <omp.h>
#include "utils.h"
#include "timing_utils.h"
#include "timing_pure_deproj.h"

//Setup function for pure_deproj timing
// - Initializes a set of maps and a mask
void *setup_pure_deproj(int nside,int spin1,int spin2)
{
  int ii;
  pure_deproj_data *data=my_malloc(sizeof(pure_deproj_data));
  data->nside=nside;

  data->maps=init_maps(nside,2);
  data->mask=init_mask(nside);
  data->temp=my_malloc(NTEMP*sizeof(double **));
  for(ii=0;ii<NTEMP;ii++)
    data->temp[ii]=init_maps(nside,2);

  return data;
}

//Evaluator for pure_deproj timing
// - Generates and frees an nmt_pure_deproj object
void func_pure_deproj(void *data)
{
  pure_deproj_data *d=(pure_deproj_data *)data;
  nmt_field *f=nmt_field_alloc_sph(d->nside,d->mask,1,d->maps,NTEMP,d->temp,NULL,0,1,3,1E-5);
  nmt_field_free(f);
}

//Destructor for pure_deproj timing
void free_pure_deproj(void *data)
{
  pure_deproj_data *d=(pure_deproj_data *)data;
  int ii;
  for(ii=0;ii<NTEMP;ii++) {
    int jj;
    for(jj=0;jj<2;jj++)
      free(d->temp[ii][jj]);
    free(d->temp[ii]);
  }
  free(d->temp);
  for(ii=0;ii<2;ii++)
    free(d->maps[ii]);
  free(d->maps);
  free(d->mask);
  free(data);
}
