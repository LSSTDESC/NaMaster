#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <namaster.h>
#include <omp.h>
#include "utils.h"
#include "timing_utils.h"
#include "timing_pure.h"

//Setup function for pure timing
// - Initializes a set of maps and a mask
void *setup_pure(int nside,int spin1,int spin2)
{
  pure_data *data=my_malloc(sizeof(pure_data));
  data->nside=nside;

  data->maps=init_maps(nside,2);
  data->mask=init_mask(nside);

  return data;
}

//Evaluator for pure timing
// - Generates and frees an nmt_field object
void func_pure(void *data)
{
  pure_data *d=(pure_data *)data;
  nmt_field *f=nmt_field_alloc_sph(d->nside,d->mask,1,d->maps,0,NULL,NULL,0,1,3,1E-5);
  nmt_field_free(f);
}

//Destructor for pure timing
void free_pure(void *data)
{
  pure_data *d=(pure_data *)data;
  int ii;
  for(ii=0;ii<2;ii++)
    free(d->maps[ii]);
  free(d->maps);
  free(d->mask);
  free(data);
}
