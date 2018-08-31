#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <namaster.h>
#include <omp.h>
#include "utils.h"
#include "timing_utils.h"
#include "timing_field.h"

//Setup function for field timing
// - Initializes a set of maps and a mask
void *setup_field(int nside,int spin1,int spin2)
{
  field_data *data=my_malloc(sizeof(field_data));
  data->nside=nside;
  if(spin1) {
    data->nmaps=2;
    data->pol=1;
  }
  else {
    data->nmaps=1;
    data->pol=0;
  }

  data->maps=init_maps(nside,data->nmaps);
  data->mask=init_mask(nside);

  return data;
}

//Evaluator for field timing
// - Generates and frees an nmt_field object
void func_field(void *data)
{
  field_data *d=(field_data *)data;
  nmt_field *f=nmt_field_alloc_sph(d->nside,d->mask,d->pol,d->maps,0,NULL,NULL,0,0,3,1E-5);
  nmt_field_free(f);
}

//Destructor for field timing
void free_field(void *data)
{
  field_data *d=(field_data *)data;
  int ii;
  for(ii=0;ii<d->nmaps;ii++)
    free(d->maps[ii]);
  free(d->maps);
  free(d->mask);
  free(data);
}
