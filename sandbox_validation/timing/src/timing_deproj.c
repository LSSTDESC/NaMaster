#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <namaster.h>
#include <omp.h>
#include "utils.h"
#include "timing_utils.h"
#include "timing_deproj.h"

//Setup function for deproj timing
// - Initializes a set of maps and a mask
void *setup_deproj(int nside,int spin1,int spin2)
{
  int ii;

  deproj_data *data=my_malloc(sizeof(deproj_data));
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
  data->temp=my_malloc(NTEMP*sizeof(double **));
  for(ii=0;ii<NTEMP;ii++)
    data->temp[ii]=init_maps(nside,data->nmaps);
  data->mask=init_mask(nside);

  return data;
}

//Evaluator for deproj timing
// - Generates and frees an nmt_deproj object
void func_deproj(void *data)
{
  deproj_data *d=(deproj_data *)data;
  nmt_field *f=nmt_field_alloc_sph(d->nside,d->mask,d->pol,d->maps,NTEMP,d->temp,NULL,0,0,3,1E-5);
  nmt_field_free(f);
}

//Destructor for deproj timing
void free_deproj(void *data)
{
  deproj_data *d=(deproj_data *)data;
  int ii;
  for(ii=0;ii<NTEMP;ii++) {
    int jj;
    for(jj=0;jj<d->nmaps;jj++)
      free(d->temp[ii][jj]);
    free(d->temp[ii]);
  }
  free(d->temp);
  for(ii=0;ii<d->nmaps;ii++)
    free(d->maps[ii]);
  free(d->maps);
  free(d->mask);
  free(data);
}
