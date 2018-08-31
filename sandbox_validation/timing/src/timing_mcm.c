#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <namaster.h>
#include <omp.h>
#include "utils.h"
#include "timing_utils.h"
#include "timing_mcm.h"

//Setup function for field timing
// - Initializes a set of maps and a mask
void *setup_mcm(int nside,int spin1,int spin2)
{
  int ii;
  mcm_data *data=my_malloc(sizeof(mcm_data));
  int nmaps1=1,nmaps2=1,pol1=0,pol2=0;
  if(spin1) {
    nmaps1=2;
    pol1=1;
  }
  if(spin2) {
    nmaps2=2;
    pol2=1;
  }

  double **maps;
  double *mask=init_mask(nside);

  maps=init_maps(nside,nmaps1);
  data->f1=nmt_field_alloc_sph(nside,mask,pol1,maps,0,NULL,NULL,0,0,3,1E-5);
  for(ii=0;ii<nmaps1;ii++)
    free(maps[ii]);
  free(maps);

  maps=init_maps(nside,nmaps2);
  data->f2=nmt_field_alloc_sph(nside,mask,pol2,maps,0,NULL,NULL,0,0,3,1E-5);
  for(ii=0;ii<nmaps2;ii++)
    free(maps[ii]);
  free(maps);

  data->bin=nmt_bins_constant(10,3*nside-1);

  free(mask);

  return data;
}

//Evaluator for mcm timing
// - Generates and frees an nmt_workspace object
void func_mcm(void *data)
{
  mcm_data *d=(mcm_data *)data;
  //  int sig;
  //  gsl_matrix *A=gsl_matrix_alloc(d->bin->n_bands,d->bin->n_bands);
  //  gsl_permutation *p=gsl_permutation_alloc(d->bin->n_bands);
  //  gsl_matrix_set_identity(A);
  //  gsl_linalg_LU_decomp(A,p,&sig);
  //  gsl_matrix_free(A);
  //  gsl_permutation_free(p);
  nmt_workspace *w=nmt_compute_coupling_matrix(d->f1,d->f2,d->bin);
  nmt_workspace_free(w);
}

//Destructor for field timing
void free_mcm(void *data)
{
  mcm_data *d=(mcm_data *)data;
  nmt_field_free(d->f1);
  nmt_field_free(d->f2);
  nmt_bins_free(d->bin);
  free(data);
}

