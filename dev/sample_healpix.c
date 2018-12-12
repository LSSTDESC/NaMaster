#include <stdlib.h>
#include <stdio.h>
#include <namaster.h>
#include <math.h>

// TODO
// make this code work for CAR
// map2alm works, but the utility functions do not
// write a global constant? or put a flag in the nside


int main(int argc,char **argv)
{
  long i;

  char map_name[] = "cosmojpg_heal.fits";
  char mask_name[] = "cosmojpg_healmask.fits";

  //Create spin-0 field
  nmt_field *fl1=nmt_field_read(mask_name, map_name,"none","none",0,0,0,0,
    1e-10);

  //Create a binning scheme (20 multipoles per bandpower)
  nmt_binning_scheme *bin=nmt_bins_constant(20,fl1->lmax);
  //Compute array of effective multipoles
  double *ell_eff=calloc(bin->n_bands,sizeof(double));
  nmt_ell_eff(bin,ell_eff);

  //Allocate memory for power spectrum
  double *cl_out=malloc(bin->n_bands*sizeof(double));

  //Dummy array to be passed as proposal and noise power spectrum
  //These are not needed here, because we don't want to remove noise bias
  //and we have assumed the field is not contaminated
  double *cl_dum=calloc((fl1->lmax+1),sizeof(double));


  //Compute pseudo-Cl estimator
  nmt_workspace *w=nmt_compute_power_spectra(fl1,fl1,bin,NULL,
    &cl_dum,&cl_dum,&cl_out);

  //Write output
  FILE *fo=fopen("sample_output_heal.txt","w");
  for(i=0;i<bin->n_bands;i++)
    fprintf(fo,"%.2lE %lE\n",ell_eff[i],cl_out[i]);
  fclose(fo);

  //Free stuff up
  nmt_workspace_free(w);
  free(cl_dum);
  free(cl_out);
  free(ell_eff);
  nmt_bins_free(bin);
  nmt_field_free(fl1);

  return 0;
}
