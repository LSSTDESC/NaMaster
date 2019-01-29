#include <stdlib.h>
#include <stdio.h>
#include <namaster.h>
#include <math.h>

int main(int argc,char **argv)
{
  long i,nside;
  if(argc!=4) {
    fprintf(stderr,"Usage: ./sample <fname_map> <fname_mask> <fname_out>\n");
    exit(1);
  }

  //Create spin-0 field
  nmt_field *fl1=nmt_field_read(argv[2],argv[1],"none","none",0,0,0,3);

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
  nmt_workspace *w=nmt_compute_power_spectra(fl1,fl1,bin,NULL,&cl_dum,&cl_dum,&cl_out,3);

  //Write output
  FILE *fo=fopen(argv[3],"w");
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
