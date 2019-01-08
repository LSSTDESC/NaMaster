#include "utils.h"

void run_master(nmt_field *fl1,nmt_field *fl2,
		char *fname_cl_noise,
		char *fname_cl_proposal,
		char *fname_coupling,
		char *fname_out,
		char *fname_bins,
		int n_lbin)
{
  FILE *fi;
  int ii;
  int lmax=fl1->lmax;
  int nspec=fl1->nmaps*fl2->nmaps;
  flouble **cl_noise,**cl_proposal,**cl_out,**cl_bias,**cl_data;

  if(fl1->nside!=fl2->nside)
    report_error(NMT_ERROR_CONSISTENT_RESO,"Can't correlate fields with different resolution\n");

  //Binning
  nmt_binning_scheme *bin;
  if(!strcmp(fname_bins,"none"))
    bin=nmt_bins_constant(n_lbin,fl1->lmax);
  else
    bin=nmt_bins_read(fname_bins,fl1->lmax);

  //Allocate cl
  cl_noise=my_malloc(nspec*sizeof(flouble *));
  cl_proposal=my_malloc(nspec*sizeof(flouble *));
  cl_bias=my_malloc(nspec*sizeof(flouble *));
  cl_data=my_malloc(nspec*sizeof(flouble *));
  cl_out=my_malloc(nspec*sizeof(flouble *));
  for(ii=0;ii<nspec;ii++) {
    cl_noise[ii]=my_calloc((lmax+1),sizeof(flouble));
    cl_proposal[ii]=my_calloc((lmax+1),sizeof(flouble));
    cl_bias[ii]=my_calloc((lmax+1),sizeof(flouble));
    cl_data[ii]=my_calloc((lmax+1),sizeof(flouble));
    cl_out[ii]=my_calloc(bin->n_bands,sizeof(flouble));
  }

  printf("Reading noise pseudo-cl\n");
  if(strcmp(fname_cl_noise,"none")) {
    fi=my_fopen(fname_cl_noise,"r");
    int nlin=my_linecount(fi); rewind(fi);
    if(nlin!=lmax+1)
      report_error(NMT_ERROR_READ,"Wrong number of multipoles for noise p.spec.\n");
    for(ii=0;ii<lmax+1;ii++) {
      int status,jj;
      flouble l;
      status=fscanf(fi,"%lf",&l);
      if(status!=1)
	report_error(NMT_ERROR_READ,"Error reading file %s\n",fname_cl_noise);
      for(jj=0;jj<nspec;jj++) {
	status=fscanf(fi,"%lf",&(cl_noise[jj][ii]));
	if(status!=1)
	  report_error(NMT_ERROR_READ,"Error reading file %s\n",fname_cl_noise);
      }
    }
    fclose(fi);
  }

  printf("Reading proposal Cl\n");
  if(strcmp(fname_cl_proposal,"none")) {
    fi=my_fopen(fname_cl_proposal,"r");
    int nlin=my_linecount(fi); rewind(fi);
    if(nlin!=lmax+1)
      report_error(NMT_ERROR_READ,"Wrong number of multipoles for noise p.spec.\n");
    for(ii=0;ii<lmax+1;ii++) {
      int status,jj;
      flouble l;
      status=fscanf(fi,"%lf",&l);
      if(status!=1)
	report_error(NMT_ERROR_READ,"Error reading file %s\n",fname_cl_proposal);
      for(jj=0;jj<nspec;jj++) {
	status=fscanf(fi,"%lf",&(cl_proposal[jj][ii]));
	if(status!=1)
	  report_error(NMT_ERROR_READ,"Error reading file %s\n",fname_cl_proposal);
      }
    }
    fclose(fi);
  }

  nmt_workspace *w;
  if(access(fname_coupling,F_OK)!=-1) { //If file exists just read matrix
    printf("Reading coupling matrix\n");
    w=nmt_workspace_read(fname_coupling);
    if(w->bin->n_bands!=bin->n_bands)
      report_error(NMT_ERROR_CONSISTENT_RESO,"Read coupling matrix doesn't fit input binning scheme\n");
  }
  else {
    printf("Computing coupling matrix \n");
    w=nmt_compute_coupling_matrix(fl1,fl2,bin,0);
    if(strcmp(fname_coupling,"none"))
      nmt_workspace_write(w,fname_coupling);
  }

  printf("Computing data pseudo-Cl\n");
  he_alm2cl(fl1->alms,fl2->alms,fl1->pol,fl2->pol,cl_data,fl1->lmax);

  printf("Computing deprojection bias\n");
  nmt_compute_deprojection_bias(fl1,fl2,cl_proposal,cl_bias);

  printf("Computing decoupled bandpowers\n");
  nmt_decouple_cl_l(w,cl_data,cl_noise,cl_bias,cl_out);

  printf("Writing output\n");
  fi=my_fopen(fname_out,"w");
  for(ii=0;ii<bin->n_bands;ii++) {
    int jj;
    double l_here=0;
    for(jj=0;jj<bin->nell_list[ii];jj++)
      l_here+=bin->ell_list[ii][jj]*bin->w_list[ii][jj];
    fprintf(fi,"%.2lf ",l_here);
    for(jj=0;jj<nspec;jj++)
      fprintf(fi,"%lE ",cl_out[jj][ii]);
    fprintf(fi,"\n");
  }
  fclose(fi);

  nmt_bins_free(bin);
  nmt_workspace_free(w);
  for(ii=0;ii<nspec;ii++) {
    free(cl_noise[ii]);
    free(cl_proposal[ii]);
    free(cl_bias[ii]);
    free(cl_data[ii]);
    free(cl_out[ii]);
  }
  free(cl_proposal);
  free(cl_bias);
  free(cl_data);
  free(cl_noise);
  free(cl_out);
}

int main(int argc,char **argv)
{
  int n_lbin=1,pol_1=0,pol_2=0,is_auto=0,print_help=0;
  int pure_e_1=0,pure_b_1=0,pure_e_2=0,pure_b_2=0;
  char fname_map_1[256]="none";
  char fname_map_2[256]="none";
  char fname_beam_1[256]="none";
  char fname_beam_2[256]="none";
  char fname_mask_1[256]="none";
  char fname_mask_2[256]="none";
  char fname_temp_1[256]="none";
  char fname_temp_2[256]="none";
  char fname_bins[256]="none";
  char fname_cl_noise[256]="none";
  char fname_cl_proposal[256]="none";
  char fname_coupling[256]="none";
  char fname_out[256]="none";
  nmt_field *fl1,*fl2;

  if(argc==1)
    print_help=1;

  char **c;
  for(c=argv+1;*c;c++) {
    if(!strcmp(*c,"-map"))
      sprintf(fname_map_1,"%s",*++c);
    else if(!strcmp(*c,"-map_2"))
      sprintf(fname_map_2,"%s",*++c);
    else if(!strcmp(*c,"-beam"))
      sprintf(fname_beam_1,"%s",*++c);
    else if(!strcmp(*c,"-beam_2"))
      sprintf(fname_beam_2,"%s",*++c);
    else if(!strcmp(*c,"-mask"))
      sprintf(fname_mask_1,"%s",*++c);
    else if(!strcmp(*c,"-mask_2"))
      sprintf(fname_mask_2,"%s",*++c);
    else if(!strcmp(*c,"-temp"))
      sprintf(fname_temp_1,"%s",*++c);
    else if(!strcmp(*c,"-temp_2"))
      sprintf(fname_temp_2,"%s",*++c);
    else if(!strcmp(*c,"-pure_e"))
      pure_e_1=atoi(*++c);
    else if(!strcmp(*c,"-pure_b"))
      pure_b_1=atoi(*++c);
    else if(!strcmp(*c,"-pure_e_2"))
      pure_e_2=atoi(*++c);
    else if(!strcmp(*c,"-pure_b_2"))
      pure_b_2=atoi(*++c);
    else if(!strcmp(*c,"-pol"))
      pol_1=atoi(*++c);
    else if(!strcmp(*c,"-pol_2"))
      pol_2=atoi(*++c);
    else if(!strcmp(*c,"-cl_noise"))
      sprintf(fname_cl_noise,"%s",*++c);
    else if(!strcmp(*c,"-cl_guess"))
      sprintf(fname_cl_proposal,"%s",*++c);
    else if(!strcmp(*c,"-coupling"))
      sprintf(fname_coupling,"%s",*++c);
    else if(!strcmp(*c,"-out"))
      sprintf(fname_out,"%s",*++c);
    else if(!strcmp(*c,"-binning"))
      sprintf(fname_bins,"%s",*++c);
    else if(!strcmp(*c,"-nlb"))
      n_lbin=atoi(*++c);
    else if(!strcmp(*c,"-h"))
      print_help=1;
    else {
      fprintf(stderr,"Unknown option %s\n",*c);
      exit(1);
    }
  }

  if(!print_help) {
    if(!strcmp(fname_map_1,"none")) {
      fprintf(stderr,"Must provide map to correlate!\n");
      print_help=1;
    }
    if(!strcmp(fname_mask_1,"none")) {
      fprintf(stderr,"Must provide mask\n");
      print_help=1;
    }
    if(!strcmp(fname_out,"none")) {
      fprintf(stderr,"Must provide output filename\n");
      print_help=1;
    }
  }

  if(print_help) {
    fprintf(stderr,"Usage: namaster -<opt-name> <option>\n");
    fprintf(stderr,"Options:\n");
    fprintf(stderr,"  -map      -> path to file containing map(s)\n");
    fprintf(stderr,"  -map_2    -> path to file containing 2nd map(s) (optional)\n");
    fprintf(stderr,"  -beam     -> path to file containing SHT of instrument beam for the first field\n");
    fprintf(stderr,"  -beam_2   -> path to file containing 2nd beam (optional)\n");
    fprintf(stderr,"  -mask     -> path to file containing mask\n");
    fprintf(stderr,"  -mask_2   -> path to file containing mask for 2nd map(s) (optional)\n");
    fprintf(stderr,"  -temp     -> path to file containing contaminant templates (optional)\n");
    fprintf(stderr,"  -temp_2   -> path to file containing contaminant templates\n");
    fprintf(stderr,"               for 2nd map(s) (optional)\n");
    fprintf(stderr,"  -pol      -> spin-0 (0) or spin-2 (1) input map(s)\n");
    fprintf(stderr,"  -pol_2    -> spin-0 (0) or spin-2 (1) 2nd input map(s)\n");
    fprintf(stderr,"  -pure_e   -> use pure E-modes for 1st maps? (0-> no or 1-> yes (default->no))\n");
    fprintf(stderr,"  -pure_b   -> use pure B-modes for 1st maps? (0-> no or 1-> yes (default->no))\n");
    fprintf(stderr,"  -pure_e_2 -> use pure E-modes for 2nd maps? (0-> no or 1-> yes (default->no))\n");
    fprintf(stderr,"  -pure_b_2 -> use pure B-modes for 2nd maps? (0-> no or 1-> yes (default->no))\n");
    fprintf(stderr,"  -cl_noise -> path to file containing noise Cl(s)\n");
    fprintf(stderr,"  -cl_guess -> path to file containing initial guess for the Cl(s)\n");
    fprintf(stderr,"  -coupling -> path to file containing coupling matrix (optional)\n");
    fprintf(stderr,"  -out      -> output filename\n");
    fprintf(stderr,"  -binning  -> path to file containing binning scheme\n");
    fprintf(stderr,"  -nlb      -> number of ells per bin (used only if -binning isn't used)\n");
    fprintf(stderr,"  -h        -> this help\n\n");
    return 0;
  }

  if(n_lbin<=0)
    report_error(NMT_ERROR_BADNO,"#ell per bin must be positive\n");

  fl1=nmt_field_read(fname_mask_1,fname_map_1,fname_temp_1,fname_beam_1,pol_1,pure_e_1,pure_b_1,10,1E-10);

  if(!strcmp(fname_map_2,"none")) {
    fl2=fl1;
    is_auto=1;
  }
  else {
    if(!strcmp(fname_mask_2,"none"))
      sprintf(fname_mask_2,"%s",fname_mask_1);
    fl2=nmt_field_read(fname_mask_2,fname_map_2,fname_temp_2,fname_beam_2,pol_2,pure_e_2,pure_b_2,10,1E-10);
  }

  run_master(fl1,fl2,
	     fname_cl_noise,
	     fname_cl_proposal,
	     fname_coupling,
	     fname_out,fname_bins,n_lbin);

  nmt_field_free(fl1);
  if(!is_auto)
    nmt_field_free(fl2);

  return 0;
}
