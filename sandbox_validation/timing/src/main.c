#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <namaster.h>
#include <omp.h>
#include "utils.h"
#include "timing_utils.h"
#include "timing_field.h"
#include "timing_mcm.h"
#include "timing_pure.h"
#include "timing_deproj.h"
#include "timing_pure_deproj.h"

void time_stuff(timing_st *tim,int ncomp,FILE *f)
{
  timing_st_time(tim,ncomp);
  timing_st_report(tim,f);
  timing_st_free(tim);
}

void help_and_exit()
{
  fprintf(stderr,"Usage: ./timings <options>\n");
  fprintf(stderr,"Available options:\n");
  fprintf(stderr,"  -nside INT\n");
  fprintf(stderr,"  -ncomp INT\n");
  fprintf(stderr,"  -out STRING\n");
  fprintf(stderr,"  -do_field\n");
  fprintf(stderr,"  -do_mcm\n");
  fprintf(stderr,"  -do_pure\n");
  fprintf(stderr,"  -do_deproj\n");
  fprintf(stderr,"  -do_pure_deproj\n");
  fprintf(stderr,"  -h\n\n");
  exit(1);
}

int main(int argc,char **argv)
{
  FILE *fout;
  char fname_out[256]="none";
  int ncomp=10,nside=256;
  int do_field=0,do_mcm=0,do_pure=0,do_deproj=0,do_pure_deproj=0;
  char **i;
  for(i=argv+1;*i;i++) {
    if(!strcmp(*i,"-ncomp")) ncomp=atoi(*++i);
    else if(!strcmp(*i,"-nside")) nside=atoi(*++i);
    else if(!strcmp(*i,"-out")) sprintf(fname_out,"%s",*++i);
    else if(!strcmp(*i,"-do_field")) do_field=1;
    else if(!strcmp(*i,"-do_mcm")) do_mcm=1;
    else if(!strcmp(*i,"-do_pure")) do_pure=1;
    else if(!strcmp(*i,"-do_deproj")) do_deproj=1;
    else if(!strcmp(*i,"-do_pure_deproj")) do_pure_deproj=1;
    else if(!strcmp(*i,"-h")) help_and_exit();
    else
      help_and_exit();
  }

  if(!strcmp(fname_out,"none"))
    fout=stdout;
  else
    fout=my_fopen(fname_out,"w");
  
  setbuf(stdout,NULL);

  timing_st *tim;
  
  if(do_field) {
    tim=timing_st_init("field      ",nside,2,setup_field,func_field,free_field);
    time_stuff(tim,ncomp,fout);
  }
  if(do_mcm) {
    tim=timing_st_init("mcm        ",nside,3,setup_mcm,func_mcm,free_mcm);
    time_stuff(tim,ncomp,fout);
  }
  if(do_pure) {
    tim=timing_st_init("pure       ",nside,1,setup_pure,func_pure,free_pure);
    time_stuff(tim,ncomp,fout);
  }
  if(do_deproj) {
    tim=timing_st_init("deproj     ",nside,2,setup_deproj,func_deproj,free_deproj);
    time_stuff(tim,ncomp,fout);
  }
  if(do_pure_deproj) {
    tim=timing_st_init("pure_deproj",nside,1,setup_pure_deproj,func_pure_deproj,free_pure_deproj);
    time_stuff(tim,ncomp,fout);
  }

  if(strcmp(fname_out,"none"))
    fclose(fout);
  
  return 0;
}
