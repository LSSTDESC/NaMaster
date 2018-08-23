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

void time_stuff(timing_st *tim,int ncomp,FILE *f)
{
  timing_st_time(tim,ncomp);
  timing_st_report(tim,f);
  timing_st_free(tim);
}

int main(int argc,char **argv)
{
  FILE *fout;
  char fname_out[256]="none";
  int ncomp=10,nside=256;
  int do_field=1,do_mcm=1,do_pure=1;
  char **i;
  for(i=argv+1;*i;i++) {
    if(!strcmp(*i,"-ncomp")) ncomp=atoi(*++i);
    else if(!strcmp(*i,"-nside")) nside=atoi(*++i);
    else if(!strcmp(*i,"-out")) sprintf(fname_out,"%s",*++i);
    else if(!strcmp(*i,"-dont_field")) do_field=0;
    else if(!strcmp(*i,"-dont_mcm")) do_mcm=0;
    else if(!strcmp(*i,"-dont_pure")) do_pure=0;
  }

  if(!strcmp(fname_out,"none"))
    fout=stdout;
  else
    fout=my_fopen(fname_out,"w");
  
  setbuf(stdout,NULL);

  timing_st *tim;
  
  if(do_field) {
    tim=timing_st_init("field     ",nside,1,setup_field,func_field,free_field);
    time_stuff(tim,ncomp,fout);
  }
  if(do_mcm) {
    tim=timing_st_init("mcm       ",nside,0,setup_mcm,func_mcm,free_mcm);
    time_stuff(tim,ncomp,fout);
  }
  if(do_pure) {
    tim=timing_st_init("pure      ",nside,1,setup_pure,func_pure,free_pure);
    time_stuff(tim,ncomp,fout);
  }

  if(strcmp(fname_out,"none"))
    fclose(fout);
  
  return 0;
}
