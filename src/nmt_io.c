#include "config.h"
#include "utils.h"
#include <fitsio.h>

static void nmt_curvedsky_info_tohdus(fitsfile *fptr,
				      nmt_curvedsky_info *cs,
				      int *status)
{
  fits_create_img(fptr,BYTE_IMG,0,NULL,status);
  fits_write_key(fptr,TINT   ,"IS_HEALPIX",&(cs->is_healpix),NULL,status);
  fits_write_key(fptr,TLONG  ,"N_EQ",&(cs->n_eq),NULL,status);
  fits_write_key(fptr,TINT   ,"LMAX_SHT",&(cs->lmax_sht),NULL,status);
  fits_write_key(fptr,TINT   ,"NX_SHORT",&(cs->nx_short),NULL,status);
  fits_write_key(fptr,TINT   ,"NX",&(cs->nx),NULL,status);
  fits_write_key(fptr,TINT   ,"NY",&(cs->ny),NULL,status);
  fits_write_key(fptr,TLONG  ,"NPIX",&(cs->npix),NULL,status);
  fits_write_key(fptr,TDOUBLE,"DELTA_THETA",&(cs->Delta_theta),NULL,status);
  fits_write_key(fptr,TDOUBLE,"DELTA_PHI",&(cs->Delta_phi),NULL,status);
  fits_write_key(fptr,TDOUBLE,"PHI0",&(cs->phi0),NULL,status);
  fits_write_key(fptr,TDOUBLE,"THETA0",&(cs->theta0),NULL,status);
}

static nmt_curvedsky_info *nmt_curvedsky_info_fromhdus(fitsfile *fptr,
						       int *status)
{
  nmt_curvedsky_info *cs=my_malloc(sizeof(nmt_curvedsky_info));

  fits_movrel_hdu(fptr,1,NULL,status);
  fits_read_key(fptr,TINT   ,"IS_HEALPIX",&(cs->is_healpix),NULL,status);
  fits_read_key(fptr,TLONG  ,"N_EQ",&(cs->n_eq),NULL,status);
  fits_read_key(fptr,TINT   ,"LMAX_SHT",&(cs->lmax_sht),NULL,status);
  fits_read_key(fptr,TINT   ,"NX_SHORT",&(cs->nx_short),NULL,status);
  fits_read_key(fptr,TINT   ,"NX",&(cs->nx),NULL,status);
  fits_read_key(fptr,TINT   ,"NY",&(cs->ny),NULL,status);
  fits_read_key(fptr,TLONG  ,"NPIX",&(cs->npix),NULL,status);
  fits_read_key(fptr,TDOUBLE,"DELTA_THETA",&(cs->Delta_theta),NULL,status);
  fits_read_key(fptr,TDOUBLE,"DELTA_PHI",&(cs->Delta_phi),NULL,status);
  fits_read_key(fptr,TDOUBLE,"PHI0",&(cs->phi0),NULL,status);
  fits_read_key(fptr,TDOUBLE,"THETA0",&(cs->theta0),NULL,status);

  return cs;
}

static void nmt_workspace_info_tohdus(fitsfile *fptr,
				      nmt_workspace *w,
				      int *status)
{
  fits_create_img(fptr,BYTE_IMG,0,NULL,status);
  fits_write_key(fptr,TINT,"LMAX",&(w->lmax),NULL,status);
  fits_write_key(fptr,TINT,"LMAX_FIELDS",&(w->lmax_fields),NULL,status);
  fits_write_key(fptr,TINT,"LMAX_MASK",&(w->lmax_mask),NULL,status);
  fits_write_key(fptr,TINT,"IS_TEB",&(w->is_teb),NULL,status);
  fits_write_key(fptr,TINT,"NCLS",&(w->ncls),NULL,status);
}

static void nmt_workspace_info_fromhdus(fitsfile *fptr,
					nmt_workspace *w,
					int *status)
{
  fits_read_key(fptr,TINT,"LMAX",&(w->lmax),NULL,status);
  fits_read_key(fptr,TINT,"LMAX_FIELDS",&(w->lmax_fields),NULL,status);
  fits_read_key(fptr,TINT,"LMAX_MASK",&(w->lmax_mask),NULL,status);
  fits_read_key(fptr,TINT,"IS_TEB",&(w->is_teb),NULL,status);
  fits_read_key(fptr,TINT,"NCLS",&(w->ncls),NULL,status);
}

static void nmt_l_arr_to_hdus(fitsfile *fptr,
			      int lmax, double *fl,char *arrname,
			      int *status)
{
  int ii;
  int *larr=my_malloc((lmax+1)*sizeof(int));
  for(ii=0;ii<=lmax;ii++)
    larr[ii]=ii;

  char **ttype,**tform,**tunit;
  ttype=my_malloc(2*sizeof(char *));
  ttype[0]=my_malloc(256); sprintf(ttype[0],"L");
  ttype[1]=my_malloc(256); sprintf(ttype[1],"%s",arrname);
  tform=my_malloc(2*sizeof(char *));
  tform[0]=my_malloc(256); sprintf(tform[0],"1J");
  tform[1]=my_malloc(256); sprintf(tform[1],"1D");
  tunit=my_malloc(2*sizeof(char *));
  tunit[0]=my_malloc(256); sprintf(tunit[0]," ");
  tunit[1]=my_malloc(256); sprintf(tunit[1]," ");

  fits_create_tbl(fptr,BINARY_TBL,0,2,ttype,tform,tunit,arrname,status);
  fits_write_col(fptr,TINT,1,1,1,lmax+1,larr,status);
  fits_write_col(fptr,TDOUBLE,2,1,1,lmax+1,fl,status);

  free(larr);
  for(ii=0;ii<2;ii++) {
    free(ttype[ii]);
    free(tform[ii]);
    free(tunit[ii]);
  }
  free(ttype);
  free(tform);
  free(tunit);
}

static double *nmt_l_arr_from_hdus(fitsfile *fptr, int lmax_expect, int *status)
{
  double *f;
  double nulval;
  int anynul;
  long nrows;
  *status=0;
  fits_movrel_hdu(fptr,1,NULL,status);
  fits_get_num_rows(fptr,&nrows,status);
  if(nrows!=lmax_expect+1) {
    report_error(NMT_ERROR_INCONSISTENT,
		 "Expected %d elements, but got %ld\n",
		 lmax_expect+1,nrows);
  }
  f=my_malloc(nrows*sizeof(double));
  fits_read_col(fptr,TDOUBLE,2,1,1,nrows,&nulval,f,&anynul,status);
  return f;
}

void nmt_workspace_write_fits(nmt_workspace *w,char *fname)
{
  fitsfile *fptr;
  int status=0;
  fits_create_file(&fptr,fname,&status);
  // Workspace info HDU
  nmt_workspace_info_tohdus(fptr,w,&status);
  // CS info HDU
  nmt_curvedsky_info_tohdus(fptr,w->cs,&status);
  // beam_prod HDU
  nmt_l_arr_to_hdus(fptr,w->lmax_fields,w->beam_prod,"BEAMS",&status);
  // pcl_masks HDU
  nmt_l_arr_to_hdus(fptr,w->lmax_mask,w->pcl_masks,"PCL_MASKS",&status);
  fits_close_file(fptr,&status);
}

nmt_workspace *nmt_workspace_read_fits(char *fname)
{
  fitsfile *fptr;
  int status=0;
  nmt_workspace *w=my_malloc(sizeof(nmt_workspace));

  fits_open_file(&fptr,fname,READONLY,&status);
  // Workspace info HDU
  nmt_workspace_info_fromhdus(fptr,w,&status);
  // CS info HDU
  w->cs=nmt_curvedsky_info_fromhdus(fptr,&status);
  // beam_prod HDU
  w->beam_prod=nmt_l_arr_from_hdus(fptr,w->lmax_fields,&status);
  // pcl_masks HDU
  w->pcl_masks=nmt_l_arr_from_hdus(fptr,w->lmax_mask,&status);
  fits_close_file(fptr,&status);

  return w;
}
