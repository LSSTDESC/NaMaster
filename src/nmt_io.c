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

  fits_movrel_hdu(fptr,1,IMAGE_HDU,status);
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
  fits_write_key(fptr,TINT,"IS_TEB",&(w->is_teb),NULL,status);
  fits_write_key(fptr,TINT,"NCLS",&(w->ncls),NULL,status);
}

static void nmt_workspace_info_fromhdus(fitsfile *fptr,
					nmt_workspace *w,
					int *status)
{
  fits_read_key(fptr,TINT,"LMAX",&(w->lmax),NULL,status);
  fits_read_key(fptr,TINT,"IS_TEB",&(w->is_teb),NULL,status);
  fits_read_key(fptr,TINT,"NCLS",&(w->ncls),NULL,status);
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
  fits_close_file(fptr,&status);

  return w;
}
