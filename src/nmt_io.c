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

  //This only means that it had some trouble converting into double precision
  //It can happen if some of these quantities were rubbish to begin with, so
  //we just ignore this one.
  if(*status == 412)
    *status=0;

  return cs;
}

static void nmt_workspace_info_tohdus(fitsfile *fptr,
				      nmt_workspace *w,
				      int *status)
{
  long ii;
  long n_el=w->ncls*(w->lmax+1);
  long naxes[2]={n_el,n_el};
  long fpixel[2]={1,1};
  fits_create_img(fptr,DOUBLE_IMG,2,naxes,status);
  fits_write_key(fptr,TINT,"LMAX",&(w->lmax),NULL,status);
  fits_write_key(fptr,TINT,"LMAX_FIELDS",&(w->lmax_fields),NULL,status);
  fits_write_key(fptr,TINT,"LMAX_MASK",&(w->lmax_mask),NULL,status);
  fits_write_key(fptr,TINT,"IS_TEB",&(w->is_teb),NULL,status);
  fits_write_key(fptr,TINT,"NCLS",&(w->ncls),NULL,status);
  for(ii=0;ii<n_el;ii++) {
    fpixel[1]=ii+1;
    fits_write_pix(fptr,TDOUBLE,fpixel,n_el,w->coupling_matrix_unbinned[ii],status);
  }
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
  long ii;
  long n_el=w->ncls*(w->lmax+1);
  long fpixel[2]={1,1};
  w->coupling_matrix_unbinned=my_malloc(n_el*sizeof(flouble *));
  for(ii=0;ii<n_el;ii++) {
    fpixel[1]=ii+1;
    w->coupling_matrix_unbinned[ii]=my_malloc(n_el*sizeof(flouble));
    fits_read_pix(fptr,TDOUBLE,fpixel,n_el,NULL,
		  w->coupling_matrix_unbinned[ii],NULL,status);
  }
}

static void nmt_l_arr_tohdus(fitsfile *fptr,
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

static double *nmt_l_arr_fromhdus(fitsfile *fptr, int lmax_expect, int *status)
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

static nmt_binning_scheme *nmt_binning_scheme_fromhdus(fitsfile *fptr,
						       int *status)
{
  int ii,anynul;
  long nrows;
  double nulval;
  nmt_binning_scheme *b=my_malloc(sizeof(nmt_binning_scheme));
  fits_movrel_hdu(fptr,1,NULL,status);
  fits_read_key(fptr,TINT,"N_BANDS",&(b->n_bands),NULL,status);
  fits_read_key(fptr,TINT,"ELL_MAX",&(b->ell_max),NULL,status);
  fits_get_num_rows(fptr,&nrows,status);
  if(nrows!=b->n_bands) {
    report_error(NMT_ERROR_INCONSISTENT,
		 "Number of bands doesn't match table size\n");
  }
  b->nell_list=my_malloc(b->n_bands*sizeof(int));
  b->ell_list=my_malloc(b->n_bands*sizeof(int *));
  b->w_list=my_malloc(b->n_bands*sizeof(double *));
  b->f_ell=my_malloc(b->n_bands*sizeof(double *));
  fits_read_col(fptr,TINT,1,1,1,nrows,&nulval,
		b->nell_list,&anynul,status);

  for(ii=0;ii<b->n_bands;ii++) {
    fits_movrel_hdu(fptr,1,NULL,status);
    fits_get_num_rows(fptr,&nrows,status);
    if(nrows!=b->nell_list[ii]) {
      report_error(NMT_ERROR_INCONSISTENT,
		   "Number of rows doesn't match number of ells\n");
    }
    b->ell_list[ii]=my_malloc(b->nell_list[ii]*sizeof(int));
    b->w_list[ii]=my_malloc(b->nell_list[ii]*sizeof(double));
    b->f_ell[ii]=my_malloc(b->nell_list[ii]*sizeof(double));
    fits_read_col(fptr,TINT,1,1,1,nrows,&nulval,
		  b->ell_list[ii],&anynul,status);
    fits_read_col(fptr,TDOUBLE,2,1,1,nrows,&nulval,
		  b->w_list[ii],&anynul,status);
    fits_read_col(fptr,TDOUBLE,3,1,1,nrows,&nulval,
		  b->f_ell[ii],&anynul,status);
  }

  return b;
}

static void nmt_binning_scheme_tohdus(fitsfile *fptr,
				      nmt_binning_scheme *b,
				      int *status)
{
  int ii;
  char title[256];
  char **ttype,**tform,**tunit;
  ttype=my_malloc(3*sizeof(char *));
  tform=my_malloc(3*sizeof(char *));
  tunit=my_malloc(3*sizeof(char *));
  for(ii=0;ii<3;ii++) {
    ttype[ii]=my_malloc(256);
    tform[ii]=my_malloc(256);
    tunit[ii]=my_malloc(256);
    sprintf(tunit[ii]," ");
  }
  sprintf(ttype[0],"NL");
  sprintf(tform[0],"1J");
  sprintf(tunit[0]," ");

  fits_create_tbl(fptr,BINARY_TBL,0,1,ttype,tform,tunit,"BINS_SUMMARY",status);
  fits_write_col(fptr,TINT,1,1,1,b->n_bands,b->nell_list,status);
  fits_write_key(fptr,TINT,"N_BANDS",&(b->n_bands),NULL,status);
  fits_write_key(fptr,TINT,"ELL_MAX",&(b->ell_max),NULL,status);

  sprintf(ttype[0],"ELLS");
  sprintf(ttype[1],"WEIGHTS");
  sprintf(ttype[2],"F_ELL");
  sprintf(tform[0],"1J");
  sprintf(tform[1],"1D");
  sprintf(tform[2],"1D");
  sprintf(tunit[0]," ");
  sprintf(tunit[1]," ");
  sprintf(tunit[2]," ");

  for(ii=0;ii<b->n_bands;ii++) {
    sprintf(title,"BAND_%d",ii+1);
    fits_create_tbl(fptr,BINARY_TBL,0,3,ttype,tform,tunit,title,status);
    fits_write_col(fptr,TINT,1,1,1,b->nell_list[ii],b->ell_list[ii],status);
    fits_write_col(fptr,TDOUBLE,2,1,1,b->nell_list[ii],b->w_list[ii],status);
    fits_write_col(fptr,TDOUBLE,3,1,1,b->nell_list[ii],b->f_ell[ii],status);
    fits_write_key(fptr,TINT,"I_BAND",&ii,NULL,status);
  }

  for(ii=0;ii<3;ii++) {
    free(ttype[ii]);
    free(tform[ii]);
    free(tunit[ii]);
  }
  free(ttype);
  free(tform);
  free(tunit);  
}

static void nmt_coupling_binned_tohdus(fitsfile *fptr,
				       nmt_workspace *w,
				       int *status)
{
  long ii,jj;
  long n_el=w->ncls*w->bin->n_bands;
  long naxes[2]={n_el,n_el};
  long fpixel[2]={1,1};

  //Flatten matrix
  double *matrix_binned=my_malloc(n_el*n_el*sizeof(double));
  for(ii=0;ii<n_el;ii++) {
    int i0=ii*n_el;
    for(jj=0;jj<n_el;jj++)
      matrix_binned[i0+jj]=gsl_matrix_get(w->coupling_matrix_binned,ii,jj);
  }

  //Create HDU and write
  fits_create_img(fptr,DOUBLE_IMG,2,naxes,status);
  fits_write_pix(fptr,TDOUBLE,fpixel,n_el*n_el,matrix_binned,status);
  free(matrix_binned);

  //Permutation to vector
  int *perm=my_malloc(n_el*sizeof(int));
  for(ii=0;ii<n_el;ii++)
    perm[ii]=(int)(w->coupling_matrix_perm->data[ii]);

  //Create HDU and write
  fits_create_img(fptr,LONG_IMG,1,naxes,status);
  fits_write_pix(fptr,TINT,fpixel,n_el,perm,status);
  free(perm);
}

static void nmt_coupling_binned_fromhdus(fitsfile *fptr,
					 nmt_workspace *w,
					 int *status)
{
  long n_el=w->ncls*w->bin->n_bands;
  long fpixel[2]={1,1};

  //Read flattened coupling matrix
  double *matrix_binned=my_malloc(n_el*n_el*sizeof(double));
  fits_movrel_hdu(fptr,1,NULL,status);
  fits_read_pix(fptr,TDOUBLE,fpixel,n_el*n_el,NULL,matrix_binned,NULL,status);

  //Unflatten
  w->coupling_matrix_binned=gsl_matrix_alloc(n_el,n_el);
  long ii,jj;
  for(ii=0;ii<n_el;ii++) {
    int i0=ii*n_el;
    for(jj=0;jj<n_el;jj++)
      gsl_matrix_set(w->coupling_matrix_binned,ii,jj,matrix_binned[i0+jj]);
  }
  free(matrix_binned);

  //Read permutation
  int *perm=my_malloc(n_el*sizeof(int));
  fits_movrel_hdu(fptr,1,NULL,status);
  fits_read_pix(fptr,TINT,fpixel,n_el,NULL,perm,NULL,status);

  w->coupling_matrix_perm=gsl_permutation_alloc(n_el);
  for(ii=0;ii<n_el;ii++)
    w->coupling_matrix_perm->data[ii]=perm[ii];
  free(perm);
}

static void check_fits(int status,char *fname,int is_read)
{
  if(status) {
    if(is_read)
      report_error(NMT_ERROR_READ,"Error reading file %s\n",fname);
    else
      report_error(NMT_ERROR_WRITE,"Error writing file %s\n",fname);
  }
}

void nmt_workspace_write_fits(nmt_workspace *w,char *fname)
{
  fitsfile *fptr;
  int status=0;
  fits_create_file(&fptr,fname,&status);
  check_fits(status,fname,0);
  // Workspace info HDU
  nmt_workspace_info_tohdus(fptr,w,&status);
  check_fits(status,fname,0);
  // CS info HDU
  nmt_curvedsky_info_tohdus(fptr,w->cs,&status);
  check_fits(status,fname,0);
  // beam_prod HDU
  nmt_l_arr_tohdus(fptr,w->lmax_fields,w->beam_prod,"BEAMS",&status);
  check_fits(status,fname,0);
  // pcl_masks HDU
  nmt_l_arr_tohdus(fptr,w->lmax_mask,w->pcl_masks,"PCL_MASKS",&status);
  check_fits(status,fname,0);
  // bins HDUs
  nmt_binning_scheme_tohdus(fptr,w->bin,&status);
  check_fits(status,fname,0);
  // binned MCM HDU
  nmt_coupling_binned_tohdus(fptr,w,&status);
  check_fits(status,fname,0);
  fits_close_file(fptr,&status);
}

nmt_workspace *nmt_workspace_read_fits(char *fname)
{
  fitsfile *fptr;
  int status=0;
  nmt_workspace *w=my_malloc(sizeof(nmt_workspace));

  fits_open_file(&fptr,fname,READONLY,&status);
  check_fits(status,fname,1);
  // Workspace info HDU
  nmt_workspace_info_fromhdus(fptr,w,&status);
  check_fits(status,fname,1);
  // CS info HDU
  w->cs=nmt_curvedsky_info_fromhdus(fptr,&status);
  check_fits(status,fname,1);
  // beam_prod HDU
  w->beam_prod=nmt_l_arr_fromhdus(fptr,w->lmax_fields,&status);
  check_fits(status,fname,1);
  // pcl_masks HDU
  w->pcl_masks=nmt_l_arr_fromhdus(fptr,w->lmax_mask,&status);
  check_fits(status,fname,1);
  // bins HDUs
  w->bin=nmt_binning_scheme_fromhdus(fptr,&status);
  check_fits(status,fname,1);
  // binned MCM HDU
  nmt_coupling_binned_fromhdus(fptr,w,&status);
  check_fits(status,fname,1);
  fits_close_file(fptr,&status);

  return w;
}
