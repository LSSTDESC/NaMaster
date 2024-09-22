#include "config.h"
#include "utils.h"
#include <fitsio.h>

static void nmt_workspace_info_tohdus(fitsfile *fptr,
				      nmt_workspace *w,
				      int *status)
{
  long ii;
  long n_el=w->ncls*(w->lmax+1);
  long naxes[2]={n_el,n_el};
  long fpixel[2]={1,1};
  fits_create_img(fptr,DOUBLE_IMG,2,naxes,status);
  fits_write_key(fptr,TSTRING,"EXTNAME","WSP_PRIMARY",NULL,status);
  fits_write_key(fptr,TINT,"LMAX",&(w->lmax),NULL,status);
  fits_write_key(fptr,TINT,"LMAX_FIELDS",&(w->lmax_fields),NULL,status);
  fits_write_key(fptr,TINT,"LMAX_MASK",&(w->lmax_mask),NULL,status);
  fits_write_key(fptr,TINT,"IS_TEB",&(w->is_teb),NULL,status);
  fits_write_key(fptr,TINT,"NCLS",&(w->ncls),NULL,status);
  fits_write_key(fptr,TINT,"NORM_TYPE",&(w->norm_type),NULL,status);
  for(ii=0;ii<n_el;ii++) {
    fpixel[1]=ii+1;
    fits_write_pix(fptr,TDOUBLE,fpixel,n_el,w->coupling_matrix_unbinned[ii],status);
  }
}

static void nmt_workspace_info_fromhdus(fitsfile *fptr,
					nmt_workspace *w,
					int w_unbinned,
					int *status)
{
  int status_pre;
  fits_movnam_hdu(fptr,IMAGE_HDU,"WSP_PRIMARY",0,status);
  fits_read_key(fptr,TINT,"LMAX",&(w->lmax),NULL,status);
  fits_read_key(fptr,TINT,"LMAX_FIELDS",&(w->lmax_fields),NULL,status);
  fits_read_key(fptr,TINT,"LMAX_MASK",&(w->lmax_mask),NULL,status);
  fits_read_key(fptr,TINT,"IS_TEB",&(w->is_teb),NULL,status);
  fits_read_key(fptr,TINT,"NCLS",&(w->ncls),NULL,status);
  status_pre = *status;
  fits_read_key(fptr,TINT,"NORM_TYPE",&(w->norm_type),NULL,status);
  if(*status) {// maybe used old format
    w->norm_type=0;
    *status=status_pre;
  }
  long ii;
  long n_el=w->ncls*(w->lmax+1);
  long fpixel[2]={1,1};
  if(w_unbinned) {
    w->coupling_matrix_unbinned=my_malloc(n_el*sizeof(flouble *));
    for(ii=0;ii<n_el;ii++) {
      fpixel[1]=ii+1;
      w->coupling_matrix_unbinned[ii]=my_malloc(n_el*sizeof(flouble));
      fits_read_pix(fptr,TDOUBLE,fpixel,n_el,NULL,
		    w->coupling_matrix_unbinned[ii],NULL,status);
    }
  }
  else
    w->coupling_matrix_unbinned=NULL;
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

static double *nmt_l_arr_fromhdus(fitsfile *fptr, int lmax_expect, char *arrname, int *status)
{
  double *f;
  double nulval;
  int anynul;
  long nrows;
  *status=0;
  fits_movnam_hdu(fptr,BINARY_TBL,arrname,0,status);
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
  nmt_binning_scheme *b;
  int *bpws,*ells;
  flouble *weights,*f_ell;
  int anynul,n_bands,ell_max;
  long nrows;
  double nulval;
  fits_movnam_hdu(fptr,BINARY_TBL,"BANDPOWERS",0,status);
  fits_read_key(fptr,TINT,"N_BANDS",&n_bands,NULL,status);
  fits_read_key(fptr,TINT,"ELL_MAX",&ell_max,NULL,status);
  fits_get_num_rows(fptr,&nrows,status);
  bpws=my_malloc(nrows*sizeof(int));
  ells=my_malloc(nrows*sizeof(int));
  weights=my_malloc(nrows*sizeof(flouble));
  f_ell=my_malloc(nrows*sizeof(flouble));
  fits_read_col(fptr,TINT,1,1,1,nrows,&nulval,
		bpws,&anynul,status);
  fits_read_col(fptr,TINT,2,1,1,nrows,&nulval,
		ells,&anynul,status);
  fits_read_col(fptr,TDOUBLE,3,1,1,nrows,&nulval,
		weights,&anynul,status);
  fits_read_col(fptr,TDOUBLE,4,1,1,nrows,&nulval,
		f_ell,&anynul,status);
  b=nmt_bins_create((int)nrows,bpws,ells,weights,f_ell,ell_max);
  if(n_bands!=b->n_bands) {
    report_error(NMT_ERROR_INCONSISTENT,
		 "Number of bands doesn't match table contents %d %d\n",
		 n_bands, b->n_bands);
  }
  free(bpws);
  free(ells);
  free(weights);
  free(f_ell);
  return b;
}

static nmt_binning_scheme *nmt_binning_scheme_fromhdus_old(fitsfile *fptr,
							   int *status)
{
  int ii,anynul;
  long nrows;
  double nulval;
  nmt_binning_scheme *b=my_malloc(sizeof(nmt_binning_scheme));
  fits_movnam_hdu(fptr,BINARY_TBL,"BINS_SUMMARY",0,status);
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
    char hduname[256];
    sprintf(hduname,"BINS_BAND_%d",ii+1);
    fits_movnam_hdu(fptr,BINARY_TBL,hduname,0,status);
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
  int nrows=b->ell_max+1;
  int *bpws=my_malloc(nrows*sizeof(int));
  int *ells=my_malloc(nrows*sizeof(int));
  flouble *weights=my_calloc(nrows,sizeof(flouble));
  flouble *f_ell=my_calloc(nrows,sizeof(flouble));
  for(ii=0;ii<nrows;ii++) {
    bpws[ii]=-1;
    ells[ii]=-1;
  }

  for(ii=0;ii<b->n_bands;ii++) {
    int jj;
    for(jj=0;jj<b->nell_list[ii];jj++) {
      int l=b->ell_list[ii][jj];
      ells[l]=l;
      bpws[l]=ii;
      weights[l]=b->w_list[ii][jj];
      f_ell[l]=b->f_ell[ii][jj];
    }
  }

  char **ttype,**tform,**tunit;
  ttype=my_malloc(4*sizeof(char *));
  tform=my_malloc(4*sizeof(char *));
  tunit=my_malloc(4*sizeof(char *));
  for(ii=0;ii<4;ii++) {
    ttype[ii]=my_malloc(256);
    tform[ii]=my_malloc(256);
    tunit[ii]=my_malloc(256);
    sprintf(tunit[ii]," ");
  }
  sprintf(ttype[0],"BAND");
  sprintf(ttype[1],"ELLS");
  sprintf(ttype[2],"WEIGHTS");
  sprintf(ttype[3],"F_ELL");
  sprintf(tform[0],"1J");
  sprintf(tform[1],"1J");
  sprintf(tform[2],"1D");
  sprintf(tform[3],"1D");
  fits_create_tbl(fptr,BINARY_TBL,0,4,ttype,tform,tunit,"BANDPOWERS",status);
  fits_write_key(fptr,TINT,"N_BANDS",&(b->n_bands),NULL,status);
  fits_write_key(fptr,TINT,"ELL_MAX",&(b->ell_max),NULL,status);
  fits_write_col(fptr,TINT,1,1,1,nrows,bpws,status);
  fits_write_col(fptr,TINT,2,1,1,nrows,ells,status);
  fits_write_col(fptr,TDOUBLE,3,1,1,nrows,weights,status);
  fits_write_col(fptr,TDOUBLE,4,1,1,nrows,f_ell,status);

  for(ii=0;ii<4;ii++) {
    free(ttype[ii]);
    free(tform[ii]);
    free(tunit[ii]);
  }
  free(ttype);
  free(tform);
  free(tunit);
  free(bpws);
  free(ells);
  free(weights);
  free(f_ell);
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
  fits_write_key(fptr,TSTRING,"EXTNAME","MCM_BINNED",NULL,status);
  fits_write_pix(fptr,TDOUBLE,fpixel,n_el*n_el,matrix_binned,status);
  free(matrix_binned);

  //Permutation to vector
  int *perm=my_malloc(n_el*sizeof(int));
  for(ii=0;ii<n_el;ii++)
    perm[ii]=(int)(w->coupling_matrix_perm->data[ii]);

  //Create HDU and write
  fits_create_img(fptr,LONG_IMG,1,naxes,status);
  fits_write_key(fptr,TSTRING,"EXTNAME","MCM_PERM",NULL,status);
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
  fits_movnam_hdu(fptr,IMAGE_HDU,"MCM_BINNED",0,status);
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
  fits_movnam_hdu(fptr,IMAGE_HDU,"MCM_PERM",0,status);
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

nmt_workspace *nmt_workspace_read_fits(char *fname, int w_unbinned)
{
  fitsfile *fptr;
  int status=0;
  nmt_workspace *w=my_malloc(sizeof(nmt_workspace));

  fits_open_file(&fptr,fname,READONLY,&status);
  check_fits(status,fname,1);
  // Workspace info HDU
  nmt_workspace_info_fromhdus(fptr,w,w_unbinned,&status);
  check_fits(status,fname,1);
  // beam_prod HDU
  w->beam_prod=nmt_l_arr_fromhdus(fptr,w->lmax_fields,"BEAMS",&status);
  check_fits(status,fname,1);
  // pcl_masks HDU
  w->pcl_masks=nmt_l_arr_fromhdus(fptr,w->lmax_mask,"PCL_MASKS",&status);
  check_fits(status,fname,1);
  // bins HDUs
  w->bin=nmt_binning_scheme_fromhdus(fptr,&status);
  if(status) // maybe used old format
    w->bin=nmt_binning_scheme_fromhdus_old(fptr,&status);
  check_fits(status,fname,1);
  // binned MCM HDU
  nmt_coupling_binned_fromhdus(fptr,w,&status);
  check_fits(status,fname,1);
  fits_close_file(fptr,&status);

  return w;
}

static void nmt_workspace_flat_info_tohdus(fitsfile *fptr,
					   nmt_workspace_flat *w,
					   int *status)
{
  long ii;
  long n_el1=w->ncls*w->bin->n_bands;
  long n_el2=w->ncls*w->fs->n_ell;
  long naxes[2]={n_el2,n_el1};
  long fpixel[2]={1,1};
  fits_create_img(fptr,DOUBLE_IMG,2,naxes,status);
  fits_write_key(fptr,TSTRING,"EXTNAME","WSP_PRIMARY",NULL,status);
  fits_write_key(fptr,TDOUBLE,"LMAX",&(w->lmax),NULL,status);
  fits_write_key(fptr,TDOUBLE,"ELLCUT_X_I",&(w->ellcut_x[0]),NULL,status);
  fits_write_key(fptr,TDOUBLE,"ELLCUT_X_F",&(w->ellcut_x[1]),NULL,status);
  fits_write_key(fptr,TDOUBLE,"ELLCUT_Y_I",&(w->ellcut_y[0]),NULL,status);
  fits_write_key(fptr,TDOUBLE,"ELLCUT_Y_F",&(w->ellcut_y[1]),NULL,status);
  fits_write_key(fptr,TINT,"PURE_E1",&(w->pe1),NULL,status);
  fits_write_key(fptr,TINT,"PURE_E2",&(w->pe2),NULL,status);
  fits_write_key(fptr,TINT,"PURE_B1",&(w->pb1),NULL,status);
  fits_write_key(fptr,TINT,"PURE_B2",&(w->pb2),NULL,status);
  fits_write_key(fptr,TINT,"IS_TEB",&(w->is_teb),NULL,status);
  fits_write_key(fptr,TINT,"NCLS",&(w->ncls),NULL,status);
  for(ii=0;ii<n_el1;ii++) {
    fpixel[1]=ii+1;
    fits_write_pix(fptr,TDOUBLE,fpixel,n_el2,w->coupling_matrix_unbinned[ii],status);
  }
}

static void nmt_workspace_flat_info_fromhdus(fitsfile *fptr,
					     nmt_workspace_flat *w,
					     int *status)
{
  fits_movnam_hdu(fptr,IMAGE_HDU,"WSP_PRIMARY",0,status);
  fits_read_key(fptr,TDOUBLE,"LMAX",&(w->lmax),NULL,status);
  fits_read_key(fptr,TDOUBLE,"ELLCUT_X_I",&(w->ellcut_x[0]),NULL,status);
  fits_read_key(fptr,TDOUBLE,"ELLCUT_X_F",&(w->ellcut_x[1]),NULL,status);
  fits_read_key(fptr,TDOUBLE,"ELLCUT_Y_I",&(w->ellcut_y[0]),NULL,status);
  fits_read_key(fptr,TDOUBLE,"ELLCUT_Y_F",&(w->ellcut_y[1]),NULL,status);
  fits_read_key(fptr,TINT,"PURE_E1",&(w->pe1),NULL,status);
  fits_read_key(fptr,TINT,"PURE_E2",&(w->pe2),NULL,status);
  fits_read_key(fptr,TINT,"PURE_B1",&(w->pb1),NULL,status);
  fits_read_key(fptr,TINT,"PURE_B2",&(w->pb2),NULL,status);
  fits_read_key(fptr,TINT,"IS_TEB",&(w->is_teb),NULL,status);
  fits_read_key(fptr,TINT,"NCLS",&(w->ncls),NULL,status);
  long ii;
  long naxes[2],fpixel[2]={1,1};
  fits_get_img_size(fptr,2,naxes,status);
  w->coupling_matrix_unbinned=my_malloc(naxes[1]*sizeof(flouble *));
  for(ii=0;ii<naxes[1];ii++) {
    fpixel[1]=ii+1;
    w->coupling_matrix_unbinned[ii]=my_malloc(naxes[0]*sizeof(flouble));
    fits_read_pix(fptr,TDOUBLE,fpixel,naxes[0],NULL,
		  w->coupling_matrix_unbinned[ii],NULL,status);
  }
  if(*status == 412)
    *status=0;
}

static void nmt_flatsky_info_tohdus(fitsfile *fptr,
				    nmt_flatsky_info *fs,
				    int *status)
{
  char **ttype,**tform,**tunit;
  ttype=my_malloc(1*sizeof(char *));
  ttype[0]=my_malloc(256); sprintf(ttype[0],"L_MIN");
  tform=my_malloc(1*sizeof(char *));
  tform[0]=my_malloc(256); sprintf(tform[0],"1D");
  tunit=my_malloc(1*sizeof(char *));
  tunit[0]=my_malloc(256); sprintf(tunit[0]," ");

  fits_create_tbl(fptr,BINARY_TBL,0,1,ttype,tform,tunit,"FS_INFO",status);
  fits_write_col(fptr,TDOUBLE,1,1,1,fs->n_ell,fs->ell_min,status);
  fits_write_key(fptr,TINT   ,"NX",&(fs->nx),NULL,status);
  fits_write_key(fptr,TINT   ,"NY",&(fs->ny),NULL,status);
  fits_write_key(fptr,TLONG  ,"NPIX",&(fs->npix),NULL,status);
  fits_write_key(fptr,TDOUBLE,"LX",&(fs->lx),NULL,status);
  fits_write_key(fptr,TDOUBLE,"LY",&(fs->ly),NULL,status);
  fits_write_key(fptr,TDOUBLE,"PIXSIZE",&(fs->pixsize),NULL,status);
  fits_write_key(fptr,TDOUBLE,"DELL",&(fs->dell),NULL,status);
  fits_write_key(fptr,TDOUBLE,"I_DELL",&(fs->i_dell),NULL,status);

  free(ttype[0]); free(ttype);
  free(tform[0]); free(tform);
  free(tunit[0]); free(tunit);
}

static nmt_flatsky_info *nmt_flatsky_info_fromhdus(fitsfile *fptr,
						   int *status)
{
  nmt_flatsky_info *fs=my_malloc(sizeof(nmt_flatsky_info));

  fits_movnam_hdu(fptr,BINARY_TBL,"FS_INFO",0,status);
  fits_read_key(fptr,TINT   ,"NX",&(fs->nx),NULL,status);
  fits_read_key(fptr,TINT   ,"NY",&(fs->ny),NULL,status);
  fits_read_key(fptr,TLONG  ,"NPIX",&(fs->npix),NULL,status);
  fits_read_key(fptr,TDOUBLE,"LX",&(fs->lx),NULL,status);
  fits_read_key(fptr,TDOUBLE,"LY",&(fs->ly),NULL,status);
  fits_read_key(fptr,TDOUBLE,"PIXSIZE",&(fs->pixsize),NULL,status);
  fits_read_key(fptr,TDOUBLE,"DELL",&(fs->dell),NULL,status);
  fits_read_key(fptr,TDOUBLE,"I_DELL",&(fs->i_dell),NULL,status);

  long nrows;
  double nulval;
  int anynul;
  fits_get_num_rows(fptr,&nrows,status);
  fs->n_ell=nrows;
  fs->ell_min=my_malloc(fs->n_ell*sizeof(flouble));
  fits_read_col(fptr,TDOUBLE,1,1,1,nrows,&nulval,fs->ell_min,&anynul,status);

  if(*status == 412)
    *status=0;

  return fs;
}

static void nmt_n_cells_tohdus(fitsfile *fptr,
			       int n,int *n_cells,
			       int *status)
{
  char **ttype,**tform,**tunit;
  ttype=my_malloc(1*sizeof(char *));
  ttype[0]=my_malloc(256); sprintf(ttype[0],"N_CELLS");
  tform=my_malloc(1*sizeof(char *));
  tform[0]=my_malloc(256); sprintf(tform[0],"1J");
  tunit=my_malloc(1*sizeof(char *));
  tunit[0]=my_malloc(256); sprintf(tunit[0]," ");

  fits_create_tbl(fptr,BINARY_TBL,0,1,ttype,tform,tunit,"N_CELLS",status);
  fits_write_col(fptr,TINT,1,1,1,n,n_cells,status);

  free(ttype[0]); free(ttype);
  free(tform[0]); free(tform);
  free(tunit[0]); free(tunit);
}

static int *nmt_n_cells_fromhdus(fitsfile *fptr,
				 int *status)
{
  int *n_cells;
  long nrows;
  int nulval,anynul;
  fits_movnam_hdu(fptr,BINARY_TBL,"N_CELLS",0,status);
  fits_get_num_rows(fptr,&nrows,status);
  n_cells=my_malloc(nrows*sizeof(int));
  fits_read_col(fptr,TINT,1,1,1,nrows,&nulval,n_cells,&anynul,status);

  if(*status == 412)
    *status=0;

  return n_cells;
}

static void nmt_flat_coupling_binned_tohdus(fitsfile *fptr,
					    nmt_workspace_flat *w,
					    int *status)
{
  long ii;
  long n_el=w->ncls*w->bin->n_bands;
  long naxes[2]={n_el,n_el};
  long fpixel[2]={1,1};

  //Non-GSL
  fits_create_img(fptr,DOUBLE_IMG,2,naxes,status);
  fits_write_key(fptr,TSTRING,"EXTNAME","MCM_BINNED",NULL,status);
  for(ii=0;ii<n_el;ii++) {
    fpixel[1]=ii+1;
    fits_write_pix(fptr,TDOUBLE,fpixel,n_el,w->coupling_matrix_binned[ii],status);
  }

  //GSL
  flouble *matrix=my_malloc(n_el*n_el*sizeof(flouble));
  for(ii=0;ii<n_el;ii++) {
    long jj,i0=ii*n_el;
    for(jj=0;jj<n_el;jj++)
      matrix[i0+jj]=gsl_matrix_get(w->coupling_matrix_binned_gsl,ii,jj);
  }
  fits_create_img(fptr,DOUBLE_IMG,2,naxes,status);
  fits_write_key(fptr,TSTRING,"EXTNAME","MCM_BINNED_GSL",NULL,status);
  fpixel[1]=1;
  fits_write_pix(fptr,TDOUBLE,fpixel,n_el*n_el,matrix,status);
  free(matrix);

  //Permutation
  int *perm=my_malloc(n_el*sizeof(int));
  for(ii=0;ii<n_el;ii++)
    perm[ii]=(int)(w->coupling_matrix_perm->data[ii]);
  fits_create_img(fptr,LONG_IMG,1,naxes,status);
  fits_write_key(fptr,TSTRING,"EXTNAME","MCM_PERM",NULL,status);
  fits_write_pix(fptr,TINT,fpixel,n_el,perm,status);
  free(perm);
}

static void nmt_flat_coupling_binned_fromhdus(fitsfile *fptr,
					      nmt_workspace_flat *w,
					      int *status)
{
  flouble *matrix;
  long ii,n_el;
  long naxes[2],fpixel[2]={1,1};

  //Non-GSL
  fits_movnam_hdu(fptr,IMAGE_HDU,"MCM_BINNED",0,status);
  fits_get_img_size(fptr,2,naxes,status);
  n_el=naxes[0];
  matrix=my_malloc(n_el*n_el*sizeof(flouble));
  fits_read_pix(fptr,TDOUBLE,fpixel,naxes[0]*naxes[1],NULL,matrix,NULL,status);
  w->coupling_matrix_binned=my_malloc(n_el*sizeof(flouble *));
  for(ii=0;ii<n_el;ii++) {
    w->coupling_matrix_binned[ii]=my_malloc(n_el*sizeof(flouble));
    memcpy(w->coupling_matrix_binned[ii],
	   &(matrix[ii*n_el]),n_el*sizeof(flouble));
  }

  //GSL
  fits_movnam_hdu(fptr,IMAGE_HDU,"MCM_BINNED_GSL",0,status);
  fits_get_img_size(fptr,2,naxes,status);
  if((naxes[0]!=n_el) || (naxes[1]!=n_el)) {
    report_error(NMT_ERROR_INCONSISTENT,
		 "Inconsistent size for GSL binned MCM\n");
  }
  fits_read_pix(fptr,TDOUBLE,fpixel,naxes[0]*naxes[1],NULL,matrix,NULL,status);
  w->coupling_matrix_binned_gsl=gsl_matrix_alloc(n_el,n_el);
  for(ii=0;ii<n_el;ii++) {
    long jj,i0=ii*n_el;
    for(jj=0;jj<n_el;jj++)
      gsl_matrix_set(w->coupling_matrix_binned_gsl,ii,jj,matrix[i0+jj]);
  }
  free(matrix);

  //Read permutation
  int *perm=my_malloc(n_el*sizeof(int));
  fits_movnam_hdu(fptr,IMAGE_HDU,"MCM_PERM",0,status);
  fits_read_pix(fptr,TINT,fpixel,n_el,NULL,perm,NULL,status);

  w->coupling_matrix_perm=gsl_permutation_alloc(n_el);
  for(ii=0;ii<n_el;ii++)
    w->coupling_matrix_perm->data[ii]=perm[ii];
  free(perm);
}

static void nmt_binning_scheme_flat_tohdus(fitsfile *fptr,
					   nmt_binning_scheme_flat *b,
					   int *status)
{
  int ii;
  char **ttype,**tform,**tunit;
  ttype=my_malloc(2*sizeof(char *));
  tform=my_malloc(2*sizeof(char *));
  tunit=my_malloc(2*sizeof(char *));
  for(ii=0;ii<2;ii++) {
    ttype[ii]=my_malloc(256);
    tform[ii]=my_malloc(256);
    tunit[ii]=my_malloc(256);
    sprintf(tform[ii],"1D");
    sprintf(tunit[ii]," ");
  }
  sprintf(ttype[0],"ELL_0");
  sprintf(ttype[1],"ELL_F");

  fits_create_tbl(fptr,BINARY_TBL,0,2,ttype,tform,tunit,"BINS_SUMMARY",status);
  fits_write_col(fptr,TDOUBLE,1,1,1,b->n_bands,b->ell_0_list,status);
  fits_write_col(fptr,TDOUBLE,2,1,1,b->n_bands,b->ell_f_list,status);

  for(ii=0;ii<2;ii++) {
    free(ttype[ii]);
    free(tform[ii]);
    free(tunit[ii]);
  }
  free(ttype);
  free(tform);
  free(tunit);  
}

static nmt_binning_scheme_flat *nmt_binning_scheme_flat_fromhdus(fitsfile *fptr,
								 int *status)
{
  int anynul;
  long nrows;
  double nulval;
  nmt_binning_scheme_flat *b=my_malloc(sizeof(nmt_binning_scheme_flat));

  fits_movnam_hdu(fptr,BINARY_TBL,"BINS_SUMMARY",0,status);
  fits_get_num_rows(fptr,&nrows,status);
  b->n_bands=nrows;
  b->ell_0_list=my_malloc(b->n_bands*sizeof(flouble));
  b->ell_f_list=my_malloc(b->n_bands*sizeof(flouble));
  fits_read_col(fptr,TDOUBLE,1,1,1,nrows,&nulval,
		b->ell_0_list,&anynul,status);
  fits_read_col(fptr,TDOUBLE,2,1,1,nrows,&nulval,
		b->ell_f_list,&anynul,status);

  return b;
}

void nmt_workspace_flat_write_fits(nmt_workspace_flat *w,char *fname)
{
  fitsfile *fptr;
  int status=0;
  fits_create_file(&fptr,fname,&status);
  check_fits(status,fname,0);
  // Workspace info HDU
  nmt_workspace_flat_info_tohdus(fptr,w,&status);
  check_fits(status,fname,0);
  // FS info HDU
  nmt_flatsky_info_tohdus(fptr,w->fs,&status);
  check_fits(status,fname,0);
  // n_cells HDU
  nmt_n_cells_tohdus(fptr,w->bin->n_bands,w->n_cells,&status);
  check_fits(status,fname,0);
  // binned MCM HDUs
  nmt_flat_coupling_binned_tohdus(fptr,w,&status);
  check_fits(status,fname,0);
  // bins HDU
  nmt_binning_scheme_flat_tohdus(fptr,w->bin,&status);
  check_fits(status,fname,0);
  fits_close_file(fptr,&status);
}

nmt_workspace_flat *nmt_workspace_flat_read_fits(char *fname)
{
  fitsfile *fptr;
  int status=0;
  nmt_workspace_flat *w=my_malloc(sizeof(nmt_workspace_flat));

  fits_open_file(&fptr,fname,READONLY,&status);
  check_fits(status,fname,1);
  // Workspace info HDU
  nmt_workspace_flat_info_fromhdus(fptr,w,&status);
  check_fits(status,fname,1);
  // FS info HDU
  w->fs=nmt_flatsky_info_fromhdus(fptr,&status);
  check_fits(status,fname,1);
  // n_cells HDU
  w->n_cells=nmt_n_cells_fromhdus(fptr,&status);
  check_fits(status,fname,1);
  // binned MCM HDUs
  nmt_flat_coupling_binned_fromhdus(fptr,w,&status);
  check_fits(status,fname,1);
  // bins HDU
  w->bin=nmt_binning_scheme_flat_fromhdus(fptr,&status);
  check_fits(status,fname,1);
  fits_close_file(fptr,&status);

  return w;
}

static void nmt_covar_coeffs_tohdus(fitsfile *fptr,
				    int n_expected,double **coeff,
				    char *name,int *status)
{
  long ii,n_el=n_expected;
  long naxes[2]={n_el,n_el};
  long fpixel[2]={1,1};
  fits_create_img(fptr,DOUBLE_IMG,2,naxes,status);
  fits_write_key(fptr,TSTRING,"EXTNAME",name,NULL,status);
  for(ii=0;ii<n_el;ii++) {
    fpixel[1]=ii+1;
    fits_write_pix(fptr,TDOUBLE,fpixel,n_el,coeff[ii],status);
  }
}

static double **nmt_covar_coeffs_fromhdus(fitsfile *fptr,
					  int n_expected,
					  char *name,
					  int *status)
{
  flouble *matrix;
  long ii,n_el;
  long naxes[2],fpixel[2]={1,1};

  fits_movnam_hdu(fptr,IMAGE_HDU,name,0,status);
  fits_get_img_size(fptr,2,naxes,status);
  n_el=naxes[0];
  if(n_el!=n_expected)
    report_error(NMT_ERROR_INCONSISTENT,"Mistmatching coefficient size\n");
  matrix=my_malloc(n_el*n_el*sizeof(flouble));
  fits_read_pix(fptr,TDOUBLE,fpixel,naxes[0]*naxes[1],NULL,matrix,NULL,status);

  double **coeff=my_malloc(n_el*sizeof(flouble *));
  for(ii=0;ii<n_el;ii++) {
    coeff[ii]=my_malloc(n_el*sizeof(flouble));
    memcpy(coeff[ii],&(matrix[ii*n_el]),n_el*sizeof(flouble));
  }
  free(matrix);
  return coeff;
}

void nmt_covar_workspace_write_fits(nmt_covar_workspace *cw,char *fname)
{
  fitsfile *fptr;
  int status=0;
  fits_create_file(&fptr,fname,&status);
  check_fits(status,fname,0);

  //Empty primary
  fits_create_img(fptr,BYTE_IMG,0,NULL,&status);
  fits_write_key(fptr,TSTRING,"EXTNAME","CWSP_PRIMARY",NULL,&status);
  fits_write_key(fptr,TINT,"LMAX",&(cw->lmax),NULL,&status);
  cw->lmax_mask=cw->lmax;
  fits_write_key(fptr,TINT,"SPIN0_ONLY",&(cw->spin0_only),NULL,&status);
  check_fits(status,fname,0);

  nmt_covar_coeffs_tohdus(fptr,cw->lmax+1,cw->xi00_1122,"XI00_1122",&status);
  check_fits(status,fname,0);
  nmt_covar_coeffs_tohdus(fptr,cw->lmax+1,cw->xi00_1221,"XI00_1221",&status);
  check_fits(status,fname,0);
  if(cw->spin0_only == 0) {
    nmt_covar_coeffs_tohdus(fptr,cw->lmax+1,cw->xi02_1122,"XI02_1122",&status);
    check_fits(status,fname,0);
    nmt_covar_coeffs_tohdus(fptr,cw->lmax+1,cw->xi02_1221,"XI02_1221",&status);
    check_fits(status,fname,0);
    nmt_covar_coeffs_tohdus(fptr,cw->lmax+1,cw->xi22p_1122,"XI22P_1122",&status);
    check_fits(status,fname,0);
    nmt_covar_coeffs_tohdus(fptr,cw->lmax+1,cw->xi22p_1221,"XI22P_1221",&status);
    check_fits(status,fname,0);
    nmt_covar_coeffs_tohdus(fptr,cw->lmax+1,cw->xi22m_1122,"XI22M_1122",&status);
    check_fits(status,fname,0);
    nmt_covar_coeffs_tohdus(fptr,cw->lmax+1,cw->xi22m_1221,"XI22M_1221",&status);
    check_fits(status,fname,0);
  }
  fits_close_file(fptr,&status);
}

nmt_covar_workspace *nmt_covar_workspace_read_fits(char *fname, int force_spin0)
{
  fitsfile *fptr;
  int status=0;
  nmt_covar_workspace *cw=my_malloc(sizeof(nmt_covar_workspace));

  fits_open_file(&fptr,fname,READONLY,&status);
  check_fits(status,fname,1);
  fits_movnam_hdu(fptr,ANY_HDU,"CWSP_PRIMARY",0,&status);
  fits_read_key(fptr,TINT,"LMAX",&(cw->lmax),NULL,&status);
  check_fits(status,fname,1);
  if(force_spin0)
    cw->spin0_only=1;
  else {
    fits_read_key(fptr,TINT,"SPIN0_ONLY",&(cw->spin0_only),NULL,&status);
    if(status) {//Old format, always includes all spins
      cw->spin0_only=0;
      status=0;
    }
  }
  //Empty primary

  cw->xi00_1122=nmt_covar_coeffs_fromhdus(fptr,cw->lmax+1,"XI00_1122",&status);
  check_fits(status,fname,1);
  cw->xi00_1221=nmt_covar_coeffs_fromhdus(fptr,cw->lmax+1,"XI00_1221",&status);
  check_fits(status,fname,1);
  if(cw->spin0_only) {
    cw->xi02_1122=cw->xi00_1122;
    cw->xi02_1221=cw->xi00_1221;
    cw->xi22p_1122=cw->xi00_1122;
    cw->xi22p_1221=cw->xi00_1221;
    cw->xi22m_1122=cw->xi00_1122;
    cw->xi22m_1221=cw->xi00_1221;
  }
  else {
    cw->xi02_1122=nmt_covar_coeffs_fromhdus(fptr,cw->lmax+1,"XI02_1122",&status);
    check_fits(status,fname,1);
    cw->xi02_1221=nmt_covar_coeffs_fromhdus(fptr,cw->lmax+1,"XI02_1221",&status);
    check_fits(status,fname,1);
    cw->xi22p_1122=nmt_covar_coeffs_fromhdus(fptr,cw->lmax+1,"XI22P_1122",&status);
    check_fits(status,fname,1);
    cw->xi22p_1221=nmt_covar_coeffs_fromhdus(fptr,cw->lmax+1,"XI22P_1221",&status);
    check_fits(status,fname,1);
    cw->xi22m_1122=nmt_covar_coeffs_fromhdus(fptr,cw->lmax+1,"XI22M_1122",&status);
    check_fits(status,fname,1);
    cw->xi22m_1221=nmt_covar_coeffs_fromhdus(fptr,cw->lmax+1,"XI22M_1221",&status);
    check_fits(status,fname,1);
  }
  fits_close_file(fptr,&status);

  return cw;
}

void nmt_covar_workspace_flat_write_fits(nmt_covar_workspace_flat *cw,char *fname)
{
  fitsfile *fptr;
  int status=0;
  fits_create_file(&fptr,fname,&status);
  check_fits(status,fname,0);

  //Empty primary
  fits_create_img(fptr,BYTE_IMG,0,NULL,&status);
  fits_write_key(fptr,TSTRING,"EXTNAME","CWSP_PRIMARY",NULL,&status);
  check_fits(status,fname,0);

  //Bins
  nmt_binning_scheme_flat_tohdus(fptr,cw->bin,&status);
  check_fits(status,fname,0);

  //Coeffs
  nmt_covar_coeffs_tohdus(fptr,cw->bin->n_bands,cw->xi00_1122,"XI00_1122",&status);
  check_fits(status,fname,0);
  nmt_covar_coeffs_tohdus(fptr,cw->bin->n_bands,cw->xi00_1221,"XI00_1221",&status);
  check_fits(status,fname,0);
  nmt_covar_coeffs_tohdus(fptr,cw->bin->n_bands,cw->xi02_1122,"XI02_1122",&status);
  check_fits(status,fname,0);
  nmt_covar_coeffs_tohdus(fptr,cw->bin->n_bands,cw->xi02_1221,"XI02_1221",&status);
  check_fits(status,fname,0);
  nmt_covar_coeffs_tohdus(fptr,cw->bin->n_bands,cw->xi22p_1122,"XI22P_1122",&status);
  check_fits(status,fname,0);
  nmt_covar_coeffs_tohdus(fptr,cw->bin->n_bands,cw->xi22p_1221,"XI22P_1221",&status);
  check_fits(status,fname,0);
  nmt_covar_coeffs_tohdus(fptr,cw->bin->n_bands,cw->xi22m_1122,"XI22M_1122",&status);
  check_fits(status,fname,0);
  nmt_covar_coeffs_tohdus(fptr,cw->bin->n_bands,cw->xi22m_1221,"XI22M_1221",&status);
  check_fits(status,fname,0);
  fits_close_file(fptr,&status);
}

nmt_covar_workspace_flat *nmt_covar_workspace_flat_read_fits(char *fname)
{
  fitsfile *fptr;
  int status=0;
  nmt_covar_workspace_flat *cw=my_malloc(sizeof(nmt_covar_workspace_flat));

  fits_open_file(&fptr,fname,READONLY,&status);
  check_fits(status,fname,1);
  fits_movnam_hdu(fptr,IMAGE_HDU,"CWSP_PRIMARY",0,&status);

  //Bins
  cw->bin=nmt_binning_scheme_flat_fromhdus(fptr,&status);
  check_fits(status,fname,1);

  //Coeffs
  cw->xi00_1122=nmt_covar_coeffs_fromhdus(fptr,cw->bin->n_bands,"XI00_1122",&status);
  check_fits(status,fname,1);
  cw->xi00_1221=nmt_covar_coeffs_fromhdus(fptr,cw->bin->n_bands,"XI00_1221",&status);
  check_fits(status,fname,1);
  cw->xi02_1122=nmt_covar_coeffs_fromhdus(fptr,cw->bin->n_bands,"XI02_1122",&status);
  check_fits(status,fname,1);
  cw->xi02_1221=nmt_covar_coeffs_fromhdus(fptr,cw->bin->n_bands,"XI02_1221",&status);
  check_fits(status,fname,1);
  cw->xi22p_1122=nmt_covar_coeffs_fromhdus(fptr,cw->bin->n_bands,"XI22P_1122",&status);
  check_fits(status,fname,1);
  cw->xi22p_1221=nmt_covar_coeffs_fromhdus(fptr,cw->bin->n_bands,"XI22P_1221",&status);
  check_fits(status,fname,1);
  cw->xi22m_1122=nmt_covar_coeffs_fromhdus(fptr,cw->bin->n_bands,"XI22M_1122",&status);
  check_fits(status,fname,1);
  cw->xi22m_1221=nmt_covar_coeffs_fromhdus(fptr,cw->bin->n_bands,"XI22M_1221",&status);
  check_fits(status,fname,1);
  fits_close_file(fptr,&status);

  return cw;
}
