#include "config.h"
#include "utils.h"
#include <fitsio.h>


static void check_fits(int status,char *fname,int is_read)
{
  if(status) {
    if(is_read)
      report_error(NMT_ERROR_READ,"Error reading file %s\n",fname);
    else
      report_error(NMT_ERROR_WRITE,"Error writing file %s\n",fname);
  }
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

  int status_here=0;
  fits_movnam_hdu(fptr,IMAGE_HDU,name,0,&status_here);
  if(status_here)  // This coefficient is not stored
    return NULL;

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
