#include <stdio.h>

#define CTEST_MAIN
#include <fitsio.h>
#include "namaster.h"
#include "ctest.h"

#define NSIDE_TESTS 128


double *he_read_healpix_map(char *fname,long *nside,int nfield)
{
  //////
  // Reads a healpix map from file fname. The map will be
  // read from column #nfield. It also returns the map's nside.
  int status=0,hdutype,nfound,anynul,ncols;
  long naxes,*naxis,npix;
  fitsfile *fptr;
  double *map,nulval;
  char order_in_file[32];
  int nested_in_file=0;

  fits_open_file(&fptr,fname,READONLY,&status);
  fits_movabs_hdu(fptr,2,&hdutype,&status);
  fits_read_key_lng(fptr,"NAXIS",&naxes,NULL,&status);
  naxis=malloc(naxes*sizeof(long));
  fits_read_keys_lng(fptr,"NAXIS",1,naxes,naxis,&nfound,&status);
  fits_read_key_lng(fptr,"NSIDE",nside,NULL,&status);
  npix=12*(*nside)*(*nside);
  ASSERT_EQUAL(npix%naxis[1],0);

  if (fits_read_key(fptr, TSTRING, "ORDERING", order_in_file, NULL, &status))
    printf("WARNING: Could not find %s keyword in in file %s\n","ORDERING",fname);
  if(!strncmp(order_in_file,"NEST",4))
    nested_in_file=1;

  if(nested_in_file) {
    printf("Map should be in ring ordering\n");
    ASSERT_EQUAL(0,1);
  }

  map=malloc(npix*sizeof(double));
  fits_get_num_cols(fptr,&ncols,&status);
  if(nfield>=ncols) {
    printf("Not enough columns in FITS file\n");
    ASSERT_EQUAL(0,1);
  }

#ifdef _SPREC
  fits_read_col(fptr,TFLOAT,nfield+1,1,1,npix,&nulval,map,&anynul,&status);
#else //_SPREC
  fits_read_col(fptr,TDOUBLE,nfield+1,1,1,npix,&nulval,map,&anynul,&status);
#endif //_SPREC
  free(naxis);

  fits_close_file(fptr,&status);

  return map;
}

int *get_sequence(int n0,int nf)
{
  int i;
  int *seq=malloc((nf-n0)*sizeof(int));
  ASSERT_NOT_NULL(seq);
  for(i=0;i<(nf-n0);i++)
    seq[i]=n0+i;
  return seq;
}

CTEST(nmt,bins_constant) {
  nmt_binning_scheme *bin=nmt_bins_constant(4,2000);
  ASSERT_EQUAL(bin->n_bands,499);
  ASSERT_EQUAL(bin->ell_list[5][2],2+4*5+2);
  nmt_bins_free(bin);
}

CTEST(nmt,bins_var) {
  int i,j;
  nmt_binning_scheme *bin=nmt_bins_constant(4,2000);
  int *ells=get_sequence(2,1998);
  int *bpws=malloc(1996*sizeof(int));
  double *weights=malloc(1996*sizeof(double));
  for(i=0;i<1996;i++) {
    bpws[i]=i/4;
    weights[i]=0.25;
  }
  nmt_binning_scheme *bin2=nmt_bins_create(1996,bpws,ells,weights,2000);

  ASSERT_EQUAL(bin->n_bands,499);
  ASSERT_EQUAL(bin->n_bands,bin2->n_bands);
  ASSERT_EQUAL(bin->ell_list[5][2],2+4*5+2);
  ASSERT_EQUAL(bin2->ell_list[5][2],2+4*5+2);
  free(bpws);
  free(ells);
  free(weights);
  nmt_bins_free(bin);
  nmt_bins_free(bin2);
}

CTEST(nmt,field_t) {
  int i;
  int spins[1]={0};
  int *ell=get_sequence(0,3*NSIDE_TESTS);
  double *cl=malloc(3*NSIDE_TESTS*sizeof(double));
  double *bm=malloc(3*NSIDE_TESTS*sizeof(double));
  for(i=0;i<3*NSIDE_TESTS;i++) {
    cl[i]=pow((ell[i]+1.)/101.,-1.2);
    bm[i]=1.;
  }
  double **map=nmt_synfast_sph(NSIDE_TESTS,1,spins,3*NSIDE_TESTS-1,&cl,&bm,1234);
  double *mask=malloc(12*NSIDE_TESTS*NSIDE_TESTS*sizeof(double));
  for(i=0;i<12*NSIDE_TESTS*NSIDE_TESTS;i++)
    mask[i]=1.;
  nmt_field *fld=nmt_field_alloc_sph(NSIDE_TESTS,mask,0,map,0,NULL,NULL,0,0,3,1E-10);
  ASSERT_EQUAL(fld->nside,NSIDE_TESTS);
  ASSERT_EQUAL(fld->npix,12*NSIDE_TESTS*NSIDE_TESTS);
  nmt_field_free(fld);
  free(mask);
  free(map[0]);
  free(map);
  free(ell);
  free(cl);
}

CTEST(nmt,field_r) {
  nmt_field *fld=nmt_field_read("test/mask.fits","test/maps.fits","none","none",0,0,0,3,1E-10);
  ASSERT_EQUAL(fld->nside,256);
  nmt_field_free(fld);
}

void compare_arrays(int n,double *y,int narr,int iarr,char *fname)
{
  int ii;
  FILE *fi=fopen(fname,"r");
  ASSERT_NOT_NULL(fi);
  for(ii=0;ii<n;ii++) {
    int j;
    double xv,yv,dum;
    int stat=fscanf(fi,"%lf",&xv);
    ASSERT_EQUAL(stat,1);
    for(j=0;j<narr;j++) {
      stat=fscanf(fi,"%lE",&dum);
      ASSERT_EQUAL(stat,1);
      if(j==iarr)
	yv=dum;
    }
    ASSERT_DBL_NEAR_TOL(yv,y[ii],fmax(fabs(yv),fabs(y[ii]))*1E-3);
  }
  fclose(fi);
}

CTEST(nmt,cell_00) {
  long nside,ipix;
  double *mpt=he_read_healpix_map("test/maps.fits",&nside,0);
  double *msk=he_read_healpix_map("test/mask.fits",&nside,0);
  nmt_field *f0=nmt_field_alloc_sph(nside,msk,0,&mpt,0,NULL,NULL,0,0,3,1E-10);
  nmt_binning_scheme *bin=nmt_bins_constant(20,3*nside-1);
  nmt_workspace *w00=nmt_compute_coupling_matrix(f0,f0,bin);
  double **cell=malloc(1*sizeof(double *));
  double **cell_out=malloc(1*sizeof(double *));
  double **cell_noise=malloc(1*sizeof(double *));
  ASSERT_NOT_NULL(cell); ASSERT_NOT_NULL(cell_noise); ASSERT_NOT_NULL(cell_out);
  cell[0]=malloc(3*nside*sizeof(double));
  cell_noise[0]=calloc(3*nside,sizeof(double));
  cell_out[0]=malloc(bin->n_bands*sizeof(double));
  ASSERT_NOT_NULL(cell[0]); ASSERT_NOT_NULL(cell_noise[0]); ASSERT_NOT_NULL(cell_out[0]);

  nmt_compute_coupled_cell(f0,f0,cell,3);
  nmt_decouple_cl_l(w00,cell,cell_noise,cell_noise,cell_out);
  compare_arrays(bin->n_bands,cell_out[0],1,0,"test/benchmarks/cl_00_benchmark.txt");
  
  free(cell[0]); free(cell);
  free(cell_noise[0]); free(cell_noise);
  free(cell_out[0]); free(cell_out);
  nmt_workspace_free(w00);
  nmt_field_free(f0);
  free(mpt);
  free(msk);
}

CTEST(nmt,cell_02) {
  long nside,ipix;
  double *mpt=he_read_healpix_map("test/maps.fits",&nside,0);
  double *mpq=he_read_healpix_map("test/maps.fits",&nside,1);
  double *mpu=he_read_healpix_map("test/maps.fits",&nside,2);
  double *msk=he_read_healpix_map("test/mask.fits",&nside,0);
  double **mpp=malloc(2*sizeof(double *));
  mpp[0]=mpq; mpp[1]=mpu;

  nmt_field *f0=nmt_field_alloc_sph(nside,msk,0,&mpt,0,NULL,NULL,0,0,3,1E-10);
  nmt_field *f2=nmt_field_alloc_sph(nside,msk,1,mpp,0,NULL,NULL,0,0,3,1E-10);
  nmt_binning_scheme *bin=nmt_bins_constant(20,3*nside-1);

  nmt_workspace *w02=nmt_compute_coupling_matrix(f0,f2,bin);
  double **cell=malloc(2*sizeof(double *));
  double **cell_out=malloc(2*sizeof(double *));
  double **cell_noise=malloc(2*sizeof(double *));
  ASSERT_NOT_NULL(cell); ASSERT_NOT_NULL(cell_noise); ASSERT_NOT_NULL(cell_out);
  for(ipix=0;ipix<2;ipix++) {
    cell[ipix]=malloc(3*nside*sizeof(double));
    cell_noise[ipix]=calloc(3*nside,sizeof(double));
    cell_out[ipix]=malloc(bin->n_bands*sizeof(double));
    ASSERT_NOT_NULL(cell[ipix]); ASSERT_NOT_NULL(cell_noise[ipix]); ASSERT_NOT_NULL(cell_out[ipix]);
  }

  nmt_compute_coupled_cell(f0,f2,cell,3);
  nmt_decouple_cl_l(w02,cell,cell_noise,cell_noise,cell_out);
  for(ipix=0;ipix<2;ipix++)
    compare_arrays(bin->n_bands,cell_out[ipix],2,ipix,"test/benchmarks/cl_02_benchmark.txt");
  
  for(ipix=0;ipix<2;ipix++) {
    free(cell[ipix]);
    free(cell_noise[ipix]);
    free(cell_out[ipix]); 
  }
  free(cell_out);
  free(cell_noise);
  free(cell);
  nmt_workspace_free(w02);
  nmt_field_free(f0);
  nmt_field_free(f2);
  free(mpt);
  free(mpq);
  free(mpu);
  free(mpp);
  free(msk);
}

CTEST(nmt,cell_22) {
  long nside,ipix;
  double *mpq=he_read_healpix_map("test/maps.fits",&nside,1);
  double *mpu=he_read_healpix_map("test/maps.fits",&nside,2);
  double *msk=he_read_healpix_map("test/mask.fits",&nside,0);
  double **mpp=malloc(2*sizeof(double *));
  mpp[0]=mpq; mpp[1]=mpu;

  nmt_field *f2=nmt_field_alloc_sph(nside,msk,1,mpp,0,NULL,NULL,0,0,3,1E-10);
  nmt_binning_scheme *bin=nmt_bins_constant(20,3*nside-1);

  nmt_workspace *w22=nmt_compute_coupling_matrix(f2,f2,bin);
  double **cell=malloc(4*sizeof(double *));
  double **cell_out=malloc(4*sizeof(double *));
  double **cell_noise=malloc(4*sizeof(double *));
  ASSERT_NOT_NULL(cell); ASSERT_NOT_NULL(cell_noise); ASSERT_NOT_NULL(cell_out);
  for(ipix=0;ipix<4;ipix++) {
    cell[ipix]=malloc(3*nside*sizeof(double));
    cell_noise[ipix]=calloc(3*nside,sizeof(double));
    cell_out[ipix]=malloc(bin->n_bands*sizeof(double));
    ASSERT_NOT_NULL(cell[ipix]); ASSERT_NOT_NULL(cell_noise[ipix]); ASSERT_NOT_NULL(cell_out[ipix]);
  }

  nmt_compute_coupled_cell(f2,f2,cell,3);
  nmt_decouple_cl_l(w22,cell,cell_noise,cell_noise,cell_out);
  for(ipix=0;ipix<4;ipix++)
    compare_arrays(bin->n_bands,cell_out[ipix],4,ipix,"test/benchmarks/cl_22_benchmark.txt");
  
  for(ipix=0;ipix<4;ipix++) {
    free(cell[ipix]);
    free(cell_noise[ipix]);
    free(cell_out[ipix]); 
  }
  free(cell_out);
  free(cell_noise);
  free(cell);
  nmt_workspace_free(w22);
  nmt_field_free(f2);
  free(mpq);
  free(mpu);
  free(mpp);
  free(msk);
}

int main(int argc,const char *argv[])
{
  int result=ctest_main(argc,argv);
  return result;
}
