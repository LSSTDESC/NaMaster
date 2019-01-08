%module nmtlib

%{
#define SWIG_FILE_WITH_INIT
#include "../src/namaster.h"
#include "../src/utils.h"
%}

%include "numpy.i"
%include "exception.i"
%init %{
  import_array();
  set_error_policy(THROW_ON_ERROR);
%}

%rename("%(strip:[nmt_])s") "";

%include "../src/namaster.h"

%apply (int* ARGOUT_ARRAY1, int DIM1) {(int* iout, int niout)};
%apply (double* ARGOUT_ARRAY1, int DIM1) {(double* dout, int ndout)};
%apply (double* ARGOUT_ARRAY1, long DIM1) {(double* ldout, long nldout)};
%apply (int DIM1,double *IN_ARRAY1) {(int npix_1,double *mask),
                                     (int nell11,double *c11),
                                     (int nell12,double *c12),
                                     (int nell21,double *c21),
                                     (int nell22,double *c22),
                                     (int nell3,double *weights)};
%apply (int DIM1,int *IN_ARRAY1) {(int nell1,int *bpws),
                                  (int nell2,int *ells),
                                  (int nfields,int *spin_arr)};
%apply (int DIM1,int DIM2,double *IN_ARRAY2) {(int nmap_2,int npix_2,double *mps),
                                              (int ncl1  ,int nell1 ,double *cls1),
                                              (int ncl2  ,int nell2 ,double *cls2),
                                              (int ncl3  ,int nell3 ,double *cls3)};
%apply (int DIM1,int DIM2,int DIM3,double *IN_ARRAY3) {(int ntmp_3,int nmap_3,int npix_3,double *tmp)};

%{
void asserting(int expression)
{
  if(!expression)
    report_error(NMT_ERROR_INCONSISTENT,"Passing inconsistent arguments from python\n");
}
%}

%exception {
  try {
    $action
      }
  finally {
    SWIG_exception(SWIG_RuntimeError,nmt_error_message);
  }
 }


%inline %{
void get_nell_list(nmt_binning_scheme *bins,int *iout,int niout)
{
  asserting(bins->n_bands==niout);

  memcpy(iout,bins->nell_list,bins->n_bands*sizeof(int));
}

int get_nell(nmt_binning_scheme *bins,int ibin)
{
  asserting(ibin<bins->n_bands);
  
  return bins->nell_list[ibin];
}

void get_ell_list(nmt_binning_scheme *bins,int ibin,int *iout,int niout)
{
  asserting(ibin<bins->n_bands);
  asserting(bins->nell_list[ibin]==niout);

  memcpy(iout,bins->ell_list[ibin],bins->nell_list[ibin]*sizeof(int));
}

void get_weight_list(nmt_binning_scheme *bins,int ibin,double *dout,int ndout)
{
  asserting(ibin<bins->n_bands);
  asserting(bins->nell_list[ibin]==ndout);

  memcpy(dout,bins->w_list[ibin],bins->nell_list[ibin]*sizeof(double));
}

void get_ell_eff(nmt_binning_scheme *bins,double *dout,int ndout)
{
  asserting(ndout==bins->n_bands);
  nmt_ell_eff(bins,dout);
}

void get_ell_eff_flat(nmt_binning_scheme_flat *bins,double *dout,int ndout)
{
  asserting(ndout==bins->n_bands);
  nmt_ell_eff_flat(bins,dout);
}

nmt_binning_scheme *bins_create_py(int nell1,int *bpws,
				   int nell2,int *ells,
				   int nell3,double *weights,
				   int lmax)
{
  asserting(nell1==nell2);
  asserting(nell2==nell3);
  
  return nmt_bins_create(nell1,bpws,ells,weights,lmax);
}

void update_mcm(nmt_workspace *w,int n_rows,int nell3,double *weights)
{
  asserting(nell3==n_rows*n_rows);

  nmt_update_coupling_matrix(w,n_rows,weights);
}

void get_mcm(nmt_workspace *w,double *dout,int ndout)
{
  int ii,nrows=(w->lmax+1)*w->ncls;

  for(ii=0;ii<nrows;ii++) {
    int jj;
    for(jj=0;jj<nrows;jj++) {
      long index=(long)(ii*nrows)+jj;
      dout[index]=w->coupling_matrix_unbinned[ii][jj];
    }
  }
}

nmt_binning_scheme_flat *bins_flat_create_py(int npix_1,double *mask,
					     int nell3,double *weights)
{
  asserting(npix_1==nell3);
  
  return nmt_bins_flat_create(nell3,mask,weights);
}

void bin_cl(nmt_binning_scheme *bins,
	    int ncl1,int nell1,double *cls1,
	    double *dout,int ndout)
{
  int i;
  asserting(ndout==ncl1*bins->n_bands);
  double **cls_in,**cls_out;
  cls_in=malloc(ncl1*sizeof(double *));
  cls_out=malloc(ncl1*sizeof(double *));
  for(i=0;i<ncl1;i++) {
    cls_in[i]=&(cls1[i*nell1]);
    cls_out[i]=&(dout[i*bins->n_bands]);
  }
  nmt_bin_cls(bins,cls_in,cls_out,ncl1);
  free(cls_in);
  free(cls_out);
}

void bin_cl_flat(nmt_binning_scheme_flat *bins,
		 int nell3,double *weights,
		 int ncl1,int nell1,double *cls1,
		 double *dout,int ndout)
{
  int i;
  asserting(nell3==nell1);
  asserting(ndout==ncl1*bins->n_bands);
  double **cls_in,**cls_out;
  cls_in=malloc(ncl1*sizeof(double *));
  cls_out=malloc(ncl1*sizeof(double *));
  for(i=0;i<ncl1;i++) {
    cls_in[i]=&(cls1[i*nell1]);
    cls_out[i]=&(dout[i*bins->n_bands]);
  }
  nmt_bin_cls_flat(bins,nell3,weights,cls_in,cls_out,ncl1);
  free(cls_in);
  free(cls_out);
}

void unbin_cl(nmt_binning_scheme *bins,
	      int ncl1,int nell1,double *cls1,
	      double *dout,int ndout)
{
  int i;
  int nellout=ndout/ncl1;
  asserting(nell1==bins->n_bands);
  double **cls_in,**cls_out;
  cls_in=malloc(ncl1*sizeof(double *));
  cls_out=malloc(ncl1*sizeof(double *));
  for(i=0;i<ncl1;i++) {
    cls_in[i]=&(cls1[i*nell1]);
    cls_out[i]=&(dout[i*nellout]);
    memset(cls_out[i],0,nellout*sizeof(double));
  }
  nmt_unbin_cls(bins,cls_in,cls_out,ncl1);
  free(cls_in);
  free(cls_out);
}

void unbin_cl_flat(nmt_binning_scheme_flat *bins,
		   int ncl1,int nell1,double *cls1,
		   int nell3,double *weights,
		   double *dout,int ndout)
{
  int i;
  asserting(ndout==nell3*ncl1);
  asserting(nell1==bins->n_bands);
  double **cls_in,**cls_out;
  cls_in=malloc(ncl1*sizeof(double *));
  cls_out=malloc(ncl1*sizeof(double *));
  for(i=0;i<ncl1;i++) {
    cls_in[i]=&(cls1[i*nell1]);
    cls_out[i]=&(dout[i*nell3]);
    memset(cls_out[i],0,nell3*sizeof(double));
  }
  nmt_unbin_cls_flat(bins,cls_in,nell3,weights,cls_out,ncl1);
  free(cls_in);
  free(cls_out);
}

nmt_field *field_alloc_new(int npix_1,double *mask,
			   int nmap_2,int npix_2,double *mps,
			   int ntmp_3,int nmap_3,int npix_3,double *tmp,
			   int nell3,double *weights,
			   int pure_e,int pure_b,int n_iter_mask_purify,double tol_pinv)
{
  int ii,jj;
  long nside=1;
  int pol=0,ntemp=0;
  double **maps;
  double ***temp=NULL;
  nmt_field *fl;
  asserting(npix_1==npix_2);
  asserting(npix_2==npix_3);
  asserting(nmap_2==nmap_3);
  asserting((nmap_2==1) || (nmap_2==2));

  while(npix_1!=12*nside*nside) {
    asserting(nside<=65536);
    nside*=2;
  }
  
  asserting(nell3==3*nside);

  if(nmap_2==2) pol=1;

  if(tmp!=NULL) {
    ntemp=ntmp_3;
    temp=malloc(ntmp_3*sizeof(double **));
    for(ii=0;ii<ntmp_3;ii++) {
      temp[ii]=malloc(nmap_3*sizeof(double *));
      for(jj=0;jj<nmap_3;jj++)
	temp[ii][jj]=tmp+npix_3*(jj+ii*nmap_3);
    }
  }

  maps=malloc(nmap_2*sizeof(double *));
  for(ii=0;ii<nmap_2;ii++)
    maps[ii]=mps+npix_2*ii;

  fl=nmt_field_alloc_sph(nside,mask,pol,maps,ntemp,temp,weights,pure_e,pure_b,
			 n_iter_mask_purify,tol_pinv);

  if(tmp!=NULL) {
    for(ii=0;ii<ntmp_3;ii++)
      free(temp[ii]);
    free(temp);
  }
  free(maps);

  return fl;
}

nmt_field *field_alloc_new_notemp(int npix_1,double *mask,
				  int nmap_2,int npix_2,double *mps,
				  int nell3,double *weights,
				  int pure_e,int pure_b,int n_iter_mask_purify)
{
  int ii;
  long nside=1;
  int pol=0,ntemp=0;
  double **maps;
  nmt_field *fl;
  asserting(npix_1==npix_2);
  asserting((nmap_2==1) || (nmap_2==2));

  while(npix_1!=12*nside*nside) {
    asserting(nside<=65536);
    nside*=2;
  }
  
  asserting(nell3==3*nside);

  if(nmap_2==2) pol=1;

  maps=malloc(nmap_2*sizeof(double *));
  for(ii=0;ii<nmap_2;ii++)
    maps[ii]=mps+npix_2*ii;

  fl=nmt_field_alloc_sph(nside,mask,pol,maps,ntemp,NULL,weights,pure_e,pure_b,n_iter_mask_purify,0.);

  free(maps);

  return fl;
}

nmt_field_flat *field_alloc_new_flat(int nx,int ny,double lx,double ly,
				     int npix_1,double *mask,
				     int nmap_2,int npix_2,double *mps,
				     int ntmp_3,int nmap_3,int npix_3,double *tmp,
				     int ncl1,int nell1,double *cls1,
				     int pure_e,int pure_b,double tol_pinv)
{
  int ii,jj;
  int pol=0,ntemp=0;
  double **maps;
  double ***temp=NULL;
  nmt_field_flat *fl;
  asserting(npix_1==npix_2);
  asserting((nmap_2==1) || (nmap_2==2));
  asserting(npix_1==nx*ny);
  asserting(ncl1==2);
  asserting(lx>0);
  asserting(ly>0);

  if(nmap_2==2) pol=1;

  if(tmp!=NULL) {
    asserting(npix_2==npix_3);
    asserting(nmap_2==nmap_3);
    ntemp=ntmp_3;
    temp=malloc(ntmp_3*sizeof(double **));
    for(ii=0;ii<ntmp_3;ii++) {
      temp[ii]=malloc(nmap_3*sizeof(double *));
      for(jj=0;jj<nmap_3;jj++)
	temp[ii][jj]=tmp+npix_3*(jj+ii*nmap_3);
    }
  }
  
  maps=malloc(nmap_2*sizeof(double *));
  for(ii=0;ii<nmap_2;ii++)
    maps[ii]=mps+npix_2*ii;

  double *larr,*beam;
  if((nell1==1) && (cls1[0]<0) && (cls1[1]<0)) {
    larr=NULL; beam=NULL;
  }
  else {
    larr=&(cls1[0]);
    beam=&(cls1[nell1]);
  }

  fl=nmt_field_flat_alloc(nx,ny,lx,ly,mask,pol,maps,ntemp,temp,
			  nell1,larr,beam,pure_e,pure_b,tol_pinv);

  if(tmp!=NULL) {
    for(ii=0;ii<ntmp_3;ii++)
      free(temp[ii]);
    free(temp);
  }
  free(maps);

  return fl;
}

nmt_field_flat *field_alloc_new_notemp_flat(int nx,int ny,double lx,double ly,
					    int npix_1,double *mask,
					    int nmap_2,int npix_2,double *mps,
					    int ncl1,int nell1,double *cls1,
					    int pure_e,int pure_b)
{
  asserting(lx>0);
  asserting(ly>0);
  return field_alloc_new_flat(nx,ny,lx,ly,npix_1,mask,nmap_2,npix_2,mps,
			      -1,-1,-1,NULL,ncl1,nell1,cls1,pure_e,pure_b,0.);
}

void get_map(nmt_field *fl,int imap,double *ldout,long nldout)
{
  asserting(imap<fl->nmaps);
  asserting(nldout==fl->npix);
  memcpy(ldout,fl->maps[imap],fl->npix*sizeof(double));
}

void get_map_flat(nmt_field_flat *fl,int imap,double *dout,int ndout)
{
  asserting(imap<fl->nmaps);
  asserting(ndout==fl->npix);
  memcpy(dout,fl->maps[imap],fl->npix*sizeof(double));
}

void get_temp(nmt_field *fl,int itemp,int imap,double *ldout,long nldout)
{
  asserting(itemp<fl->ntemp);
  asserting(imap<fl->nmaps);
  asserting(nldout==fl->npix);
  memcpy(ldout,fl->temp[itemp][imap],fl->npix*sizeof(double));
}

void get_temp_flat(nmt_field_flat *fl,int itemp,int imap,double *dout,int ndout)
{
  asserting(itemp<fl->ntemp);
  asserting(imap<fl->nmaps);
  asserting(ndout==fl->npix);
  memcpy(dout,fl->temp[itemp][imap],fl->npix*sizeof(double));
}

void apomask(int npix_1,double *mask,
	     double *ldout,long nldout,double aposize,char *apotype)
{
  long nside=1;
  asserting(nldout==npix_1);

  while(npix_1!=12*nside*nside) {
    asserting(nside<=65536);
    nside*=2;
  }

  nmt_apodize_mask(nside,mask,ldout,aposize,apotype);
}

void apomask_flat(int nx,int ny,double lx,double ly,
		  int npix_1,double *mask,
		  double *dout,int ndout,double aposize,char *apotype)
{
  asserting(lx>0);
  asserting(ly>0);
  asserting(npix_1==nx*ny);
  asserting(ndout==npix_1);

  nmt_apodize_mask_flat(nx,ny,lx,ly,mask,dout,aposize,apotype);
}

void synfast_new(int nside,
		 int nfields,int *spin_arr,
		 int seed,
		 int ncl1,int nell1,double *cls1,
		 int ncl2,int nell2,double *cls2,
		 double* ldout,long nldout)
{
  int ii,icl,nmaps=0;
  long npix=12*nside*nside;
  double **cls,**beams,**maps;

  for(ii=0;ii<nfields;ii++) {
    if(spin_arr[ii]==0)
      nmaps+=1;
    else if(spin_arr[ii]==2)
      nmaps+=2;
  }

  asserting(ncl2==nfields);
  asserting(ncl1==(nmaps*(nmaps+1))/2);
  asserting(nell1==nell2);

  cls=malloc(ncl1*sizeof(double *));
  for(icl=0;icl<ncl1;icl++)
    cls[icl]=cls1+nell1*icl;

  beams=malloc(nfields*sizeof(double *));
  for(icl=0;icl<nfields;icl++)
    beams[icl]=cls2+nell2*icl;

  maps=nmt_synfast_sph(nside,nfields,spin_arr,nell1-1,cls,beams,seed);

  for(icl=0;icl<nmaps;icl++) {
    memcpy(&(ldout[npix*icl]),maps[icl],npix*sizeof(double));
    free(maps[icl]);
  }
  free(maps);
  free(beams);
  free(cls);
}

void synfast_new_flat(int nx,int ny,double lx,double ly,
		      int nfields,int *spin_arr,
		      int seed,
		      int ncl1,int nell1,double *cls1,
		      int ncl2,int nell2,double *cls2,
		      double* dout,int ndout)
{
  int ii,icl,nmaps=0;
  long npix=nx*ny;
  double *larr;
  double **cls,**beams,**maps;

  for(ii=0;ii<nfields;ii++) {
    if(spin_arr[ii]==0)
      nmaps+=1;
    else if(spin_arr[ii]==2)
      nmaps+=2;
  }
  asserting(lx>0);
  asserting(ly>0);
  
  asserting(ncl2==nfields);
  asserting(ncl1==(nmaps*(nmaps+1))/2);
  asserting(nell1==nell2);
  
  cls=malloc(ncl1*sizeof(double *));
  for(icl=0;icl<ncl1;icl++)
    cls[icl]=cls1+nell1*icl;

  beams=malloc(nfields*sizeof(double *));
  for(icl=0;icl<nfields;icl++)
    beams[icl]=cls2+nell2*icl;

  larr=malloc(nell1*sizeof(double));
  for(ii=0;ii<nell1;ii++)
    larr[ii]=ii;

  maps=nmt_synfast_flat(nx,ny,lx,ly,nfields,spin_arr,
			nell1,larr,beams,nell1,larr,cls,seed);

  for(icl=0;icl<nmaps;icl++) {
    for(ii=0;ii<npix;ii++)
      dout[npix*icl+ii]=maps[icl][ii];
    dftw_free(maps[icl]);
  }
  free(maps);
  free(beams);
  free(cls);
  free(larr);
}

 nmt_workspace *comp_coupling_matrix(nmt_field *fl1,nmt_field *fl2,nmt_binning_scheme *bin,int is_teb)
{
  return nmt_compute_coupling_matrix(fl1,fl2,bin,is_teb);
}

nmt_workspace_flat *comp_coupling_matrix_flat(nmt_field_flat *fl1,nmt_field_flat *fl2,
					      nmt_binning_scheme_flat *bin,
					      double lmn_x,double lmx_x,double lmn_y,double lmx_y,
					      int is_teb)
{
  return nmt_compute_coupling_matrix_flat(fl1,fl2,bin,lmn_x,lmx_x,lmn_y,lmx_y,is_teb);
}

nmt_workspace *read_workspace(char *fname)
{
  return nmt_workspace_read(fname);
}

void write_workspace(nmt_workspace *w,char *fname)
{
  nmt_workspace_write(w,fname);
}

nmt_workspace_flat *read_workspace_flat(char *fname)
{
  return nmt_workspace_flat_read(fname);
}

void write_workspace_flat(nmt_workspace_flat *w,char *fname)
{
  nmt_workspace_flat_write(w,fname);
}
   
void comp_uncorr_noise_deproj_bias(nmt_field *fl1,
				   int npix_1,double *mask,
				   double *dout,int ndout)
{
  int i;
  double **cl_bias;
  int n_cl1=fl1->nmaps*fl1->nmaps;
  int n_ell1=fl1->lmax+1;
  asserting(npix_1==fl1->npix);
  asserting(ndout==n_ell1*n_cl1);
  cl_bias=malloc(n_cl1*sizeof(double *));
  for(i=0;i<n_cl1;i++)
    cl_bias[i]=&(dout[n_ell1*i]);

  nmt_compute_uncorr_noise_deprojection_bias(fl1,mask,cl_bias);

  free(cl_bias);
}

void comp_deproj_bias(nmt_field *fl1,nmt_field *fl2,
		      int ncl1,int nell1,double *cls1,
		      double *dout,int ndout)
{
  int i;
  double **cl_bias,**cl_guess;
  asserting(ncl1==fl1->nmaps*fl2->nmaps);
  asserting(nell1==fl1->lmax+1);
  asserting(ndout==nell1*ncl1);
  cl_bias=malloc(ncl1*sizeof(double *));
  cl_guess=malloc(ncl1*sizeof(double *));
  for(i=0;i<ncl1;i++) {
    cl_guess[i]=&(cls1[nell1*i]);
    cl_bias[i]=&(dout[nell1*i]);
  }

  nmt_compute_deprojection_bias(fl1,fl2,cl_guess,cl_bias);

  free(cl_bias);
  free(cl_guess);
}

void comp_deproj_bias_flat(nmt_field_flat *fl1,nmt_field_flat *fl2,
			   nmt_binning_scheme_flat *bin,
			   flouble lmn_x,flouble lmx_x,flouble lmn_y,flouble lmx_y,
			   int nell3,double *weights,
			   int ncl1,int nell1,double *cls1,
			   double *dout,int ndout)
{
  int i;
  double **cl_bias,**cl_guess;
  asserting(ncl1==fl1->nmaps*fl2->nmaps);
  asserting(nell1==nell3);
  asserting(ndout==bin->n_bands*ncl1);
  cl_bias=malloc(ncl1*sizeof(double *));
  cl_guess=malloc(ncl1*sizeof(double *));
  for(i=0;i<ncl1;i++) {
    cl_guess[i]=&(cls1[nell1*i]);
    cl_bias[i]=&(dout[bin->n_bands*i]);
  }

  nmt_compute_deprojection_bias_flat(fl1,fl2,bin,lmn_x,lmx_x,lmn_y,lmx_y,nell3,weights,cl_guess,cl_bias);

  free(cl_bias);
  free(cl_guess);
}

void write_covar_workspace(nmt_covar_workspace *cw,char *fname)
{
  nmt_covar_workspace_write(cw,fname);
}

nmt_covar_workspace *read_covar_workspace(char *fname)
{
  return nmt_covar_workspace_read(fname);
}

nmt_covar_workspace *covar_workspace_init_py(nmt_workspace *wa,nmt_workspace *wb)
{
  return nmt_covar_workspace_init(wa,wb);
}

void write_covar_workspace_flat(nmt_covar_workspace_flat *cw,char *fname)
{
  nmt_covar_workspace_flat_write(cw,fname);
}

nmt_covar_workspace_flat *read_covar_workspace_flat(char *fname)
{
  return nmt_covar_workspace_flat_read(fname);
}

nmt_covar_workspace_flat *covar_workspace_flat_init_py(nmt_workspace_flat *wa,nmt_workspace_flat *wb)
{
  return nmt_covar_workspace_flat_init(wa,wb);
}

void comp_gaussian_covariance(nmt_covar_workspace *cw,
			      int nell11,double *c11,
			      int nell12,double *c12,
			      int nell21,double *c21,
			      int nell22,double *c22,
			      double *dout,int ndout)
{
  asserting(nell11==nell12);
  asserting(nell11==nell21);
  asserting(nell11==nell22);
  nmt_compute_gaussian_covariance(cw,c11,c12,c21,c22,dout);
}

void comp_gaussian_covariance_flat(nmt_covar_workspace_flat *cw,
				   int nell3,double *weights,
				   int nell11,double *c11,
				   int nell12,double *c12,
				   int nell21,double *c21,
				   int nell22,double *c22,
				   double *dout,int ndout)
{
  asserting(nell11==nell3);
  asserting(nell11==nell12);
  asserting(nell11==nell21);
  asserting(nell11==nell22);
  nmt_compute_gaussian_covariance_flat(cw,nell3,weights,c11,c12,c21,c22,dout);
}

void comp_pspec_coupled(nmt_field *fl1,nmt_field *fl2,
			double *dout,int ndout)
{
  int i;
  double **cl_out;
  asserting(ndout==fl1->nmaps*fl2->nmaps*(fl1->lmax+1));
  cl_out=malloc(fl1->nmaps*fl2->nmaps*sizeof(double *));
  for(i=0;i<fl1->nmaps*fl2->nmaps;i++)
    cl_out[i]=&(dout[i*(fl1->lmax+1)]);

  nmt_compute_coupled_cell(fl1,fl2,cl_out);

  free(cl_out);
}

void comp_pspec_coupled_flat(nmt_field_flat *fl1,nmt_field_flat *fl2,
			     nmt_binning_scheme_flat *bin,
			     double *dout,int ndout,
			     flouble lmn_x,flouble lmx_x,flouble lmn_y,flouble lmx_y)
{
  int i;
  double **cl_out;
  asserting(fl1->fs->nx==fl2->fs->nx);
  asserting(fl1->fs->ny==fl2->fs->ny);
  asserting(fl1->fs->lx==fl2->fs->lx);
  asserting(fl1->fs->ly==fl2->fs->ly);
  asserting(ndout==fl1->nmaps*fl2->nmaps*bin->n_bands);
  cl_out=malloc(fl1->nmaps*fl2->nmaps*sizeof(double *));
  for(i=0;i<fl1->nmaps*fl2->nmaps;i++)
    cl_out[i]=&(dout[i*bin->n_bands]);

  nmt_compute_coupled_cell_flat(fl1,fl2,bin,cl_out,lmn_x,lmx_x,lmn_y,lmx_y);

  free(cl_out);
}

void decouple_cell_py(nmt_workspace *w,
		      int ncl1,int nell1,double *cls1,
		      int ncl2,int nell2,double *cls2,
		      int ncl3,int nell3,double *cls3,
		      double *dout,int ndout)
{
  int i;
  double **cl_in,**cl_noise,**cl_bias,**cl_out;
  asserting(ncl1==ncl2);
  asserting(ncl2==ncl3);
  asserting(ncl1==w->ncls);
  asserting(nell1==nell2);
  asserting(nell2==nell3);
  asserting(nell1==w->lmax+1);
  asserting(ndout==w->bin->n_bands*ncl1);
  cl_in=   malloc(ncl1*sizeof(double *));
  cl_noise=malloc(ncl2*sizeof(double *));
  cl_bias= malloc(ncl3*sizeof(double *));
  cl_out=  malloc(ncl1*sizeof(double *));
  for(i=0;i<ncl1;i++) {
    cl_in[i]   =&(cls1[i*nell1]);
    cl_noise[i]=&(cls2[i*nell2]);
    cl_bias[i] =&(cls3[i*nell3]);
    cl_out[i]  =&(dout[i*w->bin->n_bands]);
  }

  nmt_decouple_cl_l(w,cl_in,cl_noise,cl_bias,cl_out);

  free(cl_in);
  free(cl_noise);
  free(cl_bias);
  free(cl_out);
}

void decouple_cell_py_flat(nmt_workspace_flat *w,
			   int ncl1,int nell1,double *cls1,
			   int ncl2,int nell2,double *cls2,
			   int ncl3,int nell3,double *cls3,
			   double *dout,int ndout)
{
  int i;
  double **cl_in,**cl_noise,**cl_bias,**cl_out;
  asserting(ncl1==ncl2);
  asserting(ncl2==ncl3);
  asserting(ncl1==w->ncls);
  asserting(nell1==nell2);
  asserting(nell2==nell3);
  asserting(nell1==w->bin->n_bands);
  asserting(ndout==w->bin->n_bands*ncl1);
  cl_in=   malloc(ncl1*sizeof(double *));
  cl_noise=malloc(ncl2*sizeof(double *));
  cl_bias= malloc(ncl3*sizeof(double *));
  cl_out=  malloc(ncl1*sizeof(double *));
  for(i=0;i<ncl1;i++) {
    cl_in[i]   =&(cls1[i*nell1]);
    cl_noise[i]=&(cls2[i*nell2]);
    cl_bias[i] =&(cls3[i*nell3]);
    cl_out[i]  =&(dout[i*w->bin->n_bands]);
  }

  nmt_decouple_cl_l_flat(w,cl_in,cl_noise,cl_bias,cl_out);

  free(cl_in);
  free(cl_noise);
  free(cl_bias);
  free(cl_out);
}

void couple_cell_py(nmt_workspace *w,
		    int ncl1,int nell1,double *cls1,
		    double *dout,int ndout)
{
  int i;
  double **cl_in,**cl_out;
  asserting(ncl1==w->ncls);
  asserting(nell1==w->lmax+1);
  asserting(ncl1*nell1==ndout);
  cl_in=malloc(ncl1*sizeof(double *));
  cl_out=malloc(ncl1*sizeof(double *));
  for(i=0;i<ncl1;i++) {
    cl_in[i]=&(cls1[i*nell1]);
    cl_out[i]=&(dout[i*nell1]);
  }
  nmt_couple_cl_l(w,cl_in,cl_out);
  free(cl_in);
  free(cl_out);
}

void couple_cell_py_flat(nmt_workspace_flat *w,
			 int nell3,double *weights,
			 int ncl1,int nell1,double *cls1,
			 double *dout,int ndout)
{
  int i;
  double **cl_in,**cl_out;
  asserting(nell3==nell1);
  asserting(ncl1==w->ncls);
  asserting(ncl1*w->bin->n_bands==ndout);
  cl_in=malloc(ncl1*sizeof(double *));
  cl_out=malloc(ncl1*sizeof(double *));
  for(i=0;i<ncl1;i++) {
    cl_in[i]=&(cls1[i*nell1]);
    cl_out[i]=&(dout[i*w->bin->n_bands]);
  }
  nmt_couple_cl_l_flat_fast(w,nell3,weights,cl_in,cl_out);
  free(cl_in);
  free(cl_out);
}

void comp_pspec(nmt_field *fl1,nmt_field *fl2,
		nmt_binning_scheme *bin,nmt_workspace *w0,
		int ncl1,int nell1,double *cls1,
		int ncl2,int nell2,double *cls2,
		double *dout,int ndout)
{
  int i;
  double **cl_noise,**cl_guess,**cl_out;
  nmt_workspace *w;
  asserting(fl1->nside==fl2->nside);
  asserting(ncl1==fl1->nmaps*fl2->nmaps);
  asserting(nell1==fl1->lmax+1);
  asserting(ndout==bin->n_bands*ncl1);
  asserting(nell1==nell2);
  asserting(ncl1==ncl2);
  cl_noise=malloc(ncl1*sizeof(double *));
  cl_guess=malloc(ncl1*sizeof(double *));
  cl_out=malloc(ncl1*sizeof(double *));
  for(i=0;i<ncl1;i++) {
    cl_noise[i]=&(cls1[nell1*i]);
    cl_guess[i]=&(cls2[nell1*i]);
    cl_out[i]=&(dout[i*bin->n_bands]);
  }

  w=nmt_compute_power_spectra(fl1,fl2,bin,w0,cl_noise,cl_guess,cl_out);

  free(cl_out);
  free(cl_guess);
  free(cl_noise);
  if(w0==NULL)
    nmt_workspace_free(w);
}

void comp_pspec_flat(nmt_field_flat *fl1,nmt_field_flat *fl2,
		     nmt_binning_scheme_flat *bin,nmt_workspace_flat *w0,
		     int ncl1,int nell1,double *cls1,
		     int nell3,double *weights,
		     int ncl2,int nell2,double *cls2,
		     double *dout,int ndout,
		     flouble lmn_x,flouble lmx_x,flouble lmn_y,flouble lmx_y)
{
  int i;
  double **cl_noise,**cl_guess,**cl_out;
  nmt_workspace_flat *w;
  asserting(ncl1==fl1->nmaps*fl2->nmaps);
  asserting(nell1==bin->n_bands);
  asserting(ndout==bin->n_bands*ncl1);
  asserting(nell3==nell2);
  asserting(ncl1==ncl2);
  cl_noise=malloc(ncl1*sizeof(double *));
  cl_guess=malloc(ncl1*sizeof(double *));
  cl_out=malloc(ncl1*sizeof(double *));
  for(i=0;i<ncl1;i++) {
    cl_noise[i]=&(cls1[nell1*i]);
    cl_guess[i]=&(cls2[nell3*i]);
    cl_out[i]=&(dout[i*bin->n_bands]);
  }

  w=nmt_compute_power_spectra_flat(fl1,fl2,bin,lmn_x,lmx_x,lmn_y,lmx_y,
				   w0,cl_noise,nell3,weights,cl_guess,cl_out);

  free(cl_out);
  free(cl_guess);
  free(cl_noise);
  if(w0==NULL)
    nmt_workspace_flat_free(w);
}
%}
