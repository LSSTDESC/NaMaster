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
     (int nmcm_in,double *mcm_in),
     (int nell3,double *weights),
     (int nell4,double *f_ell),
     (int nlb1,double *beam1),
     (int nlb2,double *beam2),
     (int nl00,double *fl00),
     (int nl0e,double *fl0e),
     (int nl0b,double *fl0b),
     (int nle0,double *fle0),
     (int nlb0,double *flb0),
     (int nlee,double *flee),
     (int nleb,double *fleb),
     (int nlbe,double *flbe),
     (int nlbb,double *flbb),
     (int n00_1122,double *xi00_1122),
     (int n00_1221,double *xi00_1221),
     (int n02_1122,double *xi02_1122),
     (int n02_1221,double *xi02_1221),
     (int n22p_1122,double *xi22p_1122),
     (int n22p_1221,double *xi22p_1221),
     (int n22m_1122,double *xi22m_1122),
     (int n22m_1221,double *xi22m_1221)
     };
%apply (int DIM1,int *IN_ARRAY1) {(int nell1,int *bpws),
                                  (int nell2,int *ells),
                                  (int nfields,int *spin_arr)};
%apply (int DIM1,int DIM2,double *IN_ARRAY2) {(int nmap_2,int npix_2,double *mps),
                                              (int ncl11 ,int nell11,double *c11),
                                              (int ncl12 ,int nell12,double *c12),
                                              (int ncl21 ,int nell21,double *c21),
                                              (int ncl22 ,int nell22,double *c22),
                                              (int ncl1  ,int nell1 ,double *cls1),
                                              (int ncl2  ,int nell2 ,double *cls2),
                                              (int ncl3  ,int nell3 ,double *cls3),
                                              (int nl1   ,int ncell1,double *cell1)};
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

void get_xis(int lmax, int lmax_mask,
	     int ncl1, int nell1, double *cls1,
	     int s1, int s2, int pure_any,
	     int do_teb, int l_toeplitz, int l_exact,
	     int dl_band, double *ldout, long nldout)
{
  int imask,ipure,ll,nmask=ncl1;
  double **pcl_masks=my_malloc(nmask*sizeof(double *));

  asserting(nell1==lmax_mask+1);

  for(imask=0;imask<nmask;imask++)
    pcl_masks[imask]=&(cls1[imask*nell1]);

  nmt_master_calculator *c;
  c=nmt_compute_master_coefficients(lmax, lmax_mask, nmask, pcl_masks,
				    s1, s2, pure_any, do_teb,
				    l_toeplitz, l_exact, dl_band);
  long nout=0,nls=lmax+1;
  if(c->has_00)
    nout+=nmask*nls*nls;
  if(c->has_0s)
    nout+=nmask*c->npure_0s*nls*nls;
  if(c->has_ss)
    nout+=2*nmask*c->npure_ss*nls*nls;
  asserting(nout==nldout);

  long ind_sofar=0;
  if(c->has_00) {
    for(imask=0;imask<nmask;imask++) {
      long indmask=nls*nls*imask;
      for(ll=0;ll<=lmax;ll++) {
	memcpy(&(ldout[ind_sofar+indmask+nls*ll]),
	       c->xi_00[imask][ll],
	       nls*sizeof(double));
      }
    }
    ind_sofar+=nmask*nls*nls;
  }

  if(c->has_0s) {
    for(ipure=0;ipure<c->npure_0s;ipure++) {
      long indpure=nls*nls*nmask*ipure;
      for(imask=0;imask<nmask;imask++) {
	long indmask=nls*nls*imask;
	for(ll=0;ll<=lmax;ll++) {
	  memcpy(&(ldout[ind_sofar+indpure+indmask+nls*ll]),
		 c->xi_0s[imask][ipure][ll],
		 nls*sizeof(double));
	}
      }
    }
    ind_sofar+=nmask*c->npure_0s*nls*nls;
  }

  if(c->has_ss) {
    for(ipure=0;ipure<c->npure_ss;ipure++) {
      long indpure=nls*nls*nmask*ipure;
      for(imask=0;imask<nmask;imask++) {
	long indmask=nls*nls*imask;
	for(ll=0;ll<=lmax;ll++) {
	  memcpy(&(ldout[ind_sofar+indpure+indmask+nls*ll]),
		 c->xi_pp[imask][ipure][ll],
		 nls*sizeof(double));
	}
      }
    }
    ind_sofar+=nmask*c->npure_ss*nls*nls;
    for(ipure=0;ipure<c->npure_ss;ipure++) {
      long indpure=nls*nls*nmask*ipure;
      for(imask=0;imask<nmask;imask++) {
	long indmask=nls*nls*imask;
	for(ll=0;ll<=lmax;ll++) {
	  memcpy(&(ldout[ind_sofar+indpure+indmask+nls*ll]),
		 c->xi_mm[imask][ipure][ll],
		 nls*sizeof(double));
	}
      }
    }
    ind_sofar+=nmask*c->npure_ss*nls*nls;
  }

  asserting(ind_sofar==nldout);

  nmt_master_calculator_free(c);
  free(pcl_masks);
}
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

void get_fell_list(nmt_binning_scheme *bins,int ibin,double *dout,int ndout)
{
  asserting(ibin<bins->n_bands);
  asserting(bins->nell_list[ibin]==ndout);

  memcpy(dout,bins->f_ell[ibin],bins->nell_list[ibin]*sizeof(double));
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
				   int nell4,double *f_ell,
				   int lmax)
{
  asserting(nell1==nell2);
  asserting(nell2==nell3);
  asserting(nell3==nell4);
  
  return nmt_bins_create(nell1,bpws,ells,weights,f_ell,lmax);
}

nmt_binning_scheme_flat *bins_flat_create_py(int npix_1,double *mask,
					     int nell3,double *weights)
{
  asserting(npix_1==nell3);
  
  return nmt_bins_flat_create(nell3,mask,weights);
}

void bin_cl(nmt_binning_scheme *bins,
	    int nl1,int ncell1,double *cell1,
	    double *dout,int ndout)
{
  asserting(ndout==ncell1*bins->n_bands);
  nmt_bin_cls(bins,ncell1,cell1,dout);
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
	      int nl1,int ncell1,double *cell1,
	      double *dout,int ndout)
{
  asserting(nl1==bins->n_bands);
  nmt_unbin_cls(bins,ncell1,cell1,dout);
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

void bin_mcmat_oneside(nmt_binning_scheme *bins,int ncl,
		       int nmcm_in,double *mcm_in,
		       int nlb1,double *beam1,
		       int nlb2,double *beam2,
		       double *dout,int ndout)
{
  asserting(nmcm_in==ncl*ncl*(bins->ell_max+1)*(bins->ell_max+1));
  asserting(ndout==ncl*ncl*(bins->ell_max+1)*bins->n_bands);
  nmt_bin_mcm_oneside(bins,ncl,mcm_in,dout,beam1,beam2);
}  

void bin_mcmat(nmt_binning_scheme *bins,int ncl,
	       int nmcm_in,double *mcm_in,
	       int norm_type,double w2,
	       int nlb1,double *beam1,
	       int nlb2,double *beam2,
	       double *dout,int ndout)
{
  asserting(nmcm_in==ncl*ncl*(bins->ell_max+1)*(bins->ell_max+1));
  asserting(ndout==ncl*ncl*bins->n_bands*bins->n_bands);
  nmt_bin_mcm(bins,ncl,mcm_in,dout,norm_type,w2,beam1,beam2);
}

nmt_field_flat *field_alloc_empty_flat(int nx,int ny,double lx,double ly,int spin,
                                       int npix_1,double *mask,
                                       int ncl1,int nell1,double *cls1,
                                       int pure_e,int pure_b)
{
  nmt_field_flat *fl;
  asserting(npix_1==nx*ny);
  asserting(ncl1==2);
  asserting(lx>0);
  asserting(ly>0);

  double *larr,*beam;
  if((nell1==1) && (cls1[0]<0) && (cls1[1]<0)) {
    larr=NULL; beam=NULL;
  }
  else {
    larr=&(cls1[0]);
    beam=&(cls1[nell1]);
  }

  fl=nmt_field_flat_alloc(nx,ny,lx,ly,mask,spin,NULL,0,NULL,
			  nell1,larr,beam,pure_e,pure_b,0,0,1,1);

  return fl;
}

nmt_field_flat *field_alloc_new_flat(int nx,int ny,double lx,double ly,int spin,
				     int npix_1,double *mask,
				     int nmap_2,int npix_2,double *mps,
				     int ntmp_3,int nmap_3,int npix_3,double *tmp,
				     int ncl1,int nell1,double *cls1,
				     int pure_e,int pure_b,double tol_pinv,
                                     int masked_input,int lite)
{
  int ii,jj;
  int ntemp=0;
  double **maps;
  double ***temp=NULL;
  nmt_field_flat *fl;
  asserting(npix_1==npix_2);
  asserting((nmap_2==1) || (nmap_2==2));
  asserting(npix_1==nx*ny);
  asserting(ncl1==2);
  asserting(lx>0);
  asserting(ly>0);

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

  fl=nmt_field_flat_alloc(nx,ny,lx,ly,mask,spin,maps,ntemp,temp,
			  nell1,larr,beam,pure_e,pure_b,tol_pinv,
                          masked_input,lite,0);

  if(tmp!=NULL) {
    for(ii=0;ii<ntmp_3;ii++)
      free(temp[ii]);
    free(temp);
  }
  free(maps);

  return fl;
}

nmt_field_flat *field_alloc_new_notemp_flat(int nx,int ny,double lx,double ly,int spin,
					    int npix_1,double *mask,
					    int nmap_2,int npix_2,double *mps,
					    int ncl1,int nell1,double *cls1,
					    int pure_e,int pure_b,
                                            int masked_input,int lite)
{
  asserting(lx>0);
  asserting(ly>0);
  return field_alloc_new_flat(nx,ny,lx,ly,spin,npix_1,mask,nmap_2,npix_2,mps,
			      -1,-1,-1,NULL,ncl1,nell1,cls1,pure_e,pure_b,0.,
                              masked_input,lite);
}

void get_mask_flat(nmt_field_flat *fl,double *dout,int ndout)
{
  asserting(ndout==fl->npix);
  memcpy(dout,fl->mask,fl->npix*sizeof(double));
}

void get_map_flat(nmt_field_flat *fl,int imap,double *dout,int ndout)
{
  asserting(imap<fl->nmaps);
  asserting(ndout==fl->npix);
  memcpy(dout,fl->maps[imap],fl->npix*sizeof(double));
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
    else
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

void comp_general_coupling_matrix(int s1, int s2, int n1, int n2,
				  int parity, int lmax,
				  int nell4,double *f_ell,
				  double *dout,int ndout)
{
  asserting(nell4==lmax+1);
  if(parity==2)
    asserting(ndout==2*nell4*nell4);
  else
    asserting(ndout==nell4*nell4);
  memset(dout,0,ndout*sizeof(double));
  nmt_compute_general_coupling_matrix(lmax,f_ell,s1,s2,n1,n2,parity,dout);
}

nmt_workspace_flat *comp_coupling_matrix_flat(nmt_field_flat *fl1,nmt_field_flat *fl2,
					      nmt_binning_scheme_flat *bin,
					      double lmn_x,double lmx_x,double lmn_y,double lmx_y,
					      int is_teb)
{
  return nmt_compute_coupling_matrix_flat(fl1,fl2,bin,lmn_x,lmx_x,lmn_y,lmx_y,is_teb);
}

nmt_workspace_flat *workspace_flat_from_data(int ncls, double lmax,
					     double lcut_x_i, double lcut_x_f,
					     double lcut_y_i, double lcut_y_f,
					     int pe1, int pe2, int pb1, int pb2, int is_teb,
					     int nell2, int *ells, // n_cells
					     int nx, int ny, long npix,  // fs_info
					     double lx, double ly, double pixsize,
					     double dell, double i_dell,
					     int nell4, double *f_ell, // fs_info.ell_min
					     int nlb1,double *beam1, // bin.l0
					     int nlb2,double *beam2, // bin.lf
					     int ncl11,int nell11,double *c11, //mcm
					     int ncl12,int nell12,double *c12, //mcm_binned
					     int ncl21,int nell21,double *c21, //mcm_binned_gsl
					     int nell1,int *bpws) //mcm_binned_gsl_perm
{
  int ii;
  asserting(nlb1==nell2);
  asserting(nlb1==nlb2);
  asserting(ncl11==nlb1*ncls);
  asserting(nell11==nell4*ncls);
  asserting(ncl12==nlb1*ncls);
  asserting(nell12==nlb1*ncls);
  asserting(ncl21==nlb1*ncls);
  asserting(nell21==nlb1*ncls);
  asserting(nell1==nlb1*ncls);

  nmt_workspace_flat *w=my_malloc(sizeof(nmt_workspace_flat));
  w->ncls=ncls;
  w->lmax=lmax;
  w->ellcut_x[0]=lcut_x_i;
  w->ellcut_x[1]=lcut_x_f;
  w->ellcut_y[0]=lcut_y_i;
  w->ellcut_y[1]=lcut_y_f;
  w->pe1=pe1;
  w->pe2=pe2;
  w->pb1=pb1;
  w->pb2=pb2;
  w->is_teb=is_teb;
  w->fs=my_malloc(sizeof(nmt_flatsky_info));
  w->fs->nx=nx;
  w->fs->ny=ny;
  w->fs->npix=npix;
  w->fs->lx=lx;
  w->fs->ly=ly;
  w->fs->pixsize=pixsize;
  w->fs->dell=dell;
  w->fs->i_dell=i_dell;
  w->fs->n_ell=nell4;
  w->fs->ell_min=my_malloc(sizeof(double)*w->fs->n_ell);
  memcpy(w->fs->ell_min,f_ell,w->fs->n_ell*sizeof(double));
  w->n_cells=my_malloc(sizeof(int)*nell4);
  memcpy(w->n_cells,ells,nell4*sizeof(int));
  w->bin=nmt_bins_flat_create(nlb1,beam1,beam2);
  w->coupling_matrix_unbinned=my_malloc(ncl11*sizeof(double *));
  for(ii=0;ii<ncl11;ii++) {
    w->coupling_matrix_unbinned[ii]=my_malloc(nell11*sizeof(double));
    memcpy(w->coupling_matrix_unbinned[ii], &(c11[ii*nell11]), nell11*sizeof(double));
  }
  w->coupling_matrix_binned=my_malloc(ncl12*sizeof(double *));
  for(ii=0;ii<ncl12;ii++) {
    w->coupling_matrix_binned[ii]=my_malloc(nell12*sizeof(double));
    memcpy(w->coupling_matrix_binned[ii], &(c12[ii*nell12]), nell12*sizeof(double));
  }
  w->coupling_matrix_binned_gsl=gsl_matrix_alloc(ncl21,nell21);
  for(ii=0;ii<ncl21;ii++) {
    long jj,i0=ii*nell21;
    for(jj=0;jj<nell21;jj++)
      gsl_matrix_set(w->coupling_matrix_binned_gsl,ii,jj,c21[i0+jj]);
  }
  w->coupling_matrix_perm=gsl_permutation_alloc(nell1);
  for(ii=0;ii<nell1;ii++)
    w->coupling_matrix_perm->data[ii]=bpws[ii];

  return w;     
}

void wsp_flat_get_n_cells(nmt_workspace_flat *w, int *iout, int niout)
{
  asserting(niout==w->bin->n_bands);
  memcpy(iout, w->n_cells, niout*sizeof(int));
}

void wsp_flat_get_mcm(nmt_workspace_flat *w,
		      int unbinned, int is_gsl,
		      double *ldout,long nldout)
{
  int ii;
  if(unbinned) {
    for(ii=0;ii<w->ncls*w->bin->n_bands;ii++) {
      memcpy(&(ldout[ii*w->ncls*w->fs->n_ell]),
	     w->coupling_matrix_unbinned[ii],
	     w->ncls*w->fs->n_ell*sizeof(double));
    }
  }
  else {
    if(is_gsl) {
      for(ii=0;ii<w->ncls*w->bin->n_bands;ii++) {
	int jj;
	long index0=ii*w->ncls*w->bin->n_bands;
	for(jj=0;jj<w->ncls*w->bin->n_bands;jj++)
	  ldout[index0+jj]=gsl_matrix_get(w->coupling_matrix_binned_gsl,ii,jj);
      }
    }
    else {
      for(ii=0;ii<w->ncls*w->bin->n_bands;ii++) {
	memcpy(&(ldout[ii*w->ncls*w->bin->n_bands]),
	       w->coupling_matrix_binned[ii],
	       w->ncls*w->bin->n_bands*sizeof(double));
      }
    }
  }
}

void wsp_flat_get_perm(nmt_workspace_flat *w,
		int *iout,int niout)
{
  int ii;
  for(ii=0;ii<w->ncls*w->bin->n_bands;ii++)
    iout[ii]=(int)(w->coupling_matrix_perm->data[ii]);
}

void wsp_flat_get_fs_ellmin(nmt_workspace_flat *w,
			    double *dout, int ndout)
{
  memcpy(dout,w->fs->ell_min,ndout*sizeof(double));
}

void wsp_flat_get_bin_ls(nmt_workspace_flat *w,
			 double *dout,int ndout)
{
  asserting(ndout==2*w->bin->n_bands);
  memcpy(dout,w->bin->ell_0_list,w->bin->n_bands*sizeof(double));
  memcpy(&(dout[w->bin->n_bands]),
	 w->bin->ell_f_list,w->bin->n_bands*sizeof(double));
}

void wsp_flat_get_lcuts(nmt_workspace_flat *w,
			double *dout,int ndout)
{
  asserting(ndout==4);
  dout[0]=w->ellcut_x[0];
  dout[1]=w->ellcut_x[1];
  dout[2]=w->ellcut_y[0];
  dout[3]=w->ellcut_y[1];
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

void write_covar_workspace_flat(nmt_covar_workspace_flat *cw,char *fname)
{
  nmt_covar_workspace_flat_write_fits(cw,fname);
}

nmt_covar_workspace_flat *read_covar_workspace_flat(char *fname)
{
  return nmt_covar_workspace_flat_read_fits(fname);
}

nmt_covar_workspace_flat *covar_workspace_flat_init_py(nmt_field_flat *fa1,nmt_field_flat *fa2,
						       nmt_binning_scheme_flat *ba,
						       nmt_field_flat *fb1,nmt_field_flat *fb2,
						       nmt_binning_scheme_flat *bb)
{
  return nmt_covar_workspace_flat_init(fa1,fa2,ba,fb1,fb2,bb);
}

void comp_gaussian_covariance_flat(nmt_covar_workspace_flat *cw,
				   int spin_a1,int spin_a2,int spin_b1,int spin_b2,
				   nmt_workspace_flat *wa,nmt_workspace_flat *wb,
				   int nell3,double *weights,
				   int ncl11,int nell11,double *c11,
				   int ncl12,int nell12,double *c12,
				   int ncl21,int nell21,double *c21,
				   int ncl22,int nell22,double *c22,
				   double *dout,int ndout)
{
  asserting(nell11==nell3);
  asserting(nell11==nell12);
  asserting(nell11==nell21);
  asserting(nell11==nell22);
  int i;
  double **c11p=malloc(ncl11*sizeof(double *));
  for(i=0;i<ncl11;i++)
    c11p[i]=&(c11[i*nell11]);
  double **c12p=malloc(ncl12*sizeof(double *));
  for(i=0;i<ncl12;i++)
    c12p[i]=&(c12[i*nell12]);
  double **c21p=malloc(ncl21*sizeof(double *));
  for(i=0;i<ncl21;i++)
    c21p[i]=&(c21[i*nell21]);
  double **c22p=malloc(ncl22*sizeof(double *));
  for(i=0;i<ncl22;i++)
    c22p[i]=&(c22[i*nell22]);
  nmt_compute_gaussian_covariance_flat(cw,spin_a1,spin_a2,spin_b1,spin_b2,wa,wb,
				       nell3,weights,c11p,c12p,c21p,c22p,dout);
  free(c11p); free(c12p); free(c21p); free(c22p);
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

void get_ell_sampling_flat(nmt_field_flat *f, double *dout, int ndout)
{
  int ii;
  asserting(ndout==f->fs->n_ell);
  for(ii=0;ii<f->fs->n_ell;ii++)
    dout[ii]=f->fs->ell_min[ii]+0.5*f->fs->dell;
}
%}
