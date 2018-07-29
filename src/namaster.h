#ifndef _NAMASTER_H_
#define _NAMASTER_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#include <omp.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_integration.h>
#include <fftw3.h>

#define NMT_MAX(a,b)  (((a)>(b)) ? (a) : (b)) // maximum
#define NMT_MIN(a,b)  (((a)<(b)) ? (a) : (b)) // minimum

#ifdef _SPREC
typedef float flouble;
typedef float complex fcomplex;
#else //_SPREC
typedef double flouble;
typedef double complex fcomplex;
#endif //_SPREC

//Defined in bins_flat.c
typedef struct {
  int n_bands;
  flouble *ell_0_list;
  flouble *ell_f_list;
} nmt_binning_scheme_flat;
nmt_binning_scheme_flat *nmt_bins_flat_constant(int nlb,flouble lmax);
nmt_binning_scheme_flat *nmt_bins_flat_create(int nell,flouble *l0,flouble *lf);
void nmt_bins_flat_free(nmt_binning_scheme_flat *bin);
void nmt_bin_cls_flat(nmt_binning_scheme_flat *bin,int nl,flouble *larr,flouble **cls_in,
		      flouble **cls_out,int ncls);
void nmt_unbin_cls_flat(nmt_binning_scheme_flat *bin,flouble **cls_in,
			int nl,flouble *larr,flouble **cls_out,int ncls);
void nmt_ell_eff_flat(nmt_binning_scheme_flat *bin,flouble *larr);
int nmt_bins_flat_search_fast(nmt_binning_scheme_flat *bin,flouble l,int il);

//Defined in bins.c
typedef struct {
  int n_bands;
  int *nell_list;
  int **ell_list;
  flouble **w_list;
  int ell_max;
} nmt_binning_scheme;
nmt_binning_scheme *nmt_bins_constant(int nlb,int lmax);
nmt_binning_scheme *nmt_bins_create(int nell,int *bpws,int *ells,flouble *weights,int lmax);
nmt_binning_scheme *nmt_bins_read(char *fname,int lmax);
void nmt_bins_free(nmt_binning_scheme *bin);
void nmt_bin_cls(nmt_binning_scheme *bin,flouble **cls_in,flouble **cls_out,int ncls);
void nmt_unbin_cls(nmt_binning_scheme *bin,flouble **cls_in,flouble **cls_out,int ncls);
void nmt_ell_eff(nmt_binning_scheme *bin,flouble *larr);

//Defined in field_flat.c
typedef struct {
  int is_const;
  int nk;
  flouble x0;
  flouble xf;
  flouble y0;
  flouble yf;
  gsl_spline *spl;
} nmt_k_function;
nmt_k_function *nmt_k_function_alloc(int nk,flouble *karr,flouble *farr,flouble y0,flouble yf,int is_const);
void nmt_k_function_free(nmt_k_function *f);
flouble nmt_k_function_eval(nmt_k_function *f,flouble k,gsl_interp_accel *intacc);

typedef struct {
  int nx;
  int ny;
  long npix;
  flouble lx;
  flouble ly;
  flouble pixsize;
  int n_ell;
  flouble i_dell;
  flouble dell;
  flouble *ell_min;
  //  int *n_cells;
} nmt_flatsky_info;
nmt_flatsky_info *nmt_flatsky_info_alloc(int nx,int ny,flouble lx,flouble ly);
void nmt_flatsky_info_free(nmt_flatsky_info *fs);

typedef struct {
  nmt_flatsky_info *fs;
  long npix;
  int pure_e;
  int pure_b;
  flouble *mask;
  fcomplex **a_mask;
  int pol;
  int nmaps;
  flouble **maps;
  fcomplex **alms;
  int ntemp;
  flouble ***temp;
  fcomplex ***a_temp;
  gsl_matrix *matrix_M;
  nmt_k_function *beam;
} nmt_field_flat;
void nmt_field_flat_free(nmt_field_flat *fl);
nmt_field_flat *nmt_field_flat_alloc(int nx,int ny,flouble lx,flouble ly,
				     flouble *mask,int pol,flouble **maps,int ntemp,flouble ***temp,
				     int nl_beam,flouble *l_beam,flouble *beam,
				     int pure_e,int pure_b,double tol_pinv);
flouble **nmt_synfast_flat(int nx,int ny,flouble lx,flouble ly,int nfields,int *spin_arr,
			   int nl_beam,flouble *l_beam,flouble **beam_fields,
			   int nl_cell,flouble *l_cell,flouble **cell_fields,
			   int seed);
void nmt_purify_flat(nmt_field_flat *fl,flouble *mask,fcomplex **walm0,
		     flouble **maps_in,flouble **maps_out,fcomplex **alms);

//Defined in field.c
typedef struct {
  long nside;
  long npix;
  int lmax;
  int pure_e;
  int pure_b;
  flouble *mask;
  fcomplex **a_mask;
  int pol;
  int nmaps;
  flouble **maps;
  fcomplex **alms;
  int ntemp;
  flouble ***temp;
  fcomplex ***a_temp;
  gsl_matrix *matrix_M;
  flouble *beam;
} nmt_field;
void nmt_field_free(nmt_field *fl);
nmt_field *nmt_field_alloc_sph(long nside,flouble *mask,int pol,flouble **maps,
			       int ntemp,flouble ***temp,flouble *beam,
			       int pure_e,int pure_b,int n_iter_mask_purify,double tol_pinv);
nmt_field *nmt_field_read(char *fname_mask,char *fname_maps,char *fname_temp,char *fname_beam,
			  int pol,int pure_e,int pure_b,int n_iter_mask_purify,double tol_pinv);
flouble **nmt_synfast_sph(int nside,int nfields,int *spin_arr,int lmax,
			  flouble **cells,flouble **beam_fields,int seed);
void nmt_purify(nmt_field *fl,flouble *mask,fcomplex **walm0,
		flouble **maps_in,flouble **maps_out,fcomplex **alms);

//Defined in mask.c
void nmt_apodize_mask(long nside,flouble *mask_in,flouble *mask_out,flouble aposize,char *apotype);

//Defined in mask.c
void nmt_apodize_mask_flat(int nx,int ny,flouble lx,flouble ly,
			   flouble *mask_in,flouble *mask_out,flouble aposize,char *apotype);

//Defined in master_flat.c
typedef struct {
  int ncls;
  flouble ellcut_x[2];
  flouble ellcut_y[2];
  int pe1;
  int pe2;
  int pb1;
  int pb2;
  nmt_flatsky_info *fs;
  flouble *mask1;
  flouble *mask2;
#ifdef _ENABLE_FLAT_THEORY_ACCURATE
  flouble *maskprod;
#endif //_ENABLE_FLAT_THEORY_ACCURATE
  //  flouble *pcl_masks;
  //  flouble *l_arr;
  //  int *i_band;
  int *n_cells;
  flouble **coupling_matrix_unbinned;
  flouble **coupling_matrix_binned;
  nmt_binning_scheme_flat *bin;
  flouble lmax;
  gsl_matrix *coupling_matrix_binned_gsl;
  gsl_permutation *coupling_matrix_perm;
} nmt_workspace_flat;
void nmt_workspace_flat_free(nmt_workspace_flat *w); //
nmt_workspace_flat *nmt_workspace_flat_read(char *fname);
void nmt_workspace_flat_write(nmt_workspace_flat *w,char *fname);
nmt_workspace_flat *nmt_compute_coupling_matrix_flat(nmt_field_flat *fl1,nmt_field_flat *fl2,
						     nmt_binning_scheme_flat *bin,
						     flouble lmn_x,flouble lmx_x,
						     flouble lmn_y,flouble lmx_y);
void nmt_compute_deprojection_bias_flat(nmt_field_flat *fl1,nmt_field_flat *fl2,
					nmt_binning_scheme_flat *bin,
					flouble lmn_x,flouble lmx_x,flouble lmn_y,flouble lmx_y,
					int nl_prop,flouble *l_prop,flouble **cl_proposal,
					flouble **cl_bias);
#ifdef _ENABLE_FLAT_THEORY_ACCURATE
void nmt_couple_cl_l_flat_accurate(nmt_workspace_flat *w,int nl,flouble *larr,flouble **cl_in,flouble **cl_out);
#endif //_ENABLE_FLAT_THEORY_ACCURATE
void nmt_couple_cl_l_flat_fast(nmt_workspace_flat *w,int nl,flouble *larr,flouble **cl_in,flouble **cl_out);
void nmt_couple_cl_l_flat_quick(nmt_workspace_flat *w,int nl,flouble *larr,flouble **cl_in,flouble **cl_out);
void nmt_decouple_cl_l_flat(nmt_workspace_flat *w,flouble **cl_in,flouble **cl_noise_in,
			    flouble **cl_bias,flouble **cl_out);
void nmt_compute_coupled_cell_flat(nmt_field_flat *fl1,nmt_field_flat *fl2,nmt_binning_scheme_flat *bin,flouble **cl_out,
				   flouble lmn_x,flouble lmx_x,flouble lmn_y,flouble lmx_y);
nmt_workspace_flat *nmt_compute_power_spectra_flat(nmt_field_flat *fl1,nmt_field_flat *fl2,
						   nmt_binning_scheme_flat *bin,
						   flouble lmn_x,flouble lmx_x,flouble lmn_y,flouble lmx_y,
						   nmt_workspace_flat *w0,flouble **cl_noise,
						   int nl_prop,flouble *l_prop,flouble **cl_prop,
						   flouble **cl_out);

//Defined in master.c
typedef struct {
  int lmax;
  int ncls;
  int nside;
  flouble *mask1;
  flouble *mask2;
  flouble *pcl_masks;
  flouble **coupling_matrix_unbinned;
  nmt_binning_scheme *bin;
  gsl_matrix *coupling_matrix_binned;
  gsl_permutation *coupling_matrix_perm;
} nmt_workspace;
nmt_workspace *nmt_compute_coupling_matrix(nmt_field *fl1,nmt_field *fl2,nmt_binning_scheme *bin);
void nmt_workspace_write(nmt_workspace *w,char *fname);
nmt_workspace *nmt_workspace_read(char *fname);
void nmt_workspace_free(nmt_workspace *w);
void nmt_compute_deprojection_bias(nmt_field *fl1,nmt_field *fl2,flouble **cl_proposal,flouble **cl_bias);
void nmt_compute_uncorr_noise_deprojection_bias(nmt_field *fl1,flouble *map_var,flouble **cl_bias);
void nmt_couple_cl_l(nmt_workspace *w,flouble **cl_in,flouble **cl_out);
void nmt_decouple_cl_l(nmt_workspace *w,flouble **cl_in,flouble **cl_noise_in,
		       flouble **cl_bias,flouble **cl_out);
void nmt_compute_coupled_cell(nmt_field *fl1,nmt_field *fl2,flouble **cl_out,int iter);
nmt_workspace *nmt_compute_power_spectra(nmt_field *fl1,nmt_field *fl2,
					 nmt_binning_scheme *bin,nmt_workspace *w0,
					 flouble **cl_noise,flouble **cl_proposal,flouble **cl_out);

//Defined in covar.c
typedef struct {
  int ncls_a;
  int ncls_b;
  nmt_binning_scheme_flat *bin;
  flouble **xi_1122;
  flouble **xi_1221;
  gsl_matrix *coupling_binned_a;
  gsl_matrix *coupling_binned_b;
  gsl_permutation *coupling_binned_perm_a;
  gsl_permutation *coupling_binned_perm_b;
} nmt_covar_workspace_flat;

void nmt_covar_workspace_flat_free(nmt_covar_workspace_flat *cw);
nmt_covar_workspace_flat *nmt_covar_workspace_flat_init(nmt_workspace_flat *wa,nmt_workspace_flat  *wb);
void nmt_compute_gaussian_covariance_flat(nmt_covar_workspace_flat *cw,
					  int nl,flouble *larr,flouble *cla1b1,flouble *cla1b2,
					  flouble *cla2b1,flouble *cla2b2,flouble *covar_out);
void nmt_covar_workspace_flat_write(nmt_covar_workspace_flat *cw,char *fnane);
nmt_covar_workspace_flat *nmt_covar_workspace_flat_read(char *fname);
  
typedef struct {
  int lmax_a;
  int lmax_b;
  int ncls_a;
  int ncls_b;
  nmt_binning_scheme *bin_a;
  nmt_binning_scheme *bin_b;
  int nside;
  flouble **xi_1122;
  flouble **xi_1221;
  gsl_matrix *coupling_binned_a;
  gsl_matrix *coupling_binned_b;
  gsl_permutation *coupling_binned_perm_a;
  gsl_permutation *coupling_binned_perm_b;
} nmt_covar_workspace;

void nmt_covar_workspace_free(nmt_covar_workspace *cw);
nmt_covar_workspace *nmt_covar_workspace_init(nmt_workspace *wa,nmt_workspace *wb);
void  nmt_compute_gaussian_covariance(nmt_covar_workspace *cw,
				      flouble *cla1b1,flouble *cla1b2,flouble *cla2b1,flouble *cla2b2,
				      flouble *covar_out);
void nmt_covar_workspace_write(nmt_covar_workspace *cw,char *fname);
nmt_covar_workspace *nmt_covar_workspace_read(char *fname);

#endif //_NAMASTER_H_
