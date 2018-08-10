#ifndef _NAMASTER_H_
#define _NAMASTER_H_

#ifndef NO_DOXY
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
#endif //NO_DOXY

#define NMT_MAX(a,b)  (((a)>(b)) ? (a) : (b)) // maximum
#define NMT_MIN(a,b)  (((a)<(b)) ? (a) : (b)) // minimum

#ifdef _SPREC
typedef float flouble;
typedef float complex fcomplex;
#else //_SPREC
typedef double flouble;
typedef double complex fcomplex;
#endif //_SPREC

/**
 * @brief Flat-sky bandpowers.
 *
 * This structure defines bandpowers for flat-sky power spectra.
 * These are currently defined only by band edges (assumed 
 * flat weights within band).
 */
typedef struct {
  int n_bands; //!< Number of bandpowers stored
  flouble *ell_0_list; //!< Lower edge of each bandpower
  flouble *ell_f_list; //!< Upper edge of each bandpower
} nmt_binning_scheme_flat;

/**
 * @brief nmt_binning_scheme_flat constructor for constant bandpowers
 *
 * nmt_binning_scheme_flat constructor for bandpowers with
 * constant width \p nlb, from ell = 2 to ell = \p lmax.
 * @param nlb Constant band width
 * @param lmax Maximum multipole
 * @return Allocated binning structure.
 */
nmt_binning_scheme_flat *nmt_bins_flat_constant(int nlb,flouble lmax);

/**
 * @brief nmt_binning_scheme_flat generic constructor.
 *
 * @param nell Number of bandpowers
 * @param l0 Lower edge of all bandpowers (should be allocated to nell elements).
 * @param lf Lower edge of all bandpowers (should be allocated to nell elements).
 * @return Allocated binning structure.
 */
nmt_binning_scheme_flat *nmt_bins_flat_create(int nell,flouble *l0,flouble *lf);

/**
 * @brief nmt_binning_scheme_flat destructor
 */
void nmt_bins_flat_free(nmt_binning_scheme_flat *bin);

/**
 * @brief Returns average of input power spectrum into bandpowers.
 *
 * @param bin nmt_binning_scheme_flat structure defining the bandpowers.
 * @param nl Number of elements in the input power spectra.
 * @param larr Array containing the \p nl multipoles at which the input power
 *        spectrum is defined.
 * @param cls_in Array of \p ncls input power spectra.
 * @param cls_out Array of \p ncls averaged output power spectra. 
 *        Should be allocated to the number of bandpowers defined bin \p.
 * @param ncls Number of input/output power spectra.
 */
void nmt_bin_cls_flat(nmt_binning_scheme_flat *bin,int nl,flouble *larr,flouble **cls_in,
		      flouble **cls_out,int ncls);

/**
 * @brief Returns binned power spectra interpolated into an given set of multipoles.
 * 
 * Nearest-neighbours interpolation is used.
 * @param bin nmt_binning_scheme_flat structure defining the bandpowers.
 * @param cls_in Array of \p ncls input power spectra. Must have the same number of 
 *        elements as bandpowers defined by \p bin.
 * @param nl Number of elements in the output power spectra.
 * @param larr Array containing the \p nl multipoles at which the output power 
 *        spectrum are requested.
 * @param cls_out Array of \p ncls interpolated output power spectra.
 * @param ncls Number of input/output power spectra.
 */
void nmt_unbin_cls_flat(nmt_binning_scheme_flat *bin,flouble **cls_in,
			int nl,flouble *larr,flouble **cls_out,int ncls);

/**
 * @brief Returns effective multipoles.
 * 
 * Returns the mid point of each bandpower defined in \p bin.
 * @param bin nmt_binning_scheme_flat structure defining the bandpowers.
 * @param larr Output array containing mid-points of the bandpowers.
 *        Should be preallocated to the correct number of bandpowers.
 */
void nmt_ell_eff_flat(nmt_binning_scheme_flat *bin,flouble *larr);

/**
 * @brief Fast bin-searching routine for flat-sky bandpowers
 *
 * Returns the bandpower index in which a given ell falls. The functions is designed
 * to be fast if a good guess for the bandpower index is supplied. A typical use would
 * be to iterate over ell values and pass, as a guess index, the index found in the
 * previous iteration.
 * @param bin nmt_binning_scheme_flat structure defining the bandpowers.
 * @param l Multipole for which you want the bandpower index.
 * @param il Guessed bandpower index.
 * @return Bandpower index.
 */
int nmt_bins_flat_search_fast(nmt_binning_scheme_flat *bin,flouble l,int il);

/**
 * @brief Full-sky bandpowers.
 *
 * This structure defines bandpowers for full-sky power spectra.
 * Although a given multipole ell can only contribute to one bandpower, 
 * the distribution of ells per bandpower and their relative weights
 * is left completely free.
 */
typedef struct {
  int n_bands; //!< Number of bandpowers.
  int *nell_list; //!< Number of multipoles belonging to each bandpower.
  int **ell_list; //!< List of multipoles in each bandpowers.
  flouble **w_list; //!< List of weights associated to each multipole in \p ell_list.
  int ell_max; //!< Maximum multipole included.
} nmt_binning_scheme;

/**
 * @brief nmt_binning_scheme constructor for constant bandpowers.
 *
 * nmt_binning_scheme constructor for bandpowers with constant
 * width \p nlb, from ell = 2 to ell = \p lmax.
 * @param nlb Constant band width
 * @param lmax Maximum multipole
 * @return Allocated binning structure.
 */
nmt_binning_scheme *nmt_bins_constant(int nlb,int lmax);

/**
 * @brief  nmt_binning_scheme generic constructor.
 *
 * @param nell Number of elements in all subsequent arrays.
 * @param bpws Array of bandpower indices.
 * @param ells Array of multipole values. This function collects all multipoles
 *        into their associated bandpowers.
 * @param weights Array of weights associated to each multipole. Weights are 
 *        normalized to 1 within each bandpower.
 * @param lmax Maximum multipole to consider.
 * @return Allocated binning structure.
 */
nmt_binning_scheme *nmt_bins_create(int nell,int *bpws,int *ells,flouble *weights,int lmax);

/**
 * @brief nmt_binning_scheme constructor from file
 *
 * Builds a nmt_binning_scheme structure from an ASCII file.
 * @param fname Path to file containing information to build bandpowers.
 *        The file should contain three columns, corresponding to:
 *        bandpower index, multipole and weight (in this order).
 *        See definition of nmt_bins_create().
 * @return Allocated binning structure.
 */
nmt_binning_scheme *nmt_bins_read(char *fname,int lmax);

/**
 * @brief nmt_binning_scheme destructor
 */
void nmt_bins_free(nmt_binning_scheme *bin);

/**
 * @brief Returns average of input power spectrum into bandpowers.
 *
 * @param bin nmt_binning_scheme structure defining the bandpowers.
 * @param cls_in Array of \p ncls input power spectra. They should be 
 *        defined in all ells that go into any bandpower defined by \p bin.
 * @param cls_out Array of \p ncls averaged output power spectra.
 *        Should be allocated to the number of bandpowers defined bin \p.
 * @param ncls Number of input/output power spectra.
 */
void nmt_bin_cls(nmt_binning_scheme *bin,flouble **cls_in,flouble **cls_out,int ncls);

/**
 * @brief Returns binned power spectra interpolated into output multipoles.
 *
 * Top-hat interpolation is used (i.e. a given ell is associated with the binned power
 * spectrum value at the bandpower that ell corresponds to).
 * @param bin nmt_binning_scheme structure defining the bandpowers.
 * @param cls_in Array of \p ncls input power spectra. Must have the same number of
 *        elements as bandpowers defined by \p bin.
 * @param cls_out Array of \p ncls interpolated output power spectra.
 * @param ncls Number of input/output power spectra.
 */
void nmt_unbin_cls(nmt_binning_scheme *bin,flouble **cls_in,flouble **cls_out,int ncls);

/**
 * @brief Returns effective multipoles.
 *
 * Return the weighted average multipole values within each bandpower defined by \p bin.
 * @param bin nmt_binning_scheme structure defining the bandpowers.
 * @param larr Output array containing the effective multipole in each bandpower.
 *        Should be preallocated to the correct number of bandpowers.
 */
void nmt_ell_eff(nmt_binning_scheme *bin,flouble *larr);

/**
 * @brief Flat-sky Fourier-space function
 *
 * Unlike multipoles in harmonic space, in the case of full-sky operations, 
 * wavenumbers k in Fourier space for flat-sky fields are in general continuous
 * variables. This structure helps define functions of these continuous variables.
 */
typedef struct {
  int is_const; //!< If >0, this function is just a constant
  flouble x0; //!< Lower edge of spline interpolation
  flouble xf; //!< Upper edge of spline interpolation
  flouble y0; //!< Function will take this value for x < \p x0
  flouble yf; //!< Function will take this value for x > \p xf
  gsl_spline *spl; //!< GSL spline interpolator.
} nmt_k_function;

/**
 * @brief nmt_k_function creator.
 *
 * @param nk Number of elements in input arrays.
 * @param karr k-values at which the input function is sampled.
 * @param farr Function values at k = \p karr.
 * @param y0 Constant function value below interpolation range.
 * @param yf Constant function value above interpolation range.
 * @param is_const If non-zero, will create a constant function.
 *        In this case all previous arguments other than \p y0 are ignored
 *        and the function will take this value for all k.
 */
nmt_k_function *nmt_k_function_alloc(int nk,flouble *karr,flouble *farr,
				     flouble y0,flouble yf,int is_const);

/**
 * @brief nmt_k_function destructor
 */
void nmt_k_function_free(nmt_k_function *f);

/**
 * @brief nmt_k_function evaluator.
 *
 * Returns value of function at \p k.
 * @param k Value of k for which you want f(k).
 * @param intacc GSL interpolation accelerator. If you don't want any, just pass a NULL pointer.
 */
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
void nmt_compute_coupled_cell(nmt_field *fl1,nmt_field *fl2,flouble **cl_out);
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
