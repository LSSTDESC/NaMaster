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

/*! \mainpage NaMaster C API
 *
 * Welcome to the documentation of NaMaster's C API. Navigate through the tabs above to learn more
 * about the different functionality implemented in the code.
 * 
 * \section general_notes General notes
 *   - Most users will prefer to use the python wrapper "pymaster", which mostly calls the 
       C-based functions.
 *   - NaMaster uses a "row-major" order to define the ordering of power spectra into vectors.
       E.g. the cross-correlation of two spin-2 fields 'a' and 'b' would give rise to 4 power
       spectra: Ea-Eb, Ea-Bb, Ba-Eb and Ba-Bb. These are stored into 1-dimensional arrays using
       exactly that order. For the case of a spin-0 - spin-2 correlation, the ordering is
       [T-E, T-B], where T is the spin-0 field and (E,B) are the harmonic components of the
       spin-2 field.
 *   - The abbreviation MCM will often be used instead of "mode-coupling matrix".
 *   - SHT will sometimes be used for "Spherical Harmonic Transform". In the context of flat-sky
       fields, this should be understood as a standard Fast Fourier Transform (FFT) (with 
       appropriate trigonometric factors if dealing with spin-2 fields).
 *   - FWHM will sometimes be used for "Full-width at half-max".
 * 
 * \section more_info More info
 *
 * Please refer to the README and LICENSE files for further information on installation,
 * credits and licensing. Do not hesitate to contact the authors (preferably via github
 * issues on https://github.com/LSSTDESC/NaMaster) if you encounter any problems using 
 * the code.
 */

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
 *        Should be allocated to the number of bandpowers defined \p bin.
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
 * @param lmax Maximum multipole to be considered.
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
 *        Should be allocated to the number of bandpowers defined \p bin.
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
 * @param f nmt_k_function to evaluate.
 * @param k Value of k for which you want f(k).
 * @param intacc GSL interpolation accelerator. If you don't want any, just pass a NULL pointer.
 */
flouble nmt_k_function_eval(nmt_k_function *f,flouble k,gsl_interp_accel *intacc);

/**
 * @brief Flat-sky information.
 *
 * This structure contains all the information defining a given rectangular flat-sky patch.
 * The structure also contains information about the optimal way of sampling the Fourier
 * version of this patch into rings of |k|.
 */
typedef struct {
  int nx; //!< Number of grid points in the x dimension
  int ny; //!< Number of grid points in the y dimension
  long npix; //!< Total number of pixels (given by \p nx * \p ny
  flouble lx; //!< Length of the x dimension (in steradians)
  flouble ly; //!< Length of the y dimension (in steradians)
  flouble pixsize; //!< Pixel area (given by \p lx * \p ly / ( \p nx * \p ny))
  int n_ell; //!< Number of |k|-values for Fourier-space sampling.
  flouble dell; //!< Width of the Fourier-space rings. This is found as min(2 &pi; / \p lx,2 &pi; / \p ly).
  flouble i_dell; //!< 1 / \p dell
  flouble *ell_min; //!< Array of \p n_ell values containing the lower edges of each of the |k| rings.
  //  int *n_cells;
} nmt_flatsky_info;

/**
 * @brief nmt_flatsky_info constructor
 *
 * Builds nmt_flatsky_info from patch dimensions.
 * @param nx Number of grid points in the x dimension
 * @param ny Number of grid points in the y dimension
 * @param lx Length of the x dimension (in steradians)
 * @param ly Length of the y dimension (in steradians)
 * @return Allocated nmt_flatsky_info structure.
 */
nmt_flatsky_info *nmt_flatsky_info_alloc(int nx,int ny,flouble lx,flouble ly);

/**
 * @brief nmt_flatsky_info destructor.
 */
void nmt_flatsky_info_free(nmt_flatsky_info *fs);

/**
 * @brief Flat-sky field
 *
 * This structure contains all the information defining a spin-s flat-sky field.
 * This includes field values, masking, purification and contamination.
 */
typedef struct {
  nmt_flatsky_info *fs; //!< Structure defining patch geometry.
  long npix; //!< Number of pixels in all maps (also contained in \p fs).
  int pure_e; //!< >0 if E-modes have been purified.
  int pure_b; //!< >0 if B-modes have been purified.
  flouble *mask; //!< Field's mask (an array of \p npix values).
  fcomplex **a_mask; //!< Fourier transform of the mask. Only computed if E or B are purified.
  int pol; //!< >0 if field is spin-2 (otherwise it's spin-0).
  int nmaps; //!< Number of maps in the field (2 for spin-2, 1 for spin-0).
  flouble **maps; //!< Observed field values. When initialized, these maps are already multiplied by the mask, contaminant deprojected and purified if requested.
  fcomplex **alms; //!< Fourier-transfoms of the maps.
  int ntemp; //!< Number of contaminant templates
  flouble ***temp; //!< Contaminant template maps (mask-multiplied but NOT purified).
  fcomplex ***a_temp; //!< Fourier-transfomrs of template maps (mask-multiplied AND purified if requested).
  gsl_matrix *matrix_M; //!< Inverse contaminant covariance matrix (see scientific documentation or companion paper).
  nmt_k_function *beam; //!< Function defining a circularly-symmetric beam function. Power spectra will be beam-deconvolved.
} nmt_field_flat;

/**
 * @brief nmt_field_flat destructor
 */
void nmt_field_flat_free(nmt_field_flat *fl);

/**
 * @brief nmt_field_flat constructor
 *
 * Builds an nmt_field_flat structure from input maps and patch parameters.
 * @param nx Number of grid points in the x dimension.
 * @param ny Number of grid points in the y dimension.
 * @param lx Length of the x dimension (in steradians).
 * @param ly Length of the y dimension (in steradians).
 * @param mask Field's mask (an array of \p nx * \p ny values).
 * @param pol >0 if this is a spin-2 field (spin-0 otherwise).
 * @param maps Observed field values BEFORE multiplying by the mask
          (this is irrelevant for binary masks).
 * @param ntemp Number of contaminant templates affecting this field.
 * @param temp Contaminant template maps (again, NOT multiplied by the mask).
 * @param nl_beam Number of multipole values defining this field's beam.
 * @param l_beam Multipole values at which this field's beam is defined.
 * @param beam Beam values at ell = \p l_beam. Pass a NULL pointer if you don't
          want any beam (\p nl_beam and \p l_beam will be ignored).
 * @param pure_e Set to >0 if you want purified E-modes.
 * @param pure_b Set to >0 if you want purified B-modes.
 * @param tol_pinv Contaminant deprojection requires the inversion of the template
          covariance matrix. This could be ill-defined if some templates are linearly
	  related. In this case we use a pseudo-inverse that accounts for this 
	  possibility in a consistent way. Effectively this is a singular-value 
	  decomposition. All eigenvalues that are smaller than \p tol_pinv the largest
	  eigenvalue will be discarded.
 */
nmt_field_flat *nmt_field_flat_alloc(int nx,int ny,flouble lx,flouble ly,
				     flouble *mask,int pol,flouble **maps,int ntemp,flouble ***temp,
				     int nl_beam,flouble *l_beam,flouble *beam,
				     int pure_e,int pure_b,double tol_pinv);

/**
 * @brief Gaussian realizations of flat-sky fields 
 *
 * Generates a Gaussian realization of an arbitrary list of possibly-correlated 
 * fields with different spins.
 * @param nx Number of grid points in the x dimension.
 * @param ny Number of grid points in the y dimension.
 * @param lx Length of the x dimension (in steradians).
 * @param ly Length of the y dimension (in steradians).
 * @param nfields Number of fields to generate.
 * @param spin_arr Array (size \p nfields) containing the spins of the fields to be generated.
 * @param nl_beam Number of multipoles at which the field beams are defined.
 * @param l_beam Array of multipoles at which the field beams are defined.
 * @param beam_fields Array of beams (one per field).
 * @param nl_cell Number of multipole values at which the input power spectra are provided.
 * @param l_cell Array of multipole values at which the input power spectra are provided.
 * @param cell_fields Array of input power spectra. Shape should be [\p n_cls][\p nl_cell],
          where \p n_cls is the number of power spectra needed to define all the fields.
	  This should be \p n_cls = n_maps * (n_maps + 1) / 2, where n_maps is the total
	  number of maps required (1 for each spin-0 field, 2 for each spin-2 field). Power
	  spectra must be provided only for the upper-triangular part in row-major order
	  (e.g. if n_maps is 3, there will be 6 power spectra ordered as [1-1,1-2,1-3,2-2,2-3,3-3].
 * @param seed Seed for this particular realization.
 * @return Gaussian realization.
 */
flouble **nmt_synfast_flat(int nx,int ny,flouble lx,flouble ly,int nfields,int *spin_arr,
			   int nl_beam,flouble *l_beam,flouble **beam_fields,
			   int nl_cell,flouble *l_cell,flouble **cell_fields,
			   int seed);

/**
 * @brief E- or B-mode purifies a given pair of flat-sky (Q,U) maps.
 *
 * This function is mostly used internally by NaMaster, and its standalone use is discouraged.
 * @param fl nmt_field_flat containing information about what should be purified.
 * @param mask Sky mask (should be appropriately apodized - see scientific documentation).
 * @param walm0 Fourier transform of the mask.
 * @param maps_in Maps to be purified (should NOT be mask-multiplied).
 * @param maps_out Output purified maps.
 * @param alms Fourier transform of the output purified maps.
 */
void nmt_purify_flat(nmt_field_flat *fl,flouble *mask,fcomplex **walm0,
		     flouble **maps_in,flouble **maps_out,fcomplex **alms);

/**
 * @brief Full-sky field
 *
 * This structure contains all the information defining a spin-s full-sky field.
 * This includes field values, masking, purification and contamination.
 */
typedef struct {
  long nside; //!< HEALPix resolution parameters
  long npix; //!< Number of pixels in all maps
  int lmax; //!< Maximum multipole used
  int pure_e; //!< >0 if E-modes have been purified
  int pure_b; //!< >0 if B-modes have been purified
  flouble *mask; //!< Field's mask (an array of \p npix values).
  fcomplex **a_mask; //!< Spherical transform of the mask. Only computed if E or B are purified.
  int pol; //!< >0 if field is spin-2 (otherwise it's spin-0).
  int nmaps; //!< Number of maps in the field (2 for spin-2, 1 for spin-0).
  flouble **maps; //!< Observed field values. When initialized, these maps are already multiplied by the mask, contaminant-deprojected and purified if requested.
  fcomplex **alms; //!< Spherical harmonic transfoms of the maps.
  int ntemp; //!< Number of contaminant templates
  flouble ***temp; //!< Contaminant template maps (mask-multiplied but NOT purified).
  fcomplex ***a_temp; //!< Spherical harmonic transfomrs of template maps (mask-multiplied AND purified if requested).
  gsl_matrix *matrix_M; //!< Inverse contaminant covariance matrix (see scientific documentation or companion paper).
  flouble *beam; //!< Field's beam (defined on all multipoles up to \p lmax).
} nmt_field;

/**
 * @brief nmt_field destructor.
 */
void nmt_field_free(nmt_field *fl);

/**
 * @brief nmt_field constructor
 *
 * Builds an nmt_field structure from input maps and resolution parameters.
 * @param nside HEALPix resolution parameter.
 * @param mask Field's mask (an array of 12 * \p nside^2 values).
 * @param pol >0 if this is a spin-2 field (spin-0 otherwise).
 * @param maps Observed field values BEFORE multiplying by the mask 
          (this is irrelevant for binary masks).
 * @param ntemp Number of contaminant templates affecting this field.
 * @param temp Contaminant template maps (again, NOT multiplied by the mask).
 * @param beam Harmonic coefficients of the beam (defined for all multipoles up to
 *        3 * \p nside - 1). Pass a NULL pointer if you don't want any beam.
 * @param pure_e Set to >0 if you want purified E-modes.
 * @param pure_b Set to >0 if you want purified B-modes.
 * @param n_iter_mask_purify E/B purification requires a number of harmonic-space
          operations on an appropriately apodized mask. This parameter sets the
          number of iterations requested to compute the spherical harmonic transform
          of the field's mask. Higher values will produce more accurate results (at
	  the cost of computational time).
 * @param tol_pinv Contaminant deprojection requires the inversion of the template
          covariance matrix. This could be ill-defined if some templates are linearly
	  related. In this case we use a pseudo-inverse that accounts for this 
	  possibility in a consistent way. Effectively this is a singular-value 
	  decomposition. All eigenvalues that are smaller than \p tol_pinv the largest
	  eigenvalue will be discarded.
 */
nmt_field *nmt_field_alloc_sph(long nside,flouble *mask,int pol,flouble **maps,
			       int ntemp,flouble ***temp,flouble *beam,
			       int pure_e,int pure_b,int n_iter_mask_purify,double tol_pinv);
/**
 * @brief nmt_field constructor from file.
 *
 * Builds an nmt_field structure from data written in files.
 * @param fname_mask Path to FITS file containing the field's mask (single HEALPix map).
 * @param pol >0 if this is a spin-2 field (spin-0 otherwise).
 * @param fname_maps Path to FITS file containing the field's observed maps
          (1(2) maps if \p pol=0(1)).
 * @param fname_temp Path to FITS file containing the field's contaminant templates.
          If \p pol > 0, spin-2 is assumed, and the file should contain an even number
          of files. Each consecutive pair of maps will be interpreted as the Q and U
	  components of a given contaminant. Pass "none" if you don't want any contaminants.
 * @param fname_beam Path to ASCII file containing the field's beam. The file should
          contain two columns: l (multipole) and b_l (beam SHT at that multipole).
	  Pass "none if you don't want a beam.
 * @param pure_e >0 if you want E-mode purification.
 * @param pure_b >0 if you want B-mode purification.
 * @param n_iter_mask_purify E/B purification requires a number of harmonic-space
          operations on an appropriately apodized mask. This parameter sets the
          number of iterations requested to compute the spherical harmonic transform
          of the field's mask. Higher values will produce more accurate results (at
	  the cost of computational time).
 * @param tol_pinv Contaminant deprojection requires the inversion of the template
          covariance matrix. This could be ill-defined if some templates are linearly
	  related. In this case we use a pseudo-inverse that accounts for this 
	  possibility in a consistent way. Effectively this is a singular-value 
	  decomposition. All eigenvalues that are smaller than \p tol_pinv the largest
	  eigenvalue will be discarded.
 */
nmt_field *nmt_field_read(char *fname_mask,char *fname_maps,char *fname_temp,char *fname_beam,
			  int pol,int pure_e,int pure_b,int n_iter_mask_purify,double tol_pinv);

/**
 * @brief Gaussian realizations of full-sky fields 
 *
 * Generates a Gaussian realization of an arbitrary list of possibly-correlated fields with different spins.
 * @param nside HEALPix resolution parameter.
 * @param lmax Maximum multipole used.
 * @param nfields Number of fields to generate.
 * @param spin_arr Array (size \p nfields) containing the spins of the fields to be generated.
 * @param beam_fields Array of beams (one per field). Must be defined at all ell <= \p lmax.
 * @param cells Array of input power spectra (defined at all ell <= \p lmax). Shape
          should be [\p n_cls][\p lmax+1], where \p n_cls is the number of power spectra
	  needed to define all the fields. This should be \p n_cls = n_maps * (n_maps + 1) / 2,
	  where n_maps is the total number of maps required (1 for each spin-0 field, 2 for
	  each spin-2 field). Power spectra must be provided only for the upper-triangular part
	  in row-major order (e.g. if n_maps is 3, there will be 6 power spectra ordered as
	  [1-1,1-2,1-3,2-2,2-3,3-3].
 * @param seed Seed for this particular realization.
 * @return Gaussian realization.
 */
flouble **nmt_synfast_sph(int nside,int nfields,int *spin_arr,int lmax,
			  flouble **cells,flouble **beam_fields,int seed);

/**
 * @brief E- or B-mode purifies a given pair of full-sky (Q,U) maps.
 *
 * This function is mostly used internally by NaMaster, and its standalone use is discouraged.
 * @param fl nmt_field containing information about what should be purified.
 * @param mask Sky mask (should be appropriately apodized - see scientific documentation).
 * @param walm0 Spherical harmonic transform of the mask.
 * @param maps_in Maps to be purified (should NOT be mask-multiplied).
 * @param maps_out Output purified maps.
 * @param alms Spherical harmonic transform of the output purified maps.
 */
void nmt_purify(nmt_field *fl,flouble *mask,fcomplex **walm0,
		flouble **maps_in,flouble **maps_out,fcomplex **alms);

/**
 * @brief Apodize full-sky mask.
 *
 * Produces apodized version of a full-sky mask for a number of apodization schemes.
 * @param nside HEALPix resolution parameter.
 * @param mask_in Input mask to be apodized.
 * @param mask_out Output apodized mask.
 * @param aposize Apodization scale (in degrees).
 * @param apotype String defining the apodization procedure. Three values allowed: 'C1', 'C2' and 'Smooth'. These correspond to:
 *    - \p apotype = "C1". All pixels are multiplied by a factor \f$f\f$, given by:
 *\f[
 *   f=\left\{
 *     \begin{array}{cc}
 *       x-\sin(2\pi x)/(2\pi) & x<1\\
 *       1 & {\rm otherwise}
 *     \end{array}
 *     \right.,
 * \f]
        where \f$x=\sqrt{(1-\cos\theta)/(1-\cos(\theta_*))}\f$, \f$\theta_*\f$ is the
	apodization scale and \f$\theta\f$ is the angular separation between a pixel and
	the nearest masked pixel (i.e. where the mask takes a zero value).
 *    - \p apotype = "C2". The same as the C1 case, but the function in this case is:
 *\f[
 *   f=\left\{
 *     \begin{array}{cc}
 *       \frac{1}{2}\left[1-\cos(\pi x)\right] & x<1\\
 *       1 & {\rm otherwise}
 *     \end{array}
 *     \right.,
 * \f]
 *    - \p apotype = "Smooth". This apodization is carried out in three steps:
 *         -# All pixels within a disc of radius \f$2.5\theta_*\f$ of a masked pixel are masked.
 *         -# The resulting map is smooth with a Gaussian window function with standard
              deviation \f$\sigma=\theta_*\f$.
 *         -# One final pass is made through all pixels to ensure that all originally masked
 *            pixels are still masked after the smoothing operation.
 */
void nmt_apodize_mask(long nside,flouble *mask_in,flouble *mask_out,flouble aposize,char *apotype);


/**
 * @brief Apodize flat-sky mask.
 *
 * Produces apodized version of a flat-sky mask for a number of apodization schemes.
 * @param nx Number of grid points in the x dimension
 * @param ny Number of grid points in the y dimension
 * @param lx Length of the x dimension (in steradians)
 * @param ly Length of the y dimension (in steradians)
 * @param mask_in Input mask to be apodized.
 * @param mask_out Output apodized mask.
 * @param aposize Apodization scale (in degrees).
 * @param apotype String defining the apodization procedure. See definitions of nmt_apodize_mask().
 */
void nmt_apodize_mask_flat(int nx,int ny,flouble lx,flouble ly,
			   flouble *mask_in,flouble *mask_out,flouble aposize,char *apotype);

/**
 * @brief Flat-sky mode-coupling matrix.
 *
 * Structure containing information about the mode-coupling matrix (MCM) for flat-sky pseudo-CLs.
 */
typedef struct {
  int ncls; //!< Number of power spectra (1, 2 or 4 depending of the spins of the fields being correlated.
  flouble ellcut_x[2]; //!< Range of ells in the x direction to be masked in Fourie space
  flouble ellcut_y[2]; //!< Range of ells in the y direction to be masked in Fourie space
  int pe1; //!< Is the E-mode component of the first field purified?
  int pe2; //!< Is the E-mode component of the second field purified?
  int pb1; //!< Is the B-mode component of the first field purified?
  int pb2; //!< Is the B-mode component of the second field purified?
  nmt_flatsky_info *fs; //!< Contains information about rectangular flat-sky patch.
  int is_teb; //!< Does it hold all MCM elements to compute all of spin0-spin0, 0-2 and 2-2 correlations?
  flouble *mask1; //!< Mask of the first field being correlated
  flouble *mask2; //!< Mask of the second field being correlated
#ifdef _ENABLE_FLAT_THEORY_ACCURATE
  flouble *maskprod; //!< Mask product used for accurate theory estimate (non-tested)
#endif //_ENABLE_FLAT_THEORY_ACCURATE
  int *n_cells; //!< Number of unmasked Fourier-space grid points contributing to a given bandpower
  flouble **coupling_matrix_unbinned; //!< Unbinned MCM
  flouble **coupling_matrix_binned; //!< Binned MCM
  nmt_binning_scheme_flat *bin; //!< Bandpowers defining the binning
  flouble lmax; //!< Maximum k-mode used
  gsl_matrix *coupling_matrix_binned_gsl; //!< GSL version of MCM (prepared for inversion)
  gsl_permutation *coupling_matrix_perm; //!< Complements \p coupling_matrix_binned_gsl for inversion.
} nmt_workspace_flat;

/**
 * @brief nmt_workspace_flat destructor
 */
void nmt_workspace_flat_free(nmt_workspace_flat *w);

/**
 * @brief Builds nmt_workspace_flat structure from file
 *
 * The input file uses a native binary format. In combination with nmt_workspace_flat_write(),
 * this can be used to save the information contained in a given workspace and reuse it for 
 * future power spectrum computations. The same workspace can be used on any pair of fields
 * with the same masks.
 * @param fname Path to input file.
 */
nmt_workspace_flat *nmt_workspace_flat_read(char *fname);

/**
 * @brief Saves nmt_workspace_flat structure to file
 *
 * The output file uses a native binary format. In combination with nmt_workspace_flat_read(),
 * this can be used to save the information contained in a given workspace and reuse it for 
 * future power spectrum computations. The same workspace can be used on any pair of fields
 * with the same masks.
 * @param w nmt_workspace_flat to be saved.
 * @param fname Path to output file.
 */
void nmt_workspace_flat_write(nmt_workspace_flat *w,char *fname);

/**
 * @brief Computes mode-coupling matrix.
 *
 * Computes MCM for a given pair of flat-sky fields.
 * @param fl1 nmt_field_flat structure defining the first field to correlate.
 * @param fl2 nmt_field_flat structure defining the second field to correlate.
 * @param bin nmt_binning_scheme_flat defining the power spectrum bandpowers.
 * @param lmn_x Lower end of the range of multipoles in the x direction that should be masked.
 * @param lmx_x Upper end of the range of multipoles in the x direction that should be masked.
 *        if \p lmx_x < \p lmn_x, no Fourier-space masked is performed.
 * @param lmn_y Same as \p lmn_x for the y direction.
 * @param lmx_y Same as \p lmx_x for the y direction.
 * @param is_teb if !=0, all mode-coupling matrices (0-0,0-2,2-2) will be computed at the same time.
 */
nmt_workspace_flat *nmt_compute_coupling_matrix_flat(nmt_field_flat *fl1,nmt_field_flat *fl2,
						     nmt_binning_scheme_flat *bin,
						     flouble lmn_x,flouble lmx_x,
						     flouble lmn_y,flouble lmx_y,int is_teb);

/**
 * @brief Computes deprojection bias.
 *
 * Computes contaminant deprojection bias for a pair of fields.
 * See notes about power spectrum ordering in the main page of this documentation.
 * @param fl1 nmt_field_flat structure defining the first field to correlate.
 * @param fl2 nmt_field_flat structure defining the second field to correlate.
 * @param bin nmt_binning_scheme_flat defining the power spectrum bandpowers.
 * @param lmn_x Lower end of the range of multipoles in the x direction that should be masked.
 * @param lmx_x Upper end of the range of multipoles in the x direction that should be masked.
 *        if \p lmx_x < \p lmn_x, no Fourier-space masked is performed.
 * @param lmn_y Same as \p lmn_x for the y direction.
 * @param lmx_y Same as \p lmx_x for the y direction.
 * @param nl_prop Number of multipoles over which the proposed power spectrum is defined.
 * @param l_prop Array of multipoles over which the proposed power spectrum is defined.
 * @param cl_proposal Proposed power spectrum. Should have shape [ncls][\p nl_prop], where
          \p ncls is the appropriate number of power spectra given the spins of the input
	  fields (e.g. \p ncls = 2*2 = 4 if both fields have spin=2).
 * @param cl_bias Ouptput deprojection bias. Should be allocated to shape [ncls][nbpw],
          where \p ncls is defined above and \p nbpw is the number of bandpowers
	  defined by \p bin.
 */
void nmt_compute_deprojection_bias_flat(nmt_field_flat *fl1,nmt_field_flat *fl2,
					nmt_binning_scheme_flat *bin,
					flouble lmn_x,flouble lmx_x,flouble lmn_y,flouble lmx_y,
					int nl_prop,flouble *l_prop,flouble **cl_proposal,
					flouble **cl_bias);
#ifdef _ENABLE_FLAT_THEORY_ACCURATE
void nmt_couple_cl_l_flat_accurate(nmt_workspace_flat *w,int nl,flouble *larr,flouble **cl_in,
				   flouble **cl_out);
#endif //_ENABLE_FLAT_THEORY_ACCURATE

/**
 * @brief Mode-couples an input power spectrum
 *
 * This function applies the effects of the mode-coupling the pseudo-CL estimator for a given 
 * input power spectrum. This function should be used in conjunction with nmt_decouple_cl_l_flat()
 * to compute the theory prediction of the pseudo-CL estimator. See the scientific documentation
 * or the companion paper for further details on how this is done in particular for the flat-sky
 * approximation.
 * See notes about power spectrum ordering in the main page of this documentation.
 * @param w nmt_workspace_flat structure containing the mode-coupling matrix
 * @param nl Number of multipoles on which the input power spectrum is defined.
 * @param larr Array of multipoles on which the input power spectrum is defined.
 * @param cl_in Array of input power spectra. Should have shape [ncls][nl], where ncls is the
          appropriate number of power spectra given the fields being correlated (e.g. ncls=4=2*2
	  for two spin-2 fields.
 * @param cl_out Array of output power spectra. Should have shape [ncls][nbpw], where ncls is
          defined above and nbpw is the number of bandpowers used to define \p w.
 */
void nmt_couple_cl_l_flat_fast(nmt_workspace_flat *w,int nl,flouble *larr,flouble **cl_in,
				 flouble **cl_out);
/**
 * @brief Mode-couples an input power spectrum
 *
 * Faster (but less accurate) version of nmt_couple_cl_l_flat_fast().
 * @param w nmt_workspace_flat structure containing the mode-coupling matrix
 * @param nl Number of multipoles on which the input power spectrum is defined.
 * @param larr Array of multipoles on which the input power spectrum is defined.
 * @param cl_in Array of input power spectra. Should have shape [ncls][nl], where ncls is the
          appropriate number of power spectra given the fields being correlated (e.g. ncls=4=2*2
	  for two spin-2 fields.
 * @param cl_out Array of output power spectra. Should have shape [ncls][nbpw], where ncls is
          defined above and nbpw is the number of bandpowers used to define \p w.
 */
void nmt_couple_cl_l_flat_quick(nmt_workspace_flat *w,int nl,flouble *larr,flouble **cl_in,
				flouble **cl_out);

/**
 * @brief Inverts mode-coupling matrix
 *
 * Multiplies coupled power spectra by inverse mode-coupling matrix.
 * See notes about power spectrum ordering in the main page of this documentation.
 * @param w nmt_workspace_flat containing the mode-coupling matrix.
 * @param cl_in Input coupled power spectra. Should have shape [ncls][nbpw], where
          \p ncls is the appropriate number of power spectra given the fields used
	  to define \p w (e.g. 4=2*2 for two spin-2 fields) and \p nbpw is the number
	  of bandpowers used when defining \p w.
 * @param cl_noise_in Noise bias (same shape as \p cl_in).
 * @param cl_bias Deprojection bias (same shape as \p cl_in, see nmt_compute_deprojection_bias_flat()).
 * @param cl_out Mode-decoupled power spectrum (same shape as \p cl_in).
 */
void nmt_decouple_cl_l_flat(nmt_workspace_flat *w,flouble **cl_in,flouble **cl_noise_in,
			    flouble **cl_bias,flouble **cl_out);

/**
 * @brief Coupled pseudo-CL
 *
 * Computes the pseudo-CL power spectrum of two fields without accounting for the mode-coupling
 * matrix.
 * See notes about power spectrum ordering in the main page of this documentation.
 * @param fl1 nmt_field_flat structure defining the first field to correlate.
 * @param fl2 nmt_field_flat structure defining the second field to correlate.
 * @param bin nmt_binning_scheme_flat defining the power spectrum bandpowers.
 * @param lmn_x Lower end of the range of multipoles in the x direction that should be masked.
 * @param lmx_x Upper end of the range of multipoles in the x direction that should be masked.
 *        if \p lmx_x < \p lmn_x, no Fourier-space masked is performed.
 * @param lmn_y Same as \p lmn_x for the y direction.
 * @param lmx_y Same as \p lmx_x for the y direction.
 * @param cl_out Ouptput power spectrum. Should be allocated to shape [ncls][nbpw], where
          \p ncls is the appropriate number of power spectra (e.g. 4=2*2 for two spin-2
	  fields), and \p nbpw is the number of bandpowers defined by \p bin.
 */
void nmt_compute_coupled_cell_flat(nmt_field_flat *fl1,nmt_field_flat *fl2,
				   nmt_binning_scheme_flat *bin,flouble **cl_out,
				   flouble lmn_x,flouble lmx_x,flouble lmn_y,flouble lmx_y);

/**
 * @brief Computes pseudo-CL specrum.
 *
 * Wrapper function containing all the steps to compute a power spectrum. For performance
 * reasons, the blind use of this function is discouraged against a smarter combination of
 * nmt_workspace_flat structures and nmt_compute_coupled_cell_flat().
 * See notes about power spectrum ordering in the main page of this documentation.
 * @param fl1 nmt_field_flat structure defining the first field to correlate.
 * @param fl2 nmt_field_flat structure defining the second field to correlate.
 * @param bin nmt_binning_scheme_flat defining the power spectrum bandpowers.
 * @param lmn_x Lower end of the range of multipoles in the x direction that should be masked.
 * @param lmx_x Upper end of the range of multipoles in the x direction that should be masked.
 *        if \p lmx_x < \p lmn_x, no Fourier-space masked is performed.
 * @param lmn_y Same as \p lmn_x for the y direction.
 * @param lmx_y Same as \p lmx_x for the y direction.
 * @param w0 nmt_workspace_flat structure containing the mode-coupling matrix. If NULL, a new 
          computation of the MCM will be carried out and stored in the output nmt_workspace_flat.
	  Otherwise, \p w0 will be used and returned by this function.
 * @param nl_prop Number of multipoles over which the proposed power spectrum is defined.
 * @param l_prop Array of multipoles over which the proposed power spectrum is defined.
 * @param cl_prop Proposed power spectrum. Should have shape [ncls][\p nl_prop], where
          \p ncls is the appropriate number of power spectra given the spins of the input
	  fields (e.g. \p ncls = 2*2 = 4 if both fields have spin=2).
 * @param cl_noise Noise bias. Should have shape [ncls][nbpw], where \p ncls is 
 *        defined above and \p nbpw is the number of bandpowers defined by \p bin.
 * @param cl_out Ouptput power spectrum. Should be allocated to shape [ncls][nbpw], 
          where \p ncls is defined above and \p nbpw is the number of bandpowers defined
	  by \p bin.
 * @return Newly allocated nmt_workspace_flat structure containing the mode-coupling matrix
           if \p w0 is NULL (will return \p w0 otherwise).
 */
nmt_workspace_flat *nmt_compute_power_spectra_flat(nmt_field_flat *fl1,nmt_field_flat *fl2,
						   nmt_binning_scheme_flat *bin,
						   flouble lmn_x,flouble lmx_x,
						   flouble lmn_y,flouble lmx_y,
						   nmt_workspace_flat *w0,flouble **cl_noise,
						   int nl_prop,flouble *l_prop,flouble **cl_prop,
						   flouble **cl_out);

/**
 * @brief Full-sky mode-coupling matrix.
 *
 * Structure containing information about the mode-coupling matrix (MCM) for full-sky pseudo-CLs.
 */
typedef struct {
  int lmax; //!< Maximum multipole used
  int is_teb; //!< Does it hold all MCM elements to compute all of spin0-spin0, 0-2 and 2-2 correlations?
  int ncls; //!< Number of power spectra (1, 2 or 4 depending of the spins of the fields being correlated.
  int nside; //!< HEALPix resolution parameter
  flouble *mask1; //!< Mask of the first field being correlated.
  flouble *mask2; //!< Mask of the second field being correlated.
  flouble *pcl_masks; //!< Pseudo-CL of the masks.
  flouble **coupling_matrix_unbinned; //!< Unbinned mode-coupling matrix
  nmt_binning_scheme *bin; //!< Bandpowers defining the binning
  gsl_matrix *coupling_matrix_binned; //!< GSL version of MCM (prepared for inversion)
  gsl_permutation *coupling_matrix_perm; //!< Complements \p coupling_matrix_binned_gsl for inversion.
} nmt_workspace;

/**
 * @brief Computes mode-coupling matrix.
 *
 * Computes MCM for a given pair of full-sky fields.
 * @param fl1 nmt_field structure defining the first field to correlate.
 * @param fl2 nmt_field structure defining the second field to correlate.
 * @param bin nmt_binning_scheme defining the power spectrum bandpowers.
 * @param is_teb if !=0, all mode-coupling matrices (0-0,0-2,2-2) will be computed at the same time.
 */
nmt_workspace *nmt_compute_coupling_matrix(nmt_field *fl1,nmt_field *fl2,nmt_binning_scheme *bin,int is_teb);

/**
 * @brief Updates the mode coupling matrix with a new one.Saves nmt_workspace structure to file
 *
 * The new matrix must be provided as a single 1D array of size n_rows\f$^2\f$.
 * Here n_rows=n_cls * n_ell is the size of the flattened power spectra, where n_cls is the number
 * of power spectra (1, 2 or 4 for spin0-0, spin0-2 and spin2-2 correlations) and n_ells=lmax+1
 * (by default lmax=3*nside-1). The ordering of the power spectra should be such that the
 * l-th element of the i-th power spectrum is stored with index l * n_cls + i.
 * @param w nmt_workspace to be updated.
 * @param n_rows size of the flattened power spectra.
 * @param new_matrix new mode-coupling matrix (flattened).
 */
void nmt_update_coupling_matrix(nmt_workspace *w,int n_rows,double *new_matrix);

/**
 * @brief Saves nmt_workspace structure to file
 *
 * The output file uses a native binary format. In combination with nmt_workspace_read(),
 * this can be used to save the information contained in a given workspace and reuse it for 
 * future power spectrum computations. The same workspace can be used on any pair of fields
 * with the same masks.
 * @param w nmt_workspace to be saved.
 * @param fname Path to output file.
 */
void nmt_workspace_write(nmt_workspace *w,char *fname);

/**
 * @brief Builds nmt_workspace structure from file
 *
 * The input file uses a native binary format. In combination with nmt_workspace_write(),
 * this can be used to save the information contained in a given workspace and reuse it for 
 * future power spectrum computations. The same workspace can be used on any pair of fields
 * with the same masks.
 * @param fname Path to input file.
 */
nmt_workspace *nmt_workspace_read(char *fname);

/**
 * @brief nmt_workspace destructor
 */
void nmt_workspace_free(nmt_workspace *w);

/**
 * @brief Computes deprojection bias.
 *
 * Computes contaminant deprojection bias for a pair of fields.
 * See notes about power spectrum ordering in the main page of this documentation.
 * @param fl1 nmt_field structure defining the first field to correlate.
 * @param fl2 nmt_field structure defining the second field to correlate.
 * @param cl_proposal Proposed power spectrum. Should have shape [ncls][3 \p nside], where
          \p ncls is the appropriate number of power spectra given the spins of the input
	  fields (e.g. \p ncls = 2*2 = 4 if both fields have spin=2).
 * @param cl_bias Ouptput deprojection bias. Should be allocated to shape [ncls][3 * \p nside],
          where \p ncls is defined above.
 */
void nmt_compute_deprojection_bias(nmt_field *fl1,nmt_field *fl2,
				   flouble **cl_proposal,flouble **cl_bias);

/**
 * @brief Noise bias from uncorrelated noise map
 *
 * Computes deprojection bias due to an source of uncorrelated noise given an input noise variance map.
 * See companion paper for more details.
 * @param fl1 nmt_field structure defining the properties of the field for which this noise bias
          applies.
 * @param map_var Noise variance map (should contain per-pixel noise variance).
 * @param cl_bias Ouptput noise bias. Should be allocated to shape [ncls][3 * \p nside],
          where \p ncls is the appropriate number of power spectra given the spins of the input
	  fields (e.g. \p ncls = 2*2 = 4 if both fields have spin=2).
 */
void nmt_compute_uncorr_noise_deprojection_bias(nmt_field *fl1,flouble *map_var,flouble **cl_bias);

/**
 * @brief Mode-couples an input power spectrum
 *
 * This function applies the effects of the mode-coupling the pseudo-CL estimator for a given 
 * input power spectrum. This function should be used in conjunction with nmt_decouple_cl_l()
 * to compute the theory prediction of the pseudo-CL estimator.
 * See notes about power spectrum ordering in the main page of this documentation.
 * @param w nmt_workspace structure containing the mode-coupling matrix
 * @param cl_in Array of input power spectra. Should have shape [ncls][3 * \p nside], where ncls
          is the appropriate number of power spectra given the fields being correlated
	  (e.g. ncls=4=2*2 for two spin-2 fields).
 * @param cl_out Array of output power spectra. Should have shape [ncls][3 * \p nside], where
          ncls is defined above.
 */
void nmt_couple_cl_l(nmt_workspace *w,flouble **cl_in,flouble **cl_out);

/**
 * @brief Inverts mode-coupling matrix
 *
 * Multiplies coupled power spectra by inverse mode-coupling matrix.
 * See notes about power spectrum ordering in the main page of this documentation.
 * @param w nmt_workspace containing the mode-coupling matrix.
 * @param cl_in Input coupled power spectra. Should have shape [ncls][3 * \p nside], where
          \p ncls is the appropriate number of power spectra given the fields used
	  to define \p w (e.g. 4=2*2 for two spin-2 fields).
 * @param cl_noise_in Noise bias (same shape as \p cl_in).
 * @param cl_bias Deprojection bias (same shape as \p cl_in, see nmt_compute_deprojection_bias()).
 * @param cl_out Mode-decoupled power spectrum. Should have shape [ncls][nbpw], where
          ncls is defined above and nbpw is the number of bandpowers used to define \p w.
 */
void nmt_decouple_cl_l(nmt_workspace *w,flouble **cl_in,flouble **cl_noise_in,
		       flouble **cl_bias,flouble **cl_out);

/**
 * @brief Coupled pseudo-CL
 *
 * Computes the pseudo-CL power spectrum of two fields without accounting for the mode-coupling
 * matrix. This is essentially equivalent to running HEALPix's 'anafast' on the purified and
 * contaminant-deprojected input fields.
 * See notes about power spectrum ordering in the main page of this documentation.
 * @param fl1 nmt_field structure defining the first field to correlate.
 * @param fl2 nmt_field structure defining the second field to correlate.
 * @param cl_out Ouptput power spectrum. Should be allocated to shape [ncls][3 * \p nside], where
          \p ncls is the appropriate number of power spectra (e.g. 4=2*2 for two spin-2 fields).
 */
void nmt_compute_coupled_cell(nmt_field *fl1,nmt_field *fl2,flouble **cl_out);

/**
 * @brief Computes pseudo-CL specrum.
 *
 * Wrapper function containing all the steps to compute a power spectrum. For performance
 * reasons, the blind use of this function is discouraged against a smarter combination of
 * nmt_workspace structures and nmt_compute_coupled_cell().
 * See notes about power spectrum ordering in the main page of this documentation.
 * @param fl1 nmt_field structure defining the first field to correlate.
 * @param fl2 nmt_field structure defining the second field to correlate.
 * @param bin nmt_binning_scheme defining the power spectrum bandpowers.
 * @param w0 nmt_workspace structure containing the mode-coupling matrix. If NULL, a new 
          computation of the MCM will be carried out and stored in the output nmt_workspace.
	  Otherwise, \p w0 will be used and returned by this function.
 * @param cl_proposal Proposed power spectrum. Should have shape [ncls][3 * \p nside], where
          \p ncls is the appropriate number of power spectra given the spins of the input
	  fields (e.g. \p ncls = 2*2 = 4 if both fields have spin=2).
 * @param cl_noise Noise bias (same shape as \p cl_prop).
 * @param cl_out Ouptput power spectrum. Should be allocated to shape [ncls][nbpw], 
          where \p ncls is defined above and \p nbpw is the number of bandpowers defined
	  by \p bin.
 * @return Newly allocated nmt_workspace structure containing the mode-coupling matrix
           if \p w0 is NULL (will return \p w0 otherwise).
 */
nmt_workspace *nmt_compute_power_spectra(nmt_field *fl1,nmt_field *fl2,
					 nmt_binning_scheme *bin,nmt_workspace *w0,
					 flouble **cl_noise,flouble **cl_proposal,flouble **cl_out);

/**
 * @brief Flat-sky Gaussian covariance matrix
 *
 * Structure containing the information necessary to compute Gaussian covariance matrices
 * for the pseudo-CL spectra of two flat-sky spin-0 fields.
 *
 * @warning All covariance-related functionality is still under development, and in the future will hopefully support.
 * fields of different spins.
 */
typedef struct {
  int ncls_a; //!< Number of elements for the first set of power spectra (1 for the time being)
  int ncls_b; //!< Number of elements for the second set of power spectra (1 for the time being)
  nmt_binning_scheme_flat *bin; //!< Bandpowers defining the binning
  flouble **xi_1122; //!< First (a1b1-a2b2) mode coupling matrix (see scientific documentation)
  flouble **xi_1221; //!< Second (a1b2-a2b1) mode coupling matrix (see scientific documentation)
  gsl_matrix *coupling_binned_a; //!< Coupling matrix associated to the first set of power spectra
  gsl_matrix *coupling_binned_b; //!< Coupling matrix associated to the second set of power spectra
  gsl_permutation *coupling_binned_perm_a; //!< GSL aid to invert first MCM
  gsl_permutation *coupling_binned_perm_b; //!< GSL aid to invert second MCM
} nmt_covar_workspace_flat;

/**
 * @brief nmt_covar_workspace_flat destructor.
 * @warning All covariance-related functionality is still under development, and in the future will hopefully support.
 */
void nmt_covar_workspace_flat_free(nmt_covar_workspace_flat *cw);

/**
 * @brief nmt_covar_workspace_flat constructor
 *
 * Builds an nmt_covar_workspace_flat structure from two nmt_workspace_flat structures, corresponding
 * to the two sets of power spectra for which the covariance is required.
 * @param wa nmt_workspace_flat for the first set of power spectra.
 * @param wb nmt_workspace_flat for the second set of power spectra.
 * @warning All covariance-related functionality is still under development, and in the future will hopefully support.
 */
nmt_covar_workspace_flat *nmt_covar_workspace_flat_init(nmt_workspace_flat *wa,nmt_workspace_flat  *wb);

/**
 * @brief Compute flat-sky Gaussian covariance matrix
 * 
 * Computes the covariance matrix for two sets of power spectra given input predicted spectra
 * and a nmt_covar_workspace_flat structure.
 * @param cw nmt_covar_workspace_flat structure containing the information necessary to compute the
          covariance matrix.
 * @param nl Number of multipoles in which input power spectra are computed.
 * @param larr Array of multipoles in which input power spectra are computed.
 * @param cla1b1 Cross-power spectrum between field 1 in set a and field 1 in set b.
 * @param cla1b2 Cross-power spectrum between field 1 in set a and field 2 in set b.
 * @param cla2b1 Cross-power spectrum between field 2 in set a and field 1 in set b.
 * @param cla2b2 Cross-power spectrum between field 2 in set a and field 2 in set b.
 * @param covar_out flattened covariance matrix. Should be allocated to shape [nbpw_a * nbpw_b],
          where nbpw_a is the number of bandpowers in the set a of pseudo-CL-estimated
	  power spectra (and analogously for nbpw_b).
 * @warning All covariance-related functionality is still under development, and in the future will hopefully support.
 */
void nmt_compute_gaussian_covariance_flat(nmt_covar_workspace_flat *cw,
					  int nl,flouble *larr,flouble *cla1b1,flouble *cla1b2,
					  flouble *cla2b1,flouble *cla2b2,flouble *covar_out);

/**
 * @brief Saves nmt_covar_workspace_flat structure to file
 *
 * The output file uses a native binary format. In combination with nmt_covar_workspace_flat_read(),
 * this can be used to save the information contained in a given workspace and reuse it for 
 * future covariance matrix computations. The same workspace can be used on any pair of power spectra
 * between fields with the same masks.
 * @param cw nmt_covar_workspace_flat to be saved.
 * @param fname Path to output file.
 * @warning All covariance-related functionality is still under development, and in the future will hopefully support.
 */
void nmt_covar_workspace_flat_write(nmt_covar_workspace_flat *cw,char *fname);

/**
 * @brief Builds nmt_covar_workspace_flat structure from file
 *
 * The input file uses a native binary format. In combination with nmt_covar_workspace_flat_write(),
 * this can be used to save the information contained in a given workspace and reuse it for 
 * future covariance matrix computations. The same workspace can be used on any pair of power spectra
 * between fields with the same masks.
 * @param fname Path to input file.
 * @warning All covariance-related functionality is still under development, and in the future will hopefully support.
 */
nmt_covar_workspace_flat *nmt_covar_workspace_flat_read(char *fname);
  
/**
 * @brief Full-sky Gaussian covariance matrix
 *
 * Structure containing the information necessary to compute Gaussian covariance matrices
 * for the pseudo-CL spectra of two full-sky spin-0 fields.
 *
 * @warning All covariance-related functionality is still under development, and in the future will hopefully support.
 * fields of different spins.
 */
typedef struct {
  int lmax_a; //!< Maximum multipole for the first set of power spectra
  int lmax_b; //!< Maximum multipole for the second set of power spectra
  int ncls_a; //!< Number of elements for the first set of power spectra (1 for the time being)
  int ncls_b; //!< Number of elements for the second set of power spectra (1 for the time being)
  nmt_binning_scheme *bin_a; //!< Bandpowers defining the binning for the first set of spectra
  nmt_binning_scheme *bin_b; //!< Bandpowers defining the binning for the second set of spectra
  int nside; //!< HEALPix resolution parameter
  flouble **xi_1122; //!< First (a1b1-a2b2) mode coupling matrix (see scientific documentation)
  flouble **xi_1221; //!< Second (a1b2-a2b1) mode coupling matrix (see scientific documentation)
  gsl_matrix *coupling_binned_a; //!< Coupling matrix associated to the first set of power spectra
  gsl_matrix *coupling_binned_b; //!< Coupling matrix associated to the second set of power spectra
  gsl_permutation *coupling_binned_perm_a; //!< GSL aid to invert first MCM
  gsl_permutation *coupling_binned_perm_b; //!< GSL aid to invert second MCM
} nmt_covar_workspace;

/**
 * @brief nmt_covar_workspace destructor.
 * @warning All covariance-related functionality is still under development, and in the future will hopefully support.
 */
void nmt_covar_workspace_free(nmt_covar_workspace *cw);

/**
 * @brief nmt_covar_workspace constructor
 *
 * Builds an nmt_covar_workspace structure from two nmt_workspace structures, corresponding
 * to the two sets of power spectra for which the covariance is required.
 * @param wa nmt_workspace for the first set of power spectra.
 * @param wb nmt_workspace for the second set of power spectra.
 * @warning All covariance-related functionality is still under development, and in the future will hopefully support.
 */
nmt_covar_workspace *nmt_covar_workspace_init(nmt_workspace *wa,nmt_workspace *wb);

/**
 * @brief Compute full-sky Gaussian covariance matrix
 * 
 * Computes the covariance matrix for two sets of power spectra given input predicted spectra
 * and a nmt_covar_workspace structure.
 * @param cw nmt_covar_workspace structure containing the information necessary to compute the
          covariance matrix.
 * @param cla1b1 Cross-power spectrum between field 1 in set a and field 1 in set b.
          All power spectra should be defined for all ell < 3 * \p nside - 1.
 * @param cla1b2 Cross-power spectrum between field 1 in set a and field 2 in set b.
 * @param cla2b1 Cross-power spectrum between field 2 in set a and field 1 in set b.
 * @param cla2b2 Cross-power spectrum between field 2 in set a and field 2 in set b.
 * @param covar_out flattened covariance matrix. Should be allocated to shape [nbpw_a * nbpw_b],
          where nbpw_a is the number of bandpowers in the set a of pseudo-CL-estimated
	  power spectra (and analogously for nbpw_b).
 * @warning All covariance-related functionality is still under development, and in the future will hopefully support.
 */
void  nmt_compute_gaussian_covariance(nmt_covar_workspace *cw,
				      flouble *cla1b1,flouble *cla1b2,flouble *cla2b1,flouble *cla2b2,
				      flouble *covar_out);

/**
 * @brief Saves nmt_covar_workspace structure to file
 *
 * The output file uses a native binary format. In combination with nmt_covar_workspace_read(),
 * this can be used to save the information contained in a given workspace and reuse it for 
 * future covariance matrix computations. The same workspace can be used on any pair of power spectra
 * between fields with the same masks.
 * @param cw nmt_covar_workspace to be saved.
 * @param fname Path to output file.
 * @warning All covariance-related functionality is still under development, and in the future will hopefully support.
 */
void nmt_covar_workspace_write(nmt_covar_workspace *cw,char *fname);

/**
 * @brief Builds nmt_covar_workspace structure from file
 *
 * The input file uses a native binary format. In combination with nmt_covar_workspace_write(),
 * this can be used to save the information contained in a given workspace and reuse it for 
 * future covariance matrix computations. The same workspace can be used on any pair of power spectra
 * between fields with the same masks.
 * @param fname Path to input file.
 * @warning All covariance-related functionality is still under development, and in the future will hopefully support.
 */
nmt_covar_workspace *nmt_covar_workspace_read(char *fname);

#endif //_NAMASTER_H_
