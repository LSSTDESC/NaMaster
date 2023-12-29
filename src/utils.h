#ifndef _NM_UTILS_
#define _NM_UTILS_

#ifndef NO_DOXY
#include "namaster.h"

#include <setjmp.h>

#define EXIT_ON_ERROR 0
#define THROW_ON_ERROR 1
extern jmp_buf nmt_exception_buffer;
extern int nmt_exception_status;
extern int nmt_error_policy;
extern char nmt_error_message[256];

#define try if ((nmt_exception_status = setjmp(nmt_exception_buffer)) == 0)
#define catch(val) else if (nmt_exception_status == val)
#define throw(val) longjmp(nmt_exception_buffer,val)
#define finally else

#define NMT_ERROR_MEMORY 1001
#define NMT_ERROR_FOPEN 1002
#define NMT_ERROR_WRITE 1003
#define NMT_ERROR_READ 1004
#define NMT_ERROR_WIG3J 1005
#define NMT_ERROR_PINV 1006
#define NMT_ERROR_BWEIGHT 1007
#define NMT_ERROR_COVAR 1008
#define NMT_ERROR_CONSISTENT_RESO 1009
#define NMT_ERROR_BADNO 1010
#define NMT_ERROR_APO 1011
#define NMT_ERROR_HPX 1012
#define NMT_ERROR_INCONSISTENT 1013
#define NMT_ERROR_VALUE 1014
#define NMT_ERROR_NOT_IMPLEMENTED 1015
#define NMT_ERROR_LITE 1016
#endif //NO_DOXY

/**
 * @brief Initialize a random number generator
 *
 * @param seed Seed.
 * @return A GSL random number generator object.
 */
gsl_rng *init_rng(unsigned int seed);

/**
 * @brief Top-hat random numbers
 *
 * Draw a random number from 0 to 1 with a top-hat distribution.
 * @param rng A random number generator.
 * @return Random number.
 */
double rng_01(gsl_rng *rng);

/**
 * @brief Gaussian random numbers.
 *
 * Draw a pair of Gaussian random numbers with zero mean and unit variance.
 * @param rng A random number generator.
 * @param r1 First output random number.
 * @param r2 Second output random number.
 */
void rng_gauss(gsl_rng *rng,double *r1,double *r2);

/**
 * @brief Destructor for random number generators.
 */
void end_rng(gsl_rng *rng);

#ifndef NO_DOXY
void set_error_policy(int i);
void report_error(int level,char *fmt,...);
#endif //NO_DOXY

void *my_malloc(size_t size);
void *my_calloc(size_t nmemb,size_t size);

/**
 * @brief Wigner 3-j symbols
 *
 * Returns all non-zero wigner-3j symbols
 * \f[
 *    \left(
 *    \begin{array}{ccc}
 *      \ell_1 & \ell_2 & \ell_3 \\
 *      m_1    & m_2    & m_3
 *     \end{array}
 *     \right),
 * \f]
 * for
 * @param il2 =\f$\ell_2\f$
 * @param im2 =\f$m_2\f$
 * @param il3 =\f$\ell_3\f$
 * @param im3 =\f$m_3\f$
 * @param l1min_out Minimum value of \f$\ell_1\f$ allowed by selection rules (output).
 * @param l1max_out Maximum value of \f$\ell_1\f$ allowed by selection rules (output).
 * @param thrcof Output array that will contain the values of the wigner-3j symbols for \p l1min_out \f$\leq\ell_1\leq\f$ \p l1max_out.
 * @param size Number of elements allocated for thrcof.
 *
 * Note that the selection rule \f$m_1+m_2+m_3=0\f$ completely fix \f$m_3\f$.
 */
int drc3jj(int il2,int il3,int im2, int im3,int *l1min_out,
	   int *l1max_out,double *thrcof,int size);

/**
 * @brief Wigner 3-j symbols
 *
 * Returns all non-zero wigner-3j symbols with m=0
 * \f[
 *    \left(
 *    \begin{array}{ccc}
 *      \ell_1 & \ell_2 & \ell_3 \\
 *      0      & 0      & 0
 *     \end{array}
 *     \right),
 * \f]
 * for
 * @param il2 =\f$\ell_2\f$
 * @param il3 =\f$\ell_3\f$
 * @param l1min_out Minimum value of \f$\ell_1\f$ allowed by selection rules (output).
 * @param l1max_out Maximum value of \f$\ell_1\f$ allowed by selection rules (output).
 * @param thrcof Output array that will contain the values of the wigner-3j symbols for \p l1min_out \f$\leq\ell_1\leq\f$ \p l1max_out.
 * @param size Number of elements allocated for thrcof.
 */
int drc3jj_000(int il2,int il3,int *l1min_out,int *l1max_out,
	       double *lfac,double *thrcof,int size);

/**
 * @brief Moore-Penrose pseudo-inverse.
 *
 * Returns the <a href="https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse">Moore-Penrose pseudo-inverse</a>.
 *
 * @param M Input matrix to invert. Will be overwritten with inverse.
 * @param threshold When inverting, all eigenvalues smaller than \p threshold times
          the largest eigenvalue will be discarded (i.e. the corresponding eigenvalue
	  of the pseudo-inverse will be set to zero).
 */
void moore_penrose_pinv(gsl_matrix *M,double threshold);

/**
 * @brief Wrapper around <a href="http://www.fftw.org/fftw3_doc/Memory-Allocation.html"> fftw_malloc </a>
 *
 * Wrapping function to support simultaneously single and double precision FFTW memory allocation.
 * @param n Number of bytes to allocate
 * @return Allocated pointer.
 */
void *dftw_malloc(size_t n);

/**
 * @brief Wrapper around <a href="http://www.fftw.org/fftw3_doc/Memory-Allocation.html"> fftw_free </a>
 *
 * Wrapping function to support simultaneously single and double precision FFTW memory freeing.
 * @param p Pointer to free.
 */
void dftw_free(void *p);

/**
 * @brief Copies a map.
 *
 * @param fs nmt_flatsky_info structure describing the flat-sky patch.
 * @param destmap Map to copy into.
 * @param destmap Map to copy from.
 */
void fs_mapcpy(nmt_flatsky_info *fs,flouble *destmap,flouble *srcmap);

/**
 * @brief Multiplies two flat-sky maps.
 *
 * @param fs nmt_flatsky_info structure describing the flat-sky patch.
 * @param mp1 First map to multiply
 * @param mp2 Second map to multiply.
 * @param mp_out Output map containing the product of \p mp1 and \p mp2. It is safe to
          pass either of the input maps as \p mp_out, in which case that map will be
	  overwritten with the product.
 */
void fs_map_product(nmt_flatsky_info *fs,flouble *mp1,flouble *mp2,flouble *mp_out);

/**
 * @brief Dot product of flat-sky maps.
 *
 * Computes the integral over the patch of the product of two maps:
 * \f[
 *    \int_\Omega d\Omega\, m_1(\hat{\bf n})\,m_2(\hat{\bf n}),
 * \f]
 * where \f$\Omega\f$ is the flat-sky patch defined by \p fs. The integral is computed as a
 * Riemann sum over all pixels in the map.
 * @param fs nmt_flatsky_info structure describing the flat-sky patch.
 * @param mp1 First map to multiply.
 * @param mp2 Second map to multiply.
 * @return Dot product.
 */
flouble fs_map_dot(nmt_flatsky_info *fs,flouble *mp1,flouble *mp2);

/**
 * @brief Flat-sky SHT
 *
 * Computes the direct SHT of a set of spin-s flat-sky fields.
 * See scientific documentation and companion paper for further details.
 * @param fs nmt_flatsky_info structure describing the flat-sky patch.
 * @param ntrans Number of transfoms to carry out.
 * @param spin Spin of the fields to transform (0 or 2).
 * @param map Maps to transform. Must have shape [\p ntrans * \p nmap][\p nx * \p ny],
          where \p nmap is 1 or 2 for spin-0 and spin-2 respectively, and (\p nx,\p ny)
	  are the dimensions of the patch in the (x,y) directions as defined by \p fs.
 * @param alm Will hold the output SHT coefficients. Must have shape
          [\p ntrans * \p nmap][\p ny * (\p nx /2+1)], where \p nmap, \p nx and \p ny
	  have been defined above.
 */
void fs_map2alm(nmt_flatsky_info *fs,int ntrans,int spin,flouble **map,fcomplex **alm);

/**
 * @brief Flat-sky inverse SHT
 *
 * Computes the inverse SHT of a set of spin-s flat-sky fields.
 * See scientific documentation and companion paper for further details.
 * @param fs nmt_flatsky_info structure describing the flat-sky patch.
 * @param ntrans Number of transfoms to carry out.
 * @param spin Spin of the fields to transform (0 or 2).
 * @param map Will hold the output maps. Must have shape [\p ntrans * \p nmap][\p nx * \p ny],
          where \p nmap is 1 or 2 for spin-0 and spin-2 respectively, and (\p nx,\p ny)
	  are the dimensions of the patch in the (x,y) directions as defined by \p fs.
 * @param alm SHT coefficients to inverse-transform. Must have shape
          [\p ntrans * \p nmap][\p ny * (\p nx /2+1)], where \p nmap, \p nx and \p ny
	  have been defined above.
 */
void fs_alm2map(nmt_flatsky_info *fs,int ntrans,int spin,flouble **map,fcomplex **alm);

/**
 * @brief Gaussian beam
 *
 * Generates an nmt_k_function structure defining a circularly symmetric Gaussian beam
 * for flat-sky fields.
 * @param fwhm_amin Full-width at half-maximum of the beam in arcminutes.
 * @return nmt_k_function defining the beam.
 */
nmt_k_function *fs_generate_beam_window(double fwhm_amin);

/**
 * @brief Zero SHT coefficients.
 *
 * Sets all elements of a set of flat-sky Fourier coefficients to zero.
 * @param fs nmt_flatsky_info structure describing the flat-sky patch.
 * @param alm Set of Fourier coefficients to zero.
          See fs_map2alm() for the expected shape of flat-sky Fourier coefficients.
 */
void fs_zero_alm(nmt_flatsky_info *fs,fcomplex *alm);

/**
 * @brief Multiply SHT coefficients by beam.
 *
 * Multiplies a given set of flat-sky Fourier coefficients by a circularly-symmetric function.
 * @param fs nmt_flatsky_info structure describing the flat-sky patch.
 * @param fwhm_amin Full-width at half-maximum of the Gaussian beam in arcminutes.
          Only used if \p window is a NULL pointer.
 * @param alm_in Input Fourier coefficients.
 * @param alm_out Output Fourier coefficients.
 * @param window nmt_k_function defining the function to multiply the coefficients by.
          Pass a NULL pointer if you want a Gaussian beam with a FWHM defined by \p fwhm_amin.
 * @param add_to_out If >0, the result of multiplying \p alm_in with the window function
          will be added to the current contents of \p alm_out. Otherwise, \p alm_out is
	  overwritten with the product.
 */
void fs_alter_alm(nmt_flatsky_info *fs,double fwhm_amin,fcomplex *alm_in,fcomplex *alm_out,
		  nmt_k_function *window,int add_to_out);

/**
 * @brief Computes Flat-sky power spectrum from Fourier coefficients.
 *
 * Bins the product of two sets of Fourier coefficients into bandpowers.
 * @param fs nmt_flatsky_info structure describing the flat-sky patch.
 * @param bin nmt_binning_scheme_flat structure defining the bandpowers to use.
 * @param alms_1 First set of Fourier coefficients to correlate.
 * @param alms_2 Second set of Fourier coefficients to correlate.
 * @param spin_1 alms_1 spin.
 * @param spin_2 alms_2 spin.
 * @param cls Will hold the output power spectra. Should have shape [\p ncls][\p nbands],
          where \p ncls is the appropriate number of power spectra given the
	  spins of the input fields  (e.g. \p ncls = 2*2 = 4 if both fields have
	  spin=2) and \p nbpw is the number of bandpowers defined by \p bin.
 * @param lmn_x Lower end of the range of multipoles in the x direction that should be masked.
 * @param lmx_x Upper end of the range of multipoles in the x direction that should be masked.
 *        if \p lmx_x < \p lmn_x, no Fourier-space masked is performed.
 * @param lmn_y Same as \p lmn_x for the y direction.
 * @param lmx_y Same as \p lmx_x for the y direction.
 */
void fs_alm2cl(nmt_flatsky_info *fs,nmt_binning_scheme_flat *bin,
	       fcomplex **alms_1,fcomplex **alms_2,int spin_1,int spin_2,flouble **cls,
	       flouble lmn_x,flouble lmx_x,flouble lmn_y,flouble lmx_y);

/**
 * @brief Computes Flat-sky power spectrum from maps.
 *
 * Similar to fs_alm2cl() but starting from maps.
 * @param fs nmt_flatsky_info structure describing the flat-sky patch.
 * @param bin nmt_binning_scheme_flat structure defining the bandpowers to use.
 * @param maps_1 First set of maps to correlate.
 * @param maps_2 Second set of maps to correlate.
 * @param spin_1 maps_1 spin.
 * @param spin_2 maps_2 spin.
 * @param cls Will hold the output power spectra. Should have shape [\p ncls][\p nbands],
          where \p ncls is the appropriate number of power spectra given the
	  spins of the input fields  (e.g. \p ncls = 2*2 = 4 if both fields have
	  spin=2). and \p nbpw is the number of bandpowers defined by \p bin.
 */
void fs_anafast(nmt_flatsky_info *fs,nmt_binning_scheme_flat *bin,
		flouble **maps_1,flouble **maps_2,int spin_1,int spin_2,flouble **cls);

/**
 * @brief Gaussian realizations of flat-sky Fourier coefficients.
 *
 * Generates a Gaussian realization of a set of Fourier coefficients given an input power spectrum.
 * @param nx Number of grid points in the x dimension.
 * @param ny Number of grid points in the y dimension.
 * @param lx Length of the x dimension (in steradians).
 * @param ly Length of the y dimension (in steradians).
 * @param nmaps Number of fields to generate.
 * @param cells Set of \p nmaps * (\p nmaps + 1) / 2 nmt_k_function structures defining each of the
          power spectra needed to generate the Fourier coefficients. It must contain only the
	  the upper-triangular part in row-major order (e.g. if \p nmaps is 3, there will be 6
	  power spectra ordered as [1-1,1-2,1-3,2-2,2-3,3-3].
 * @param beam Set of \p nmaps nmt_k_function structures defining the beam of each field.
 * @param seed Seed for this particular realization.
 * @return Gaussian realization with shape [\p nmaps][\p ny * (\p nx /2 +1)].
 */
fcomplex **fs_synalm(int nx,int ny,flouble lx,flouble ly,int nmaps,
		     nmt_k_function **cells,nmt_k_function **beam,int seed);

/**
 * @brief Reads flat-sky map
 *
 * Reads a flat-sky map from a FITS file. The flat map should be in an image HDU with
 * WCS header keywords defining the sky patch (read from file).
 * @param fname Path to input file.
 * @param nx Number of grid points in the x dimension (read from file).
 * @param ny Number of grid points in the y dimension (read from file).
 * @param lx Length of the x dimension (in steradians) (read from file).
 * @param ly Length of the y dimension (in steradians) (read from file).
 * @param nfield Which field to read (i.e. HDU number to read the map from, starting from 0).
 * @return Read map (with size \p ny * \p nx).
 */
flouble *fs_read_flat_map(char *fname,int *nx,int *ny,flouble *lx,flouble *ly,int nfield);


#define HE_NITER_DEFAULT 3 //!< Default number of iterations used for full-sky spherical harmonic transforms

/**
 * @brief HEALPix number of pixels.
 *
 * @param nside HEALPix resolution parameter
 * @return 12 * \p nside * \p nside.
 */
long he_nside2npix(long nside);

/**
 * @brief HEALPix pix2vec
 *
 * Returns normal vector pointing in the direction of a given pixel.
 * @param nside HEALPix resolution parameter.
 * @param ipix Pixel index in RING ordering.
 * @param vec Output vector.
 */
void he_pix2vec_ring(long nside, long ipix, double *vec);

/**
 * @brief Modified HEALPix ang2pix
 *
 * Returns pixel containing a given point in the sphere.
 * @param nside HEALPix resolution parameter
 * @param phi Azimuth spherical coordinate \f$\varphi\f$ (in radians).
 * @param cth Cosine of inclination spherical coordinate \f$\theta\f$.
 * @return Pixel index.
 */
long he_ang2pix(long nside,double cth,double phi);

/**
 * @brief HEALPix ring number.
 *
 * @param nside HEALPix resolution parameter.
 * @param z z (cos(theta)) coordinate.
 * @return ring number
 */
int he_ring_num(long nside,double z);

/**
 * @brief Returns pixel indices in a given latitude strip
 *
 * @param nside HEALPix resolution parameter
 * @param theta1 Lower edge of latitude range.
 * @param theta2 Upper edge of latitude range.
 * @param pixlist Output list of pixels
 * @param npix_strip On input, it should hold the number
          of elements allocated in \p pixlist. On output,
	  it contains the number of pixels in the strip.
 */
void he_query_strip(long nside,double theta1,double theta2,int *pixlist,long *npix_strip);

/**
 * @brief Transform from RING to NEST
 *
 * Transforms a HEALPix map from RING to NEST ordering in place.
 * @param map_in Map to transform.
 * @param nside HEALPix resolution parameter
 */
void he_ring2nest_inplace(flouble *map_in,long nside);

/**
 * @brief Transform from NEST to RING
 *
 * Transforms a HEALPix map from NEST to RING ordering in place.
 * @param map_in Map to transform.
 * @param nside HEALPix resolution parameter
 */
void he_nest2ring_inplace(flouble *map_in,long nside);

/**
 * @brief Returns pixel indices in a given ring.
 *
 * @param nside HEALPix resolution parameter
 * @param iz Ring index.
 * @param phi0 Center of azimuth range.
 * @param dphi Width of azimuth range.
 * @param listir Output list of pixels
 * @param nir On input, it should hold the number
          of elements allocated in \p listir. On output,
	  it contains the number of pixels in the ring.
 */
void he_in_ring(int nside,int iz,flouble phi0,flouble dphi,int *listir,int *nir);

/**
 * @brief Returns pixel indices in a given disc.
 *
 * @param nside HEALPix resolution parameter
 * @param cth0 \f$\cos(\theta)\f$ for the disc centre.
 * @param phi Azimuth for the disc centre.
 * @param radius Disc radius in radians.
 * @param listtot Output list of pixels
 * @param nlist On input, it should hold the number
          of elements allocated in \p listtot. On output,
	  it contains the number of pixels in the disc.
 * @param inclusive If >0, include all pixels that are
          partially insie the disc. Otherwise include only
	  pixels whose centres are inside the disc.
 */
void he_query_disc(int nside,double cth0,double phi,flouble radius,int *listtot,int *nlist,
		   int inclusive);

/**
 * @brief Up/down grade map resolution.
 *
 * @param map_in Input map.
 * @param nside_in Input HEALPix resolution parameter.
 * @param map_out Output map.
 * @param nside_out Output HEALPix resolution parameter.
 * @param nest If >0, intput and output maps are in NESTED ordering.
 */
void he_udgrade(flouble *map_in,long nside_in,flouble *map_out,long nside_out,int nest);

/**
 * @brief Computes pixel area
 *
 * @param nside nside parameter.
 * @param i ring number.
 * @return pixel area in sterad.
 */
flouble he_get_pix_area(long nside,long i);

/**
 * @brief Multiplies two full-sky maps.
 *
 * @param nside nside parameter.
 * @param mp1 First map to multiply
 * @param mp2 Second map to multiply.
 * @param mp_out Output map containing the product of \p mp1 and \p mp2. It is safe to
          pass either of the input maps as \p mp_out, in which case that map will be
	  overwritten with the product.
 */
void he_map_product(long nside,flouble *mp1,flouble *mp2,flouble *mp_out);

/**
 * @brief Dot product of full-sky maps.
 *
 * Computes the integral over the full sphere of the product of two maps:
 * \f[
 *    \int d\Omega\, m_1(\hat{\bf n})\,m_2(\hat{\bf n}),
 * \f]
 * The integral is computed as a Riemann sum over all pixels in the map.
 * @param nside nside parameter.
 * @param mp1 First map to multiply.
 * @param mp2 Second map to multiply.
 * @return Dot product.
 */
flouble he_map_dot(long nside,flouble *mp1,flouble *mp2);

int cov_get_coupling_pair_index(int na,int nc,int nb,int nd,
				int ia1,int ia2,int ic1,int ic2,
				int ib1,int ib2,int id1,int id2);

#endif //_NM_UTILS_
