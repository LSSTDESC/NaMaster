#ifndef _NM_UTILS_
#define _NM_UTILS_
#include "namaster.h"

//Defined in utils.c
gsl_rng *init_rng(unsigned int seed);
double rng_01(gsl_rng *rng);
int rng_poisson(double lambda,gsl_rng *rng);
void rng_delta_gauss(double *module,double *phase,
		     gsl_rng *rng,double sigma2);
void rng_gauss(gsl_rng *rng,double *r1,double *r2);
void end_rng(gsl_rng *rng);
int my_linecount(FILE *f);
void report_error(int level,char *fmt,...);
void *my_malloc(size_t size);
void *my_calloc(size_t nmemb,size_t size);
FILE *my_fopen(const char *path,const char *mode);
size_t my_fwrite(const void *ptr, size_t size, size_t nmemb,FILE *stream);
size_t my_fread(void *ptr,size_t size,size_t count,FILE *stream);
//Returns all non-zero wigner-3j symbols
// il2 (in) : l2
// il3 (in) : l3
// im2 (in) : m2
// im3 (in) : m3
// l1min_out (out) : min value for l1
// l1max_out (out) : max value for l1
// thrcof (out) : array with the values of the wigner-3j
// size (in) : size allocated for thrcof
int drc3jj(int il2,int il3,int im2, int im3,int *l1min_out,
	   int *l1max_out,double *thrcof,int size);
// Moore-penrose pseudo-inverse
void moore_penrose_pinv(gsl_matrix *M,double threshold);

//Defined in flatsky_utils.c
void *dftw_malloc(size_t n);
void dftw_free(void *p);
void fs_map_product(nmt_flatsky_info *fs,flouble *mp1,flouble *mp2,flouble *mp_out);
flouble fs_map_dot(nmt_flatsky_info *fs,flouble *mp1,flouble *mp2);
void fs_map2alm(nmt_flatsky_info *fs,int ntrans,int spin,flouble **map,fcomplex **alm);
void fs_alm2map(nmt_flatsky_info *fs,int ntrans,int spin,flouble **map,fcomplex **alm);
nmt_k_function *fs_generate_beam_window(double fwhm_amin);
void fs_zero_alm(nmt_flatsky_info *fs,fcomplex *alm);
void fs_alter_alm(nmt_flatsky_info *fs,double fwhm_amin,fcomplex *alm_in,fcomplex *alm_out,
		  nmt_k_function *window,int add_to_out);
void fs_alm2cl(nmt_flatsky_info *fs,nmt_binning_scheme_flat *bin,
	       fcomplex **alms_1,fcomplex **alms_2,int pol_1,int pol_2,flouble **cls,
	       flouble lmn_x,flouble lmx_x,flouble lmn_y,flouble lmx_y);
void fs_anafast(nmt_flatsky_info *fs,nmt_binning_scheme_flat *bin,
		flouble **maps_1,flouble **maps_2,int pol_1,int pol_2,flouble **cls);
fcomplex **fs_synalm(int nx,int ny,flouble lx,flouble ly,int nmaps,
		     nmt_k_function **cells,nmt_k_function **beam,int seed);

//Defined in healpix_extra.c
#define HE_NITER_DEFAULT 3
long he_nside2npix(long nside);
void he_pix2vec_ring(long nside, long ipix, double *vec);
long he_ang2pix(long nside,double cth,double phi);
long he_nalms(int lmax);
long he_indexlm(int l,int m,int lmax);
void he_alm2map(int nside,int lmax,int ntrans,int spin,flouble **maps,fcomplex **alms);
void he_map2alm(int nside,int lmax,int ntrans,int spin,flouble **maps,fcomplex **alms,int niter);
void he_alm2cl(fcomplex **alms_1,fcomplex **alms_2,int pol_1,int pol_2,flouble **cls,int lmax);
void he_anafast(flouble **maps_1,flouble **maps_2,int pol_1,int pol_2,flouble **cls,
		int nside,int lmax,int iter);
void he_write_healpix_map(flouble **tmap,int nfields,long nside,char *fname);
void he_get_file_params(char *fname,long *nside,int *nfields,int *isnest);
flouble *he_read_healpix_map(char *fname,long *nside,int nfield);
int he_ring_num(long nside,double z);
void he_query_strip(long nside,double theta1,double theta2,int *pixlist,long *npix_strip);
void he_ring2nest_inplace(flouble *map_in,long nside);
void he_nest2ring_inplace(flouble *map_in,long nside);
void he_in_ring(int nside,int iz,flouble phi0,flouble dphi,int *listir,int *nir);
void he_query_disc(int nside,double cth0,double phi,flouble radius,int *listtot,int *nlist,int inclusive);
void he_udgrade(flouble *map_in,long nside_in,flouble *map_out,long nside_out,int nest);
double *he_generate_beam_window(int lmax,double fwhm_amin);
void he_zero_alm(int lmax,fcomplex *alm);
void he_alter_alm(int lmax,double fwhm_amin,fcomplex *alm_in,fcomplex *alm_out,double *window,int add_to_out);
void he_map_product(int nside,flouble *mp1,flouble *mp2,flouble *mp_out);
flouble he_map_dot(int nside,flouble *mp1,flouble *mp2);
fcomplex **he_synalm(int nside,int nmaps,int lmax,flouble **cells,flouble **beam,int seed);
#ifdef _WITH_NEEDLET
#define HE_NBAND_NX 512
#define HE_NORM_FT 2.2522836206907617
#define HE_NL_INTPREC 1E-6
#define HE_NT_NSIDE_MIN 32
typedef struct {
  double b;
  double inv_b;
  gsl_spline *b_spline;
  gsl_interp_accel *b_intacc;
  int niter;
  int nside0;
  int jmax_min;
  int nj;
  int *nside_arr;
  int *lmax_arr;
  flouble **b_arr;
} he_needlet_params;
void he_nt_end(he_needlet_params *par);
he_needlet_params *he_nt_init(flouble b_nt,int nside0,int niter);
void he_free_needlet(he_needlet_params *par,int pol,flouble ***nt);
flouble ***he_alloc_needlet(he_needlet_params *par,int pol);
void he_nt_get_window(he_needlet_params *par,int j,flouble *b);
fcomplex **he_needlet2map(he_needlet_params *par,flouble **map,flouble ***nt,
			  int return_alm,int pol,int input_TEB,int output_TEB);
fcomplex **he_map2needlet(he_needlet_params *par,flouble **map,flouble ***nt,
			  int return_alm,int pol,int input_TEB,int output_TEB);
#endif //_WITH_NEEDLET

#endif //_NM_UTILS_
