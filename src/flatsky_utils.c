#include "config.h"
#include "utils.h"
#include <fitsio.h>

void *dftw_malloc(size_t n)
{
#ifdef _SPREC
  void *p=fftwf_malloc(n);
#else //_SPREC
  void *p=fftw_malloc(n);
#endif //_SPREC
  if(p==NULL)
    report_error(NMT_ERROR_MEMORY,"Ran out of memory\n");
  return p;
}

void dftw_free(void *p)
{
#ifdef _SPREC
  fftwf_free(p);
#else //_SPREC
  fftw_free(p);
#endif //_SPREC
}

void fs_mapcpy(nmt_flatsky_info *fs,flouble *destmap,flouble *srcmap)
{
#pragma omp parallel default(none) \
  shared(fs,destmap,srcmap)
  {
    long ip;
#pragma omp for
    for(ip=0;ip<fs->npix;ip++) {
      destmap[ip]=srcmap[ip];
    } //end omp for
  } //end omp parallel
}
      
void fs_map_product(nmt_flatsky_info *fs,flouble *mp1,flouble *mp2,flouble *mp_out)
{
#pragma omp parallel default(none)		\
  shared(fs,mp1,mp2,mp_out)
  {
    long ip;

#pragma omp for
    for(ip=0;ip<fs->npix;ip++) {
      mp_out[ip]=mp1[ip]*mp2[ip];
    } //end omp for
  } //end omp parallel
}

flouble fs_map_dot(nmt_flatsky_info *fs,flouble *mp1,flouble *mp2)
{
  double sum=0;

#pragma omp parallel default(none)		\
  shared(mp1,mp2,sum,fs)
  {
    long ip;
    double sum_thr=0;
    
#pragma omp for
    for(ip=0;ip<fs->npix;ip++) {
      sum_thr+=mp1[ip]*mp2[ip];
    } //end omp for

#pragma omp critical
    {
      sum+=sum_thr;
    } //end omp critical
  } //end omp parallel

  return (flouble)(sum*fs->pixsize);
}  

static void qu2eb(nmt_flatsky_info *fs,int spin,fcomplex **alm)
{
  int sig_overall=-1;
  if(spin==0)
    sig_overall=1;

#pragma omp parallel default(none)		\
  shared(fs,spin,alm,sig_overall)
  {
    int iy;
    fcomplex sig=sig_overall*cpow(I,spin);
    flouble dkx=2*M_PI/fs->lx;
    flouble dky=2*M_PI/fs->ly;

#pragma omp for
    for(iy=0;iy<fs->ny;iy++) {
      int ix;
      flouble ky;
      if(2*iy<=fs->ny)
	ky=iy*dky;
      else
	ky=-(fs->ny-iy)*dky;
      for(ix=0;ix<=fs->nx/2;ix++) {
	flouble csphi,ssphi,cph,sph;
	fcomplex e,b;
	int s=0;
	flouble kx=ix*dkx;
	long index=ix+((long)(fs->nx/2+1))*iy;
	flouble kmod2=kx*kx+ky*ky;

	if(kmod2<=0) {
	  cph=1;
	  sph=0;
	}
	else {
	  flouble i_kmod=1./sqrt(kmod2);
	  cph=kx*i_kmod;
	  sph=ky*i_kmod;
	}

	csphi=1; ssphi=0;
	while(s<spin) {
	  flouble c2=csphi*cph-ssphi*sph;
	  flouble s2=ssphi*cph+csphi*sph;
	  csphi=c2;
	  ssphi=s2;
	  s++;
	}
	e=sig*(alm[0][index]*csphi-alm[1][index]*ssphi);
	b=sig*(alm[0][index]*ssphi+alm[1][index]*csphi);
	alm[0][index]=e;
	alm[1][index]=b;
      }
    } //end omp for
  } //end omp parallel
}

static void eb2qu(nmt_flatsky_info *fs,int spin,fcomplex **alm)
{
  int sig_overall=-1;
  if(spin==0)
    sig_overall=1;

#pragma omp parallel default(none)		\
  shared(fs,spin,alm,sig_overall)
  { 
    int iy;
    fcomplex sig=sig_overall*cpow(-I,spin);
    flouble dkx=2*M_PI/fs->lx;
    flouble dky=2*M_PI/fs->ly;

#pragma omp for
    for(iy=0;iy<fs->ny;iy++) {
      int ix;
      flouble ky;
      if(2*iy<=fs->ny)
	ky=iy*dky;
      else
	ky=-(fs->ny-iy)*dky;
      for(ix=0;ix<=fs->nx/2;ix++) {
	flouble csphi,ssphi,cph,sph;
	fcomplex q,u;
	int s=0;
	flouble kx=ix*dkx;
	long index=ix+((long)(fs->nx/2+1))*iy;
	flouble kmod2=kx*kx+ky*ky;
	
	if(kmod2<=0) {
	  cph=1;
	  sph=0;
	}
	else {
	  flouble i_kmod=1./sqrt(kmod2);
	  cph=kx*i_kmod;
	  sph=ky*i_kmod;
	}

	csphi=1; ssphi=0;
	while(s<spin) {
	  flouble c2=csphi*cph-ssphi*sph;
	  flouble s2=ssphi*cph+csphi*sph;
	  csphi=c2;
	  ssphi=s2;
	  s++;
	}

	q=sig*( alm[0][index]*csphi+alm[1][index]*ssphi);
	u=sig*(-alm[0][index]*ssphi+alm[1][index]*csphi);
	alm[0][index]=q;
	alm[1][index]=u;
      }
    } //end omp for
  } //end omp parallel
}

void fs_map2alm(nmt_flatsky_info *fs,int ntrans,int spin,flouble **map,fcomplex **alm)
{
  //TODO init threads??
#ifdef _SPREC
  fftwf_plan plan_ft;
#else //_SPREC
  fftw_plan plan_ft;
#endif //_SPREC
  int imap,nmaps=1;
  long nmodes=fs->ny*((long)(fs->nx/2+1));
  if(spin)
    nmaps=2;
  
  for(imap=0;imap<nmaps*ntrans;imap++) {
#ifdef _SPREC
    plan_ft=fftwf_plan_dft_r2c_2d(fs->ny,fs->nx,map[imap],alm[imap],FFTW_ESTIMATE);
    fftwf_execute(plan_ft);
    fftwf_destroy_plan(plan_ft);
#else //_SPREC
    plan_ft=fftw_plan_dft_r2c_2d(fs->ny,fs->nx,map[imap],alm[imap],FFTW_ESTIMATE);
    fftw_execute(plan_ft);
    fftw_destroy_plan(plan_ft);
#endif //_SPREC

#pragma omp parallel default(none) \
  shared(fs,alm,imap,nmodes)
    {
      long ipix;
      flouble norm=fs->lx*fs->ly/(2*M_PI*fs->nx*fs->ny);
#pragma omp for
      for(ipix=0;ipix<nmodes;ipix++) {
	alm[imap][ipix]*=norm;
      } //end omp for
    } //end omp parallel
  }

  if(nmaps>1) { //Q,U -> E,B
    for(imap=0;imap<ntrans*nmaps;imap+=nmaps)
      qu2eb(fs,spin,&(alm[imap]));
  }
}

void fs_alm2map(nmt_flatsky_info *fs,int ntrans,int spin,flouble **map,fcomplex **alm)
{
  //TODO init threads??
#ifdef _SPREC
  fftwf_plan plan_ft;
#else //_SPREC
  fftw_plan plan_ft;
#endif //_SPREC
  int imap,nmaps=1;
  if(spin)
    nmaps=2;
  
  if(nmaps>1) { //E,B -> Q,U
    for(imap=0;imap<ntrans*nmaps;imap+=nmaps)
      eb2qu(fs,spin,&(alm[imap]));
  }

  for(imap=0;imap<nmaps*ntrans;imap++) {
#ifdef _SPREC
    plan_ft=fftwf_plan_dft_c2r_2d(fs->ny,fs->nx,alm[imap],map[imap],FFTW_ESTIMATE);
    fftwf_execute(plan_ft);
    fftwf_destroy_plan(plan_ft);
#else //_SPREC
    plan_ft=fftw_plan_dft_c2r_2d(fs->ny,fs->nx,alm[imap],map[imap],FFTW_ESTIMATE);
    fftw_execute(plan_ft);
    fftw_destroy_plan(plan_ft);
#endif //_SPREC

#pragma omp parallel default(none)		\
  shared(fs,map,imap)
    {
      long ipix;
      flouble norm=2*M_PI/(fs->lx*fs->ly);
#pragma omp for
      for(ipix=0;ipix<fs->npix;ipix++) {
	map[imap][ipix]*=norm;
      } //end omp for
    } //end omp parallel
  }

  if(nmaps>1) { //Q,U -> E,B
    for(imap=0;imap<ntrans*nmaps;imap+=nmaps)
      qu2eb(fs,spin,&(alm[imap]));
  }
}

#define SAMP_RATE_SIGMA 128
#define FWHM2SIGMA_FLAT 0.00012352884853326381 
nmt_k_function *fs_generate_beam_window(double fwhm_amin)
{
  int ii;
  nmt_k_function *beam;
  flouble *larr=my_malloc(5*SAMP_RATE_SIGMA*sizeof(flouble));
  flouble *farr=my_malloc(5*SAMP_RATE_SIGMA*sizeof(flouble));
  double sigma=FWHM2SIGMA_FLAT*fwhm_amin;
  for(ii=0;ii<5*SAMP_RATE_SIGMA;ii++) {
    flouble l=(ii+0.0)/(SAMP_RATE_SIGMA*sigma);
    larr[ii]=l;
    farr[ii]=exp(-0.5*l*l*sigma*sigma);
  }

  beam=nmt_k_function_alloc(5*SAMP_RATE_SIGMA,larr,farr,1.,0.,0);
  free(larr);
  free(farr);

  return beam;
}

void fs_zero_alm(nmt_flatsky_info *fs,fcomplex *alm)
{

#pragma omp parallel default(none)		\
  shared(fs,alm)
  {
    long ii;
    long nmodes=fs->ny*((long)(fs->nx/2+1));
#pragma omp for
    for(ii=0;ii<nmodes;ii++) {
      alm[ii]=0;
    } //end omp for
  } //end omp parallel
}

void fs_alter_alm(nmt_flatsky_info *fs,double fwhm_amin,fcomplex *alm_in,fcomplex *alm_out,
		  nmt_k_function *window,int add_to_out)
{
  nmt_k_function *beam;
  if(window==NULL) beam=fs_generate_beam_window(fwhm_amin);
  else beam=window;

#pragma omp parallel default(none)		\
  shared(fs,alm_in,alm_out,beam,add_to_out)
  {
    int iy;
    flouble dkx=2*M_PI/fs->lx;
    flouble dky=2*M_PI/fs->ly;
    gsl_interp_accel *intacc_thr=gsl_interp_accel_alloc();

#pragma omp for
    for(iy=0;iy<fs->ny;iy++) {
      int ix;
      flouble ky;
      if(2*iy<=fs->ny)
	ky=iy*dky;
      else
	ky=-(fs->ny-iy)*dky;
      for(ix=0;ix<=fs->nx/2;ix++) {
	flouble kx=ix*dkx;
	long index=ix+((long)(fs->nx/2+1))*iy;
	flouble kmod=sqrt(kx*kx+ky*ky);
	if(add_to_out)
	  alm_out[index]+=alm_in[index]*nmt_k_function_eval(beam,kmod,intacc_thr);
	else
	  alm_out[index]=alm_in[index]*nmt_k_function_eval(beam,kmod,intacc_thr);
      }
    } //end omp for
    gsl_interp_accel_free(intacc_thr);
  } //end omp parallel

  if(window==NULL) nmt_k_function_free(beam);
}

void fs_alm2cl(nmt_flatsky_info *fs,nmt_binning_scheme_flat *bin,
	       fcomplex **alms_1,fcomplex **alms_2,int spin_1,int spin_2,flouble **cls,
	       flouble lmn_x,flouble lmx_x,flouble lmn_y,flouble lmx_y)
{
  int i1,nmaps_1=1,nmaps_2=1;
  int *n_cells=my_malloc(bin->n_bands*sizeof(int));
  if(spin_1) nmaps_1=2;
  if(spin_2) nmaps_2=2;

  for(i1=0;i1<nmaps_1;i1++) {
    int i2;
    fcomplex *alm1=alms_1[i1];
    for(i2=0;i2<nmaps_2;i2++) {
      int il;
      fcomplex *alm2=alms_2[i2];
      int index_cl=i2+nmaps_2*i1;
      flouble norm_factor=4*M_PI*M_PI/(fs->lx*fs->ly);
      for(il=0;il<bin->n_bands;il++) {
	cls[index_cl][il]=0;
	n_cells[il]=0;
      }

#pragma omp parallel default(none)		\
  shared(fs,bin,alm1,alm2,index_cl,cls)		\
  shared(lmn_x,lmx_x,lmn_y,lmx_y,n_cells)
      {
	int iy;
	flouble dkx=2*M_PI/fs->lx;
	flouble dky=2*M_PI/fs->ly;

#pragma omp for
	for(iy=0;iy<fs->ny;iy++) {
	  int ix;
	  flouble ky;
	  int ik=0;
	  if(2*iy<=fs->ny) ky=iy*dky;
	  else ky=-(fs->ny-iy)*dky;
	  if((ky>=lmn_y) && (ky<=lmx_y))
	    continue;
	  for(ix=0;ix<fs->nx;ix++) {
	    int ix_here;
	    long index;
	    flouble kmod,kx;
	    if(2*ix<=fs->nx) {
	      kx=ix*dkx;
	      ix_here=ix;
	    }
	    else {
	      kx=-(fs->nx-ix)*dkx;
	      ix_here=fs->nx-ix;
	    }
	    if((kx>=lmn_x) && (kx<=lmx_x))
	      continue;

	    index=ix_here+((long)(fs->nx/2+1))*iy;
	    kmod=sqrt(kx*kx+ky*ky);
	    ik=nmt_bins_flat_search_fast(bin,kmod,ik);
	    if(ik>=0) {
#pragma omp atomic
	      cls[index_cl][ik]+=(creal(alm1[index])*creal(alm2[index])+cimag(alm1[index])*cimag(alm2[index]));
#pragma omp atomic
	      n_cells[ik]++;
	    }
	  }
	} //end omp for
      } //end omp parallel

      for(il=0;il<bin->n_bands;il++) {
	if(n_cells[il]<=0)
	  cls[index_cl][il]=0;
	else
	  cls[index_cl][il]*=norm_factor/n_cells[il];
      }
    }
  }
  free(n_cells);
}

void fs_anafast(nmt_flatsky_info *fs,nmt_binning_scheme_flat *bin,
		flouble **maps_1,flouble **maps_2,int spin_1,int spin_2,flouble **cls)
{
  int i1;
  fcomplex **alms_1,**alms_2;
  int nmaps_1=1,nmaps_2=1;
  long nmodes=fs->ny*((long)(fs->nx/2+1));
  if(spin_1) nmaps_1=2;
  if(spin_2) nmaps_2=2;

  alms_1=my_malloc(nmaps_1*sizeof(fcomplex *));
  for(i1=0;i1<nmaps_1;i1++)
    alms_1[i1]=dftw_malloc(nmodes*sizeof(fcomplex));
  fs_map2alm(fs,1,spin_1,maps_1,alms_1);

  if(maps_1==maps_2)
    alms_2=alms_1;
  else {
    alms_2=my_malloc(nmaps_2*sizeof(fcomplex *));
    for(i1=0;i1<nmaps_2;i1++)
      alms_2[i1]=dftw_malloc(nmodes*sizeof(fcomplex));
    fs_map2alm(fs,1,spin_2,maps_2,alms_2);
  }

  fs_alm2cl(fs,bin,alms_1,alms_2,spin_1,spin_2,cls,1.,-1.,1.,-1.);

  for(i1=0;i1<nmaps_1;i1++)
    dftw_free(alms_1[i1]);
  free(alms_1);
  if(maps_1!=maps_2) {
    for(i1=0;i1<nmaps_2;i1++)
      dftw_free(alms_2[i1]);
    free(alms_2);
  }
}

fcomplex **fs_synalm(int nx,int ny,flouble lx,flouble ly,int nmaps,
		     nmt_k_function **cells,nmt_k_function **beam,int seed)
{
  int imap;
  fcomplex **alms;
  long nmodes=ny*((long)(nx/2+1));

  alms=my_malloc(nmaps*sizeof(fcomplex *));
  for(imap=0;imap<nmaps;imap++)
    alms[imap]=dftw_malloc(nmodes*sizeof(fcomplex));

  //Switch off error handler for Cholesky decomposition
  gsl_error_handler_t *geh=gsl_set_error_handler_off();

  int numthr=0;

#pragma omp parallel default(none)			\
  shared(nx,ny,lx,ly,nmaps,cells,beam,seed,alms,numthr)
  {
    //This is to avoid using the omp.h library
    int ithr;
#pragma omp critical
    {
      ithr=numthr;
      numthr++;
    }

    int iy;
    double dkx=2*M_PI/lx,dky=2*M_PI/ly;
    double inv_dkvol=1./(dkx*dky);
    gsl_vector *rv1=gsl_vector_alloc(nmaps);
    gsl_vector *iv1=gsl_vector_alloc(nmaps);
    gsl_vector *rv2=gsl_vector_alloc(nmaps);
    gsl_vector *iv2=gsl_vector_alloc(nmaps);
    gsl_matrix *clmat=gsl_matrix_calloc(nmaps,nmaps); 
    gsl_vector *eval =gsl_vector_alloc(nmaps);
    gsl_matrix *evec =gsl_matrix_alloc(nmaps,nmaps); 
    gsl_eigen_symmv_workspace *wsym=gsl_eigen_symmv_alloc(nmaps);
    unsigned int seed_thr=(unsigned int)(seed+ithr);
    gsl_rng *rng=init_rng(seed_thr);
    gsl_interp_accel *intacc_cells=gsl_interp_accel_alloc();
    gsl_interp_accel *intacc_beam=gsl_interp_accel_alloc();

#pragma omp for
    for(iy=0;iy<ny;iy++) {
      int ix;
      flouble ky;
      if(2*iy<=ny)
	ky=iy*dky;
      else
	ky=-(ny-iy)*dky;
      for(ix=0;ix<=nx/2;ix++) {
	int imp1,imp2;
	flouble kx=ix*dkx;
	long index=ix+((long)(nx/2+1))*iy;
	flouble kmod=sqrt(kx*kx+ky*ky);
	if(kmod<0) {
	  for(imp1=0;imp1<nmaps;imp1++)
	    alms[imp1][index]=0;
	}
	else {
	  //Get power spectrum
	  int icl=0;
	  for(imp1=0;imp1<nmaps;imp1++) {
	    for(imp2=imp1;imp2<nmaps;imp2++) {//Fill up only lower triangular part
	      flouble cl=0.5*inv_dkvol*nmt_k_function_eval(cells[icl],kmod,intacc_cells);
	      gsl_matrix_set(clmat,imp1,imp2,cl);
	      if(imp2!=imp1)
		gsl_matrix_set(clmat,imp2,imp1,cl);
	      icl++;
	    }
	  }

	  //Take square root
	  gsl_eigen_symmv(clmat,eval,evec,wsym);
	  for(imp1=0;imp1<nmaps;imp1++) {
	    double dr,di; //At the same time get white random numbers
	    rng_gauss(rng,&dr,&di);
	    gsl_vector_set(rv1,imp1,dr);
	    gsl_vector_set(iv1,imp1,di);
	    for(imp2=0;imp2<nmaps;imp2++) {
	      double oij=gsl_matrix_get(evec,imp1,imp2);
	      double lambda=gsl_vector_get(eval,imp2);
	      if(lambda<=0) lambda=0;
	      else lambda=sqrt(lambda);
	      gsl_matrix_set(clmat,imp1,imp2,oij*lambda);
	    }
	  }

	  //Get correlate random numbers
	  gsl_blas_dgemv(CblasNoTrans,1.,clmat,rv1,0,rv2);
	  gsl_blas_dgemv(CblasNoTrans,1.,clmat,iv1,0,iv2);
	  for(imp1=0;imp1<nmaps;imp1++) {
	    flouble bm=nmt_k_function_eval(beam[imp1],kmod,intacc_beam);
	    flouble a_re=bm*gsl_vector_get(rv2,imp1);
	    flouble a_im=bm*gsl_vector_get(iv2,imp1);
	    if(ix==0) {
	      if(iy>ny/2)
		continue;
	      else {
		if(iy==0)
		  alms[imp1][index]=(fcomplex)(M_SQRT2*a_re+I*0*a_im);
		else {
		  int iyy=ny-iy;
		  alms[imp1][index]=(fcomplex)(a_re+I*a_im);
		  alms[imp1][ix+((long)(nx/2+1))*iyy]=(fcomplex)(a_re-I*a_im);
		}
	      }
	    }
	    else
	      alms[imp1][index]=(fcomplex)(a_re+I*a_im);
	  }
	}
      }
    } //omp end for
    gsl_vector_free(rv1);
    gsl_vector_free(iv1);
    gsl_vector_free(rv2);
    gsl_vector_free(iv2);
    gsl_matrix_free(clmat);
    gsl_vector_free(eval);
    gsl_matrix_free(evec);
    gsl_eigen_symmv_free(wsym);
    end_rng(rng);
    gsl_interp_accel_free(intacc_cells);
    gsl_interp_accel_free(intacc_beam);
  } //omp end parallel

  //Restore error handler
  gsl_set_error_handler(geh);

  return alms;
}

static void read_key(fitsfile *fptr,int dtype,char *key,void *val,int *status)
{
  fits_read_key(fptr,dtype,key,val,NULL,status);
  if(*status)
    report_error(NMT_ERROR_READ,"Key %s not found\n",key);
}

flouble *fs_read_flat_map(char *fname,int *nx,int *ny,flouble *lx,flouble *ly,int nfield)
{
  fitsfile *fptr;
  int numhdu,hdutype,naxis,naxis1,naxis2;
  double cdelt1,cdelt2;
  flouble nulval=-999;
  int status=0;

  fits_open_file(&fptr,fname,READONLY,&status);
  if(status)
    report_error(NMT_ERROR_FOPEN,"Can't open file %s\n",fname);
  fits_get_num_hdus(fptr,&numhdu,&status);
  if(nfield>=numhdu)
    report_error(NMT_ERROR_READ,"%d-th field doesn't exist\n",nfield);
  fits_movabs_hdu(fptr,nfield+1,&hdutype,&status);
  if(hdutype!=IMAGE_HDU)
    report_error(NMT_ERROR_READ,"Requested HDU is not an image\n");

  //Read patch properties
  read_key(fptr,TINT,"NAXIS",&naxis,&status);
  read_key(fptr,TINT,"NAXIS1",&naxis1,&status);
  read_key(fptr,TINT,"NAXIS2",&naxis2,&status);
  read_key(fptr,TDOUBLE,"CDELT1",&cdelt1,&status);
  read_key(fptr,TDOUBLE,"CDELT2",&cdelt2,&status);
  if(naxis!=2)
    report_error(NMT_ERROR_READ,"Can't find a two-dimensional map\n");
  *nx=naxis1;
  *ny=naxis2;
  *lx=fabs(naxis1*cdelt1)*M_PI/180;
  *ly=fabs(naxis2*cdelt2)*M_PI/180;

  //Read data
  long fpixel[2]={1,1}; 
  flouble *map_out=my_malloc(naxis1*naxis2*sizeof(double));
 
#ifdef _SPREC
  fits_read_pix(fptr,TFLOAT,fpixel,naxis1*naxis2,&nulval,map_out,NULL,&status);
#else //_SPREC
  fits_read_pix(fptr,TDOUBLE,fpixel,naxis1*naxis2,&nulval,map_out,NULL,&status);
#endif //_SPREC
  if(status)
    report_error(NMT_ERROR_READ,"Error reading image from file %s\n",fname);
  
  fits_close_file(fptr,&status);

  return map_out;
}
