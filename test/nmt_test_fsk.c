#include "namaster.h"
#include "ctest.h"
#include "utils.h"
#include "nmt_test_utils.h"
#include <chealpix.h>

CTEST(nmt,fsk_mapcpy) {
  int ii;
  nmt_flatsky_info *fsk=nmt_flatsky_info_alloc(100,100,M_PI/180,M_PI/180);
  double *mp1=my_malloc(fsk->npix*sizeof(double));
  double *mp2=my_malloc(fsk->npix*sizeof(double));

  for(ii=0;ii<fsk->npix;ii++) {
    mp1[ii]=2.;
    mp2[ii]=0.5;
  }

  fs_mapcpy(fsk,mp1,mp2);
  for(ii=0;ii<fsk->npix;ii++)
    ASSERT_DBL_NEAR_TOL(mp1[ii],mp2[ii],1E-10);
  
  free(mp1);
  free(mp2);
  nmt_flatsky_info_free(fsk);
}
  
CTEST(nmt,fsk_read_bad) {
  int nx,ny;
  flouble lx,ly;
  flouble *map;

  set_error_policy(THROW_ON_ERROR);

  //Test non-existent file
  try{ map=fs_read_flat_map("none.fits",&nx,&ny,&lx,&ly,0); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);

  //Test incorrect file format
  try{ map=fs_read_flat_map("test/benchmarks/msk.fits",&nx,&ny,&lx,&ly,1); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);
  try{ map=fs_read_flat_map("test/benchmarks/msk.fits",&nx,&ny,&lx,&ly,0); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);

  //Test inexistent field
  try{ map=fs_read_flat_map("test/benchmarks/msk_flat.fits",&nx,&ny,&lx,&ly,1); }
  ASSERT_NOT_EQUAL(0,nmt_exception_status);

  set_error_policy(EXIT_ON_ERROR);
}

CTEST(nmt,fsk_read_good) {
  int nx,ny;
  flouble lx,ly;
  flouble *map;

  //Test successful read
  map=fs_read_flat_map("test/benchmarks/msk_flat.fits",&nx,&ny,&lx,&ly,0);
  ASSERT_EQUAL(NX_TEST,nx);
  ASSERT_EQUAL(NY_TEST,ny);
  ASSERT_DBL_NEAR_TOL(DX_TEST*NX_TEST*M_PI/180.,lx,1E-10);
  ASSERT_DBL_NEAR_TOL(DY_TEST*NY_TEST*M_PI/180.,ly,1E-10);
  ASSERT_NOT_NULL(map);
  ASSERT_DBL_NEAR_TOL(0.,map[0],1E-10);
  ASSERT_DBL_NEAR_TOL(6.064284705880828E-01,map[50  +nx*20    ],1E-10);
  ASSERT_DBL_NEAR_TOL(9.999850684720466E-01,map[nx/2+nx*(ny/2)],1E-10);
  free(map);
}

CTEST(nmt,fsk_synalm) {
  int ii;
  int nbpw=30;
  int nmaps=2;
  int ncls=nmaps*nmaps;
  nmt_flatsky_info *fsk=nmt_flatsky_info_alloc(141,311,M_PI/180,M_PI/180);
  double lmax_x=fsk->nx*M_PI/fsk->lx;
  double lmax_y=fsk->ny*M_PI/fsk->ly;
  double lmax=sqrt(lmax_x*lmax_x+lmax_y*lmax_y);
  double lpivot=lmax/6.;
  double alpha_pivot=1.;
  double *larr=my_malloc((nbpw+1)*sizeof(double));
  double **cells=my_malloc(ncls*sizeof(double *));
  long *npixls=my_calloc(nbpw,sizeof(long));
  
  for(ii=0;ii<ncls;ii++)
    cells[ii]=my_calloc(nbpw+1,sizeof(double));
  
  for(ii=0;ii<=nbpw;ii++) {
    double ll=ii*lmax/nbpw;
    larr[ii]=ll;
    cells[0][ii]=pow((2*lpivot)/(ll+lpivot),alpha_pivot);
    cells[3][ii]=pow((2*lpivot)/(ll+lpivot),alpha_pivot);
  }

  nmt_binning_scheme_flat *bpw=nmt_bins_flat_create(nbpw,larr,&(larr[1]));
  double dkx=2*M_PI/fsk->lx,dky=2*M_PI/fsk->ly;
  for(ii=0;ii<fsk->ny;ii++) {
    int jj;
    double ky;
    int ibin=0;
    if(2*ii<=fsk->ny) ky=ii*dky;
    else ky=-(fsk->ny-ii)*dky;
    for(jj=0;jj<fsk->nx;jj++) {
      double kx,kmod;
      if(2*jj<=fsk->nx) kx=jj*dkx;
      else kx=-(fsk->nx-jj)*dkx;
      kmod=sqrt(kx*kx+ky*ky);
      ibin=nmt_bins_flat_search_fast(bpw,kmod,ibin);//(int)(kmod/lmax*nbpw);
      if((ibin>=0) && (ibin<nbpw))
	npixls[ibin]++;
    }
  }
  
  nmt_k_function **clf=my_malloc(ncls*sizeof(nmt_k_function *));
  nmt_k_function **clf_pass=my_malloc(3*sizeof(nmt_k_function *));
  nmt_k_function **bmf=my_malloc(nmaps*sizeof(nmt_k_function *));
  clf[0]=nmt_k_function_alloc(nbpw+1,larr,cells[0],1.,0.,0);
  clf[1]=nmt_k_function_alloc(nbpw+1,larr,cells[1],0.,0.,0);
  clf[2]=nmt_k_function_alloc(nbpw+1,larr,cells[2],0.,0.,0);
  clf[3]=nmt_k_function_alloc(nbpw+1,larr,cells[3],1.,0.,0);
  clf_pass[0]=clf[0];
  clf_pass[1]=clf[1];
  clf_pass[2]=clf[3];
  for(ii=0;ii<nmaps;ii++)
    bmf[ii]=nmt_k_function_alloc(nbpw+1,larr,NULL,1.,1.,1);

  fcomplex **alms=fs_synalm(fsk->nx,fsk->ny,fsk->lx,fsk->ly,nmaps,clf_pass,bmf,1234);

  fs_alm2cl(fsk,bpw,alms,alms,1,1,cells,1.,-1.,1.,-1.);

  for(ii=0;ii<nbpw;ii++) {
    int im1;
    double l=0.5*(larr[ii]+larr[ii+1]);
    for(im1=0;im1<nmaps;im1++) {
      int im2;
      for(im2=0;im2<nmaps;im2++) {
	double c11=nmt_k_function_eval(clf[im1+nmaps*im1],l,NULL);
	double c12=nmt_k_function_eval(clf[im2+nmaps*im1],l,NULL);
	double c21=nmt_k_function_eval(clf[im1+nmaps*im2],l,NULL);
	double c22=nmt_k_function_eval(clf[im2+nmaps*im2],l,NULL);
	double sig=sqrt((c11*c22+c12*c21)/npixls[ii]);
	double diff=fabs(cells[im2+nmaps*im1][ii]-c12);
	//Check that there are no >5-sigma fluctuations around input power spectrum
	ASSERT_TRUE((int)(diff<5*sig));
      }
    }
  }
  for(ii=0;ii<nmaps;ii++)
    dftw_free(alms[ii]);
  free(alms);
  
  for(ii=0;ii<ncls;ii++)
    free(cells[ii]);
  for(ii=0;ii<ncls;ii++)
    nmt_k_function_free(clf[ii]);
  for(ii=0;ii<nmaps;ii++)
    nmt_k_function_free(bmf[ii]);

  nmt_bins_flat_free(bpw);
  nmt_flatsky_info_free(fsk);
  free(npixls);
  free(clf);
  free(clf_pass);
  free(bmf);
  free(cells);
  free(larr);
}

CTEST(nmt,fsk_cls) {
  int ii;
  int nmaps=34;
  int nbpw=10;
  nmt_flatsky_info *fsk=nmt_flatsky_info_alloc(141,311,M_PI/180,M_PI/180);
  double **cells=my_malloc(17*sizeof(double *));
  
  for(ii=0;ii<17;ii++)
    cells[ii]=my_calloc(nbpw,sizeof(double));

  //Analytic example (same as in fsk_fft)
  int i0_x=2,i0_y=3;
  double k0_x=i0_x*2*M_PI/fsk->lx;
  double k0_y=i0_y*2*M_PI/fsk->ly;
  double **maps0=test_make_map_analytic_flat(fsk,0,i0_x,i0_y);
  double **maps1=test_make_map_analytic_flat(fsk,1,i0_x,i0_y);
  double **maps2=test_make_map_analytic_flat(fsk,2,i0_x,i0_y);
  fcomplex **alms0=my_malloc(1*sizeof(fcomplex *));
  fcomplex **alms1=my_malloc(2*sizeof(fcomplex *));
  fcomplex **alms2=my_malloc(2*sizeof(fcomplex *));
  alms0[0]=dftw_malloc(fsk->ny*(fsk->nx/2+1)*sizeof(fcomplex));
  alms1[0]=dftw_malloc(fsk->ny*(fsk->nx/2+1)*sizeof(fcomplex));
  alms1[1]=dftw_malloc(fsk->ny*(fsk->nx/2+1)*sizeof(fcomplex));
  alms2[0]=dftw_malloc(fsk->ny*(fsk->nx/2+1)*sizeof(fcomplex));
  alms2[1]=dftw_malloc(fsk->ny*(fsk->nx/2+1)*sizeof(fcomplex));
  fs_map2alm(fsk,1,0,maps0,alms0);
  fs_map2alm(fsk,1,1,maps1,alms1);
  fs_map2alm(fsk,1,2,maps2,alms2);

  //Bandpowers
  double lmax=fmax(fsk->nx*M_PI/fsk->lx,fsk->ny*M_PI/fsk->ly);
  double *l0=my_malloc(nbpw*sizeof(double));
  double *lf=my_malloc(nbpw*sizeof(double));
  long *npixls=my_calloc(nbpw,sizeof(long));
  double dkx=2*M_PI/fsk->lx,dky=2*M_PI/fsk->ly;
  int ibpw0=(int)(sqrt(k0_x*k0_x+k0_y*k0_y)*nbpw/lmax);
  for(ii=0;ii<nbpw;ii++) {
    l0[ii]=ii*lmax/nbpw;
    lf[ii]=(ii+1.)*lmax/nbpw;
  }
  nmt_binning_scheme_flat *bpw=nmt_bins_flat_create(nbpw,l0,lf);
  for(ii=0;ii<fsk->ny;ii++) {
    int jj;
    double ky;
    int ibin=0;
    if(2*ii<=fsk->ny) ky=ii*dky;
    else ky=-(fsk->ny-ii)*dky;
    for(jj=0;jj<fsk->nx;jj++) {
      double kx,kmod;
      if(2*jj<=fsk->nx) kx=jj*dkx;
      else kx=-(fsk->nx-jj)*dkx;
      kmod=sqrt(kx*kx+ky*ky);
      ibin=nmt_bins_flat_search_fast(bpw,kmod,ibin);//(int)(kmod/lmax*nbpw);
      if((ibin>=0) && (ibin<nbpw))
	npixls[ibin]++;
    }
  }

  //Theory prediction <|a|^2>*(2*pi/Lx)*(2*pi/Ly) [<|a|^2>=0.5]
  double predict=0.5*2*M_PI*2*M_PI/(fsk->lx*fsk->ly*npixls[ibpw0]);

  //Compute power spectra and compare with prediction
  fs_alm2cl(fsk,bpw,alms0,alms0,0,0,&(cells[0]),1.,-1.,1.,-1.);
  fs_alm2cl(fsk,bpw,alms0,alms1,0,1,&(cells[1]),1.,-1.,1.,-1.);
  fs_alm2cl(fsk,bpw,alms0,alms2,0,2,&(cells[3]),1.,-1.,1.,-1.);
  fs_alm2cl(fsk,bpw,alms1,alms1,1,1,&(cells[5]),1.,-1.,1.,-1.);
  fs_alm2cl(fsk,bpw,alms1,alms2,1,2,&(cells[9]),1.,-1.,1.,-1.);
  fs_alm2cl(fsk,bpw,alms2,alms2,2,2,&(cells[13]),1.,-1.,1.,-1.);
  for(ii=0;ii<nbpw;ii++) {
    int jj;
    double pred=0;
    if(ii==ibpw0)
      pred=predict;
    ASSERT_DBL_NEAR_TOL(pred,cells[0][ii],1E-5);
    ASSERT_DBL_NEAR_TOL(pred,cells[1][ii],1E-5);
    ASSERT_DBL_NEAR_TOL(pred,cells[3][ii],1E-5);
    ASSERT_DBL_NEAR_TOL(pred,cells[5][ii],1E-5);
    ASSERT_DBL_NEAR_TOL(pred,cells[9][ii],1E-5);
    ASSERT_DBL_NEAR_TOL(pred,cells[13][ii],1E-5);
    ASSERT_DBL_NEAR_TOL(0.,cells[2][ii],1E-5);
    ASSERT_DBL_NEAR_TOL(0.,cells[4][ii],1E-5);
    ASSERT_DBL_NEAR_TOL(0.,cells[6][ii],1E-5);
    ASSERT_DBL_NEAR_TOL(0.,cells[7][ii],1E-5);
    ASSERT_DBL_NEAR_TOL(0.,cells[8][ii],1E-5);
    ASSERT_DBL_NEAR_TOL(0.,cells[10][ii],1E-5);
    ASSERT_DBL_NEAR_TOL(0.,cells[11][ii],1E-5);
    ASSERT_DBL_NEAR_TOL(0.,cells[12][ii],1E-5);
    ASSERT_DBL_NEAR_TOL(0.,cells[14][ii],1E-5);
    ASSERT_DBL_NEAR_TOL(0.,cells[15][ii],1E-5);
    ASSERT_DBL_NEAR_TOL(0.,cells[16][ii],1E-5);
  }

  fs_anafast(fsk,bpw,maps0,maps0,0,0,&(cells[0]));
  fs_anafast(fsk,bpw,maps0,maps1,0,1,&(cells[1]));
  fs_anafast(fsk,bpw,maps0,maps2,0,2,&(cells[3]));
  fs_anafast(fsk,bpw,maps1,maps1,1,1,&(cells[5]));
  fs_anafast(fsk,bpw,maps1,maps2,1,2,&(cells[9]));
  fs_anafast(fsk,bpw,maps2,maps2,2,2,&(cells[13]));
  for(ii=0;ii<nbpw;ii++) {
    int jj;
    double pred=0;
    if(ii==ibpw0)
      pred=predict;
    ASSERT_DBL_NEAR_TOL(pred,cells[0][ii],1E-5);
    ASSERT_DBL_NEAR_TOL(pred,cells[1][ii],1E-5);
    ASSERT_DBL_NEAR_TOL(pred,cells[3][ii],1E-5);
    ASSERT_DBL_NEAR_TOL(pred,cells[5][ii],1E-5);
    ASSERT_DBL_NEAR_TOL(pred,cells[9][ii],1E-5);
    ASSERT_DBL_NEAR_TOL(pred,cells[13][ii],1E-5);
    ASSERT_DBL_NEAR_TOL(0.,cells[2][ii],1E-5);
    ASSERT_DBL_NEAR_TOL(0.,cells[4][ii],1E-5);
    ASSERT_DBL_NEAR_TOL(0.,cells[6][ii],1E-5);
    ASSERT_DBL_NEAR_TOL(0.,cells[7][ii],1E-5);
    ASSERT_DBL_NEAR_TOL(0.,cells[8][ii],1E-5);
    ASSERT_DBL_NEAR_TOL(0.,cells[10][ii],1E-5);
    ASSERT_DBL_NEAR_TOL(0.,cells[11][ii],1E-5);
    ASSERT_DBL_NEAR_TOL(0.,cells[12][ii],1E-5);
    ASSERT_DBL_NEAR_TOL(0.,cells[14][ii],1E-5);
    ASSERT_DBL_NEAR_TOL(0.,cells[15][ii],1E-5);
    ASSERT_DBL_NEAR_TOL(0.,cells[16][ii],1E-5);
  }

  free(l0); free(lf); free(npixls);
  nmt_bins_flat_free(bpw);
  for(ii=0;ii<17;ii++)
    free(cells[ii]);
  free(cells);
  dftw_free(maps0[0]);
  dftw_free(maps1[0]);
  dftw_free(maps1[1]);
  dftw_free(maps2[0]);
  dftw_free(maps2[1]);
  dftw_free(alms0[0]);
  dftw_free(alms1[0]);
  dftw_free(alms1[1]);
  dftw_free(alms2[0]);
  dftw_free(alms2[1]);
  free(maps0);
  free(maps1);
  free(maps2);
  free(alms0);
  free(alms1);
  free(alms2);
  nmt_flatsky_info_free(fsk);
}

CTEST(nmt,fsk_fft) {
  int ii;
  int nmaps=34;
  nmt_flatsky_info *fsk=nmt_flatsky_info_alloc(141,311,M_PI/180,M_PI/180);
  double **maps=my_malloc(2*nmaps*sizeof(double *));
  fcomplex **alms=my_malloc(2*nmaps*sizeof(fcomplex *));

  for(ii=0;ii<2*nmaps;ii++) {
    maps[ii]=dftw_malloc(fsk->npix*sizeof(double));
    alms[ii]=dftw_malloc(fsk->ny*(fsk->nx/2+1)*sizeof(fcomplex));
  }

  //Direct FFT
  //Single FFT, spin-0
  fs_map2alm(fsk,1,0,maps,alms);
  //Several FFT, spin-0
  fs_map2alm(fsk,nmaps,0,maps,alms);
  //Single FFT, spin-2
  fs_map2alm(fsk,1,2,maps,alms);
  //Several FFT, spin-2
  fs_map2alm(fsk,nmaps,2,maps,alms);

  //Zero_alm and alter_alm
  fs_zero_alm(fsk,alms[0]);
  fs_zero_alm(fsk,alms[1]);
  nmt_k_function *b=fs_generate_beam_window(10.);
  fs_alter_alm(fsk,10.,alms[0],alms[1],NULL,0);
  fs_alter_alm(fsk,10.,alms[0],alms[1],b,0);
  fs_alter_alm(fsk,10.,alms[0],alms[1],b,1);
  nmt_k_function_free(b);
  
  //Inverse FFT
  //Single FFT, spin-0
  fs_alm2map(fsk,1,0,maps,alms);
  //Several FFT, spin-0
  fs_alm2map(fsk,nmaps,0,maps,alms);
  //Single FFT, spin-2
  fs_alm2map(fsk,1,2,maps,alms);
  //Several FFT, spin-2
  fs_alm2map(fsk,nmaps,2,maps,alms);

  for(ii=0;ii<2*nmaps;ii++)
    dftw_free(maps[ii]);
  free(maps);
  
  //Particular example
  //Spin-0. map = 2*pi/A * Re[exp(i*k0*x)] ->
  //        a(k) = (delta_{k,k0}+delta_{k,-k0})/2
  int i0_x=2,i0_y=3;
  maps=test_make_map_analytic_flat(fsk,0,i0_x,i0_y);
  fs_map2alm(fsk,1,0,maps,alms);
  for(ii=0;ii<fsk->ny;ii++) {
    int jj;
    for(jj=0;jj<=fsk->nx/2;jj++) {
      double re=creal(alms[0][jj+(fsk->nx/2+1)*ii]);
      double im=cimag(alms[0][jj+(fsk->nx/2+1)*ii]);
      if((jj==i0_x) && (ii==i0_y)) {
	ASSERT_DBL_NEAR_TOL(0.5,re,1E-5);
	ASSERT_DBL_NEAR_TOL(0.0,im,1E-5);
      }
      else {
	ASSERT_DBL_NEAR_TOL(0.0,re,1E-5);
	ASSERT_DBL_NEAR_TOL(0.0,im,1E-5);
      }
    }
  }
  dftw_free(maps[0]); free(maps);
  //Spin-1. map = 2*pi/A * (-cos(phi_k0),sin(phi_k0)) Im[exp(i*k0*x)] ->
  //        a_E(k) = (delta_{k,k0}+delta_{k,-k0})/2
  //        a_B(k) = 0
  maps=test_make_map_analytic_flat(fsk,1,i0_x,i0_y);
  fs_map2alm(fsk,1,1,maps,alms);
  for(ii=0;ii<fsk->ny;ii++) {
    int jj;
    for(jj=0;jj<=fsk->nx/2;jj++) {
      double re0=creal(alms[0][jj+(fsk->nx/2+1)*ii]);
      double im0=cimag(alms[0][jj+(fsk->nx/2+1)*ii]);
      double re1=creal(alms[1][jj+(fsk->nx/2+1)*ii]);
      double im1=cimag(alms[1][jj+(fsk->nx/2+1)*ii]);
      if((jj==i0_x) && (ii==i0_y)) {
	ASSERT_DBL_NEAR_TOL(0.5,re0,1E-5);
	ASSERT_DBL_NEAR_TOL(0.0,im0,1E-5);
	ASSERT_DBL_NEAR_TOL(0.0,re1,1E-5);
	ASSERT_DBL_NEAR_TOL(0.0,im1,1E-5);
      }
      else {
	ASSERT_DBL_NEAR_TOL(0.0,re0,1E-5);
	ASSERT_DBL_NEAR_TOL(0.0,im0,1E-5);
	ASSERT_DBL_NEAR_TOL(0.0,re1,1E-5);
	ASSERT_DBL_NEAR_TOL(0.0,im1,1E-5);
      }
    }
  }
  dftw_free(maps[0]); dftw_free(maps[1]); free(maps);
  //Spin-2. map = 2*pi/A * (cos(2*phi_k0),-sin(2*phi_k0)) Re[exp(i*k0*x)] ->
  //        a_E(k) = (delta_{k,k0}+delta_{k,-k0})/2
  //        a_B(k) = 0
  maps=test_make_map_analytic_flat(fsk,2,i0_x,i0_y);
  fs_map2alm(fsk,1,2,maps,alms);
  for(ii=0;ii<fsk->ny;ii++) {
    int jj;
    for(jj=0;jj<=fsk->nx/2;jj++) {
      double re0=creal(alms[0][jj+(fsk->nx/2+1)*ii]);
      double im0=cimag(alms[0][jj+(fsk->nx/2+1)*ii]);
      double re1=creal(alms[1][jj+(fsk->nx/2+1)*ii]);
      double im1=cimag(alms[1][jj+(fsk->nx/2+1)*ii]);
      if((jj==i0_x) && (ii==i0_y)) {
	ASSERT_DBL_NEAR_TOL(0.5,re0,1E-5);
	ASSERT_DBL_NEAR_TOL(0.0,im0,1E-5);
	ASSERT_DBL_NEAR_TOL(0.0,re1,1E-5);
	ASSERT_DBL_NEAR_TOL(0.0,im1,1E-5);
      }
      else {
	ASSERT_DBL_NEAR_TOL(0.0,re0,1E-5);
	ASSERT_DBL_NEAR_TOL(0.0,im0,1E-5);
	ASSERT_DBL_NEAR_TOL(0.0,re1,1E-5);
	ASSERT_DBL_NEAR_TOL(0.0,im1,1E-5);
      }
    }
  }
  dftw_free(maps[0]); dftw_free(maps[1]); free(maps);

  for(ii=0;ii<2*nmaps;ii++)
    dftw_free(alms[ii]);
  free(alms);
  nmt_flatsky_info_free(fsk);
}

CTEST(nmt,fsk_fft_malloc) {
  set_error_policy(THROW_ON_ERROR);

  double *dum=dftw_malloc(10);
  free(dum);
  try{ dftw_malloc(-1); }
  catch(1) {}
  ASSERT_NOT_EQUAL(0,nmt_exception_status);

  set_error_policy(EXIT_ON_ERROR);
}

CTEST(nmt,fsk_info) {
  nmt_flatsky_info *fsk=nmt_flatsky_info_alloc(100,100,M_PI/180,M_PI/180);
  ASSERT_EQUAL(10000,fsk->npix);
  ASSERT_EQUAL(pow(M_PI/180,2)/10000,fsk->pixsize);
  nmt_flatsky_info_free(fsk);
}
  
CTEST(nmt,fsk_algb) {
  int ii;
  nmt_flatsky_info *fsk=nmt_flatsky_info_alloc(100,100,M_PI/180,M_PI/180);
  double *mp1=my_malloc(fsk->npix*sizeof(double));
  double *mp2=my_malloc(fsk->npix*sizeof(double));
  double *mpr=my_malloc(fsk->npix*sizeof(double));

  for(ii=0;ii<fsk->npix;ii++) {
    mp1[ii]=2.;
    mp2[ii]=0.5;
  }

  double d=fs_map_dot(fsk,mp1,mp2);
  fs_map_product(fsk,mp1,mp2,mpr);
  fs_map_product(fsk,mp1,mp2,mp2);
  for(ii=0;ii<fsk->npix;ii++) {
    ASSERT_DBL_NEAR_TOL(1.,mpr[ii],1E-10);
    ASSERT_DBL_NEAR_TOL(1.,mp2[ii],1E-10);
  }
  ASSERT_DBL_NEAR_TOL(pow(M_PI/180,2),d,1E-5);
  
  free(mp1);
  free(mp2);
  free(mpr);
  nmt_flatsky_info_free(fsk);
}


static double fk(double k)
{
  return 100./(k+100.);
}

CTEST(nmt,fsk_func) {
  int l;
  long lmax=2000;
  double *karr=my_malloc((lmax+1)*sizeof(double));
  double *farr=my_malloc((lmax+1)*sizeof(double));

  for(l=0;l<=lmax;l++) {
    karr[l]=l;
    farr[l]=fk(karr[l]);
  }

  nmt_k_function *kf=nmt_k_function_alloc(lmax+1,karr,farr,1.,0.,0);

  for(l=0;l<lmax;l++) {
    double k=l+0.5;
    double f_int=nmt_k_function_eval(kf,k,NULL);
    double f_exc=fk(k);
    ASSERT_DBL_NEAR_TOL(1.,f_int/f_exc,1E-3);
  }
  
  nmt_k_function_free(kf);

  //Beams
  double sigma=1.*M_PI/180; //Beam sigma in radians
  double fwhm_amin=sigma*180*60/M_PI*2.35482;
  kf=fs_generate_beam_window(fwhm_amin);
  for(l=0;l<100;l++) {
    double ll=(l+0.5)*4.8/(100.*sigma);
    double b=nmt_k_function_eval(kf,ll,NULL);
    double bt=exp(-0.5*ll*ll*sigma*sigma);
    ASSERT_DBL_NEAR_TOL(1.,b/bt,1E-3);
  }
  nmt_k_function_free(kf);
  
  free(karr);
  free(farr);
}
