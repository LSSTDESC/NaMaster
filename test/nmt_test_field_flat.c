#include "namaster.h"
#include "ctest.h"
#include "utils.h"
#include "nmt_test_utils.h"

CTEST(nmt,field_flat_empty) {
  nmt_field_flat *f;
  int ii,nmaps;
  double ntemp=5;
  nmt_flatsky_info *fsk=nmt_flatsky_info_alloc(141,311,M_PI/180,M_PI/180);
  long npix=fsk->npix;
  int nbpw=30;
  double *beam=my_malloc((nbpw+1)*sizeof(double));
  double *larr=my_malloc((nbpw+1)*sizeof(double));
  double *mask=my_malloc(npix*sizeof(double));
  
  for(ii=0;ii<npix;ii++)
    mask[ii]=1.;
  
  for(ii=0;ii<=nbpw;ii++) {
    larr[ii]=ii;
    beam[ii]=1.;
  }

  ////////
  //Spin-2
  //With purification
  f=nmt_field_flat_alloc(fsk->nx,fsk->ny,fsk->lx,fsk->ly,mask,2,NULL,0,NULL,
			 nbpw+1,larr,beam,1,1,1E-5,0,1,1);
  //Sanity checks
  ASSERT_EQUAL(2,f->spin);
  ASSERT_EQUAL(2,f->nmaps);
  nmt_field_flat_free(f);
  
  nmt_flatsky_info_free(fsk);
  free(beam);
  free(larr);
  free(mask);
}

CTEST(nmt,field_flat_lite) {
  nmt_field_flat *f;
  int ii,nmaps;
  double ntemp=5;
  nmt_flatsky_info *fsk=nmt_flatsky_info_alloc(141,311,M_PI/180,M_PI/180);
  long npix=fsk->npix;
  double **maps;
  double ***temp=my_malloc(ntemp*sizeof(double **));
  int nbpw=30;
  double *beam=my_malloc((nbpw+1)*sizeof(double));
  double *larr=my_malloc((nbpw+1)*sizeof(double));
  double *mask=my_malloc(npix*sizeof(double));
  int i0_x=2,i0_y=3;
  
  for(ii=0;ii<npix;ii++)
    mask[ii]=1.;
  
  for(ii=0;ii<=nbpw;ii++) {
    larr[ii]=ii;
    beam[ii]=1.;
  }

  ////////
  //Spin-2
  nmaps=2;
  //Create inputs
  maps=test_make_map_analytic_flat(fsk,2,i0_x,i0_y);
  for(ii=0;ii<ntemp;ii++)
    temp[ii]=test_make_map_analytic_flat(fsk,2,i0_x,i0_y);

  //With purification (nothing should change)
  f=nmt_field_flat_alloc(fsk->nx,fsk->ny,fsk->lx,fsk->ly,mask,2,maps,0,NULL,
			 nbpw+1,larr,beam,1,1,1E-5,0,1,0);
  //Sanity checks
  ASSERT_EQUAL(2,f->spin);
  ASSERT_EQUAL(2,f->nmaps);
  //Harmonic transform
  for(ii=0;ii<fsk->ny;ii++) {
    int jj;
    for(jj=0;jj<=fsk->nx/2;jj++) {
      double re0=creal(f->alms[0][jj+(fsk->nx/2+1)*ii]);
      double im0=cimag(f->alms[0][jj+(fsk->nx/2+1)*ii]);
      double re1=creal(f->alms[1][jj+(fsk->nx/2+1)*ii]);
      double im1=cimag(f->alms[1][jj+(fsk->nx/2+1)*ii]);
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
  nmt_field_flat_free(f);

  //With templates
  f=nmt_field_flat_alloc(fsk->nx,fsk->ny,fsk->lx,fsk->ly,mask,2,maps,ntemp,temp,
			 0,NULL,NULL,0,0,1E-5,0,1,0);
  //Since maps and templates are the same, template-deprojected alms should be 0
  for(ii=0;ii<fsk->ny;ii++) {
    int jj;
    for(jj=0;jj<=fsk->nx/2;jj++) {
      double re0=creal(f->alms[0][jj+(fsk->nx/2+1)*ii]);
      double im0=cimag(f->alms[0][jj+(fsk->nx/2+1)*ii]);
      double re1=creal(f->alms[1][jj+(fsk->nx/2+1)*ii]);
      double im1=cimag(f->alms[1][jj+(fsk->nx/2+1)*ii]);
      ASSERT_DBL_NEAR_TOL(0.0,re0,1E-5);
      ASSERT_DBL_NEAR_TOL(0.0,im0,1E-5);
      ASSERT_DBL_NEAR_TOL(0.0,re1,1E-5);
      ASSERT_DBL_NEAR_TOL(0.0,im1,1E-5);
    }
  }
  nmt_field_flat_free(f);

  //Free inputs
  for(ii=0;ii<ntemp;ii++) {
    int jj;
    for(jj=0;jj<nmaps;jj++)
      dftw_free(temp[ii][jj]);
    free(temp[ii]);
  }
  for(ii=0;ii<nmaps;ii++)
    dftw_free(maps[ii]);
  free(maps);
  
  nmt_flatsky_info_free(fsk);
  free(temp);
  free(beam);
  free(larr);
  free(mask);
}

CTEST(nmt,field_flat_alloc) {
  nmt_field_flat *f;
  int ii,nmaps;
  double ntemp=5;
  nmt_flatsky_info *fsk=nmt_flatsky_info_alloc(141,311,M_PI/180,M_PI/180);
  long npix=fsk->npix;
  double **maps;
  double ***temp=my_malloc(ntemp*sizeof(double **));
  int nbpw=30;
  double *beam=my_malloc((nbpw+1)*sizeof(double));
  double *larr=my_malloc((nbpw+1)*sizeof(double));
  double *mask=my_malloc(npix*sizeof(double));
  
  for(ii=0;ii<npix;ii++)
    mask[ii]=1.;
  
  for(ii=0;ii<=nbpw;ii++) {
    larr[ii]=ii;
    beam[ii]=1.;
  }

  ////////
  //Spin-0
  nmaps=1;
  //Create inputs
  int i0_x=2,i0_y=3;
  maps=test_make_map_analytic_flat(fsk,0,i0_x,i0_y);
  for(ii=0;ii<ntemp;ii++)
    temp[ii]=test_make_map_analytic_flat(fsk,0,i0_x,i0_y);

  //No templates
  f=nmt_field_flat_alloc(fsk->nx,fsk->ny,fsk->lx,fsk->ly,mask,0,maps,0,NULL,
			 nbpw+1,larr,beam,0,0,1E-5,0,0,0);
  //Sanity checks
  ASSERT_EQUAL(fsk->nx,f->fs->nx);
  ASSERT_EQUAL(fsk->npix,f->npix);
  ASSERT_EQUAL(0,f->pure_e);
  ASSERT_EQUAL(0,f->pure_b);
  ASSERT_EQUAL(0,f->spin);
  ASSERT_EQUAL(1,f->nmaps);
  ASSERT_EQUAL(0,f->ntemp);
  //Harmonic transform
  for(ii=0;ii<fsk->ny;ii++) {
    int jj;
    for(jj=0;jj<=fsk->nx/2;jj++) {
      double re=creal(f->alms[0][jj+(fsk->nx/2+1)*ii]);
      double im=cimag(f->alms[0][jj+(fsk->nx/2+1)*ii]);
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
  nmt_field_flat_free(f);

  //With templates
  f=nmt_field_flat_alloc(fsk->nx,fsk->ny,fsk->lx,fsk->ly,mask,0,maps,ntemp,temp,
			 0,NULL,NULL,0,0,1E-5,0,0,0);
  //Since maps and templates are the same, template-deprojected map should be 0
  for(ii=0;ii<npix;ii++)
    ASSERT_DBL_NEAR_TOL(0.0,f->maps[0][ii],1E-10);
  for(ii=0;ii<fsk->ny;ii++) {
    int jj;
    for(jj=0;jj<=fsk->nx/2;jj++) {
      int i_t;
      for(i_t=0;i_t<ntemp;i_t++) {
	double re=creal(f->a_temp[i_t][0][jj+(fsk->nx/2+1)*ii]);
	double im=cimag(f->a_temp[i_t][0][jj+(fsk->nx/2+1)*ii]);
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
  }  
  nmt_field_flat_free(f);

  //Free inputs
  for(ii=0;ii<ntemp;ii++) {
    int jj;
    for(jj=0;jj<nmaps;jj++)
      dftw_free(temp[ii][jj]);
    free(temp[ii]);
  }
  for(ii=0;ii<nmaps;ii++)
    dftw_free(maps[ii]);
  free(maps);
  ////////

  ////////
  //Spin-1
  nmaps=2;
  //Create inputs
  maps=test_make_map_analytic_flat(fsk,1,i0_x,i0_y);

  //No templates
  f=nmt_field_flat_alloc(fsk->nx,fsk->ny,fsk->lx,fsk->ly,mask,1,maps,0,NULL,
			 nbpw+1,larr,beam,0,0,1E-5,0,0,0);
  //Sanity checks
  ASSERT_EQUAL(1,f->spin);
  ASSERT_EQUAL(2,f->nmaps);
  //Harmonic transform
  for(ii=0;ii<fsk->ny;ii++) {
    int jj;
    for(jj=0;jj<=fsk->nx/2;jj++) {
      double re0=creal(f->alms[0][jj+(fsk->nx/2+1)*ii]);
      double im0=cimag(f->alms[0][jj+(fsk->nx/2+1)*ii]);
      double re1=creal(f->alms[1][jj+(fsk->nx/2+1)*ii]);
      double im1=cimag(f->alms[1][jj+(fsk->nx/2+1)*ii]);
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
  nmt_field_flat_free(f);
  for(ii=0;ii<nmaps;ii++)
    dftw_free(maps[ii]);
  free(maps);
  ////////

  ////////
  //Spin-2
  nmaps=2;
  //Create inputs
  maps=test_make_map_analytic_flat(fsk,2,i0_x,i0_y);
  for(ii=0;ii<ntemp;ii++)
    temp[ii]=test_make_map_analytic_flat(fsk,2,i0_x,i0_y);

  //No templates
  f=nmt_field_flat_alloc(fsk->nx,fsk->ny,fsk->lx,fsk->ly,mask,2,maps,0,NULL,
			 nbpw+1,larr,beam,0,0,1E-5,0,0,0);
  //Sanity checks
  ASSERT_EQUAL(2,f->spin);
  ASSERT_EQUAL(2,f->nmaps);
  //Harmonic transform
  for(ii=0;ii<fsk->ny;ii++) {
    int jj;
    for(jj=0;jj<=fsk->nx/2;jj++) {
      double re0=creal(f->alms[0][jj+(fsk->nx/2+1)*ii]);
      double im0=cimag(f->alms[0][jj+(fsk->nx/2+1)*ii]);
      double re1=creal(f->alms[1][jj+(fsk->nx/2+1)*ii]);
      double im1=cimag(f->alms[1][jj+(fsk->nx/2+1)*ii]);
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
  nmt_field_flat_free(f);

  //With purification (nothing should change)
  f=nmt_field_flat_alloc(fsk->nx,fsk->ny,fsk->lx,fsk->ly,mask,2,maps,0,NULL,
			 nbpw+1,larr,beam,1,1,1E-5,0,0,0);
  //Sanity checks
  ASSERT_EQUAL(2,f->spin);
  ASSERT_EQUAL(2,f->nmaps);
  //Harmonic transform
  for(ii=0;ii<fsk->ny;ii++) {
    int jj;
    for(jj=0;jj<=fsk->nx/2;jj++) {
      double re0=creal(f->alms[0][jj+(fsk->nx/2+1)*ii]);
      double im0=cimag(f->alms[0][jj+(fsk->nx/2+1)*ii]);
      double re1=creal(f->alms[1][jj+(fsk->nx/2+1)*ii]);
      double im1=cimag(f->alms[1][jj+(fsk->nx/2+1)*ii]);
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
  nmt_field_flat_free(f);

  //With templates
  f=nmt_field_flat_alloc(fsk->nx,fsk->ny,fsk->lx,fsk->ly,mask,2,maps,ntemp,temp,
			 0,NULL,NULL,0,0,1E-5,0,0,0);
  //Since maps and templates are the same, template-deprojected map should be 0
  for(ii=0;ii<nmaps;ii++) {
    int jj;
    for(jj=0;jj<npix;jj++)
      ASSERT_DBL_NEAR_TOL(0.0,f->maps[ii][jj],1E-10);
  }
  //Harmonic transform
  for(ii=0;ii<fsk->ny;ii++) {
    int jj;
    for(jj=0;jj<=fsk->nx/2;jj++) {
      int i_t;
      for(i_t=0;i_t<ntemp;i_t++) {
	double re0=creal(f->a_temp[i_t][0][jj+(fsk->nx/2+1)*ii]);
	double im0=cimag(f->a_temp[i_t][0][jj+(fsk->nx/2+1)*ii]);
	double re1=creal(f->a_temp[i_t][1][jj+(fsk->nx/2+1)*ii]);
	double im1=cimag(f->a_temp[i_t][1][jj+(fsk->nx/2+1)*ii]);
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
  }
  nmt_field_flat_free(f);

  //With templates and purification (nothing should change)
  f=nmt_field_flat_alloc(fsk->nx,fsk->ny,fsk->lx,fsk->ly,mask,2,maps,ntemp,temp,
			 nbpw+1,larr,beam,1,1,1E-5,0,0,0);
  //Since maps and templates are the same, template-deprojected map should be 0
  for(ii=0;ii<nmaps;ii++) {
    int jj;
    for(jj=0;jj<npix;jj++)
      ASSERT_DBL_NEAR_TOL(0.0,f->maps[ii][jj],1E-10);
  }
  //Harmonic transform
  for(ii=0;ii<fsk->ny;ii++) {
    int jj;
    for(jj=0;jj<=fsk->nx/2;jj++) {
      int i_t;
      for(i_t=0;i_t<ntemp;i_t++) {
	double re0=creal(f->a_temp[i_t][0][jj+(fsk->nx/2+1)*ii]);
	double im0=cimag(f->a_temp[i_t][0][jj+(fsk->nx/2+1)*ii]);
	double re1=creal(f->a_temp[i_t][1][jj+(fsk->nx/2+1)*ii]);
	double im1=cimag(f->a_temp[i_t][1][jj+(fsk->nx/2+1)*ii]);
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
  }
  nmt_field_flat_free(f);

  //Free inputs
  for(ii=0;ii<ntemp;ii++) {
    int jj;
    for(jj=0;jj<nmaps;jj++)
      dftw_free(temp[ii][jj]);
    free(temp[ii]);
  }
  for(ii=0;ii<nmaps;ii++)
    dftw_free(maps[ii]);
  free(maps);
  
  nmt_flatsky_info_free(fsk);
  free(temp);
  free(beam);
  free(larr);
  free(mask);
}

CTEST(nmt,field_flat_synfast) {
  int ii,im1,im2,l,if1,if2;
  int nbpw=30;
  int nfields=3;
  int field_spins[3]={0,2,0};
  int field_nmaps[3]={1,2,1};
  int nmaps=4;
  int ncls_pass=(nmaps*(nmaps+1))/2;
  int ncls=nmaps*nmaps;
  nmt_flatsky_info *fsk=nmt_flatsky_info_alloc(141,311,M_PI/180,M_PI/180);
  double lmax_x=fsk->nx*M_PI/fsk->lx;
  double lmax_y=fsk->ny*M_PI/fsk->ly;
  double lmax=sqrt(lmax_x*lmax_x+lmax_y*lmax_y);
  double lpivot=lmax/6.;
  double alpha_pivot=1.;
  double *larr=my_malloc((nbpw+1)*sizeof(double));
  double **cells_in=my_malloc(ncls*sizeof(double *));
  double **cells_out=my_malloc(ncls*sizeof(double *));
  double **cells_pass=my_malloc(ncls_pass*sizeof(double *));
  double **beams=my_malloc(nfields*sizeof(double *));
  long *npixls=my_calloc(nbpw,sizeof(long));

  for(l=0;l<=nbpw;l++)
    larr[l]=l*lmax/nbpw;

  //Initialize beams
  for(ii=0;ii<nfields;ii++) {
    beams[ii]=my_malloc((nbpw+1)*sizeof(double *));
    for(l=0;l<=nbpw;l++)
      beams[ii][l]=1.;
  }

  //Initialize power spectra
  for(im1=0;im1<nmaps;im1++) {
    for(im2=0;im2<nmaps;im2++) {
      int index=im2+nmaps*im1;
      cells_in[index]=my_malloc((nbpw+1)*sizeof(double));
      cells_out[index]=my_malloc((nbpw+1)*sizeof(double));
      for(l=0;l<=nbpw;l++) {
	if(im1==im2)
	  cells_in[index][l]=pow((2*lpivot)/(larr[l]+lpivot),alpha_pivot);
	else
	  cells_in[index][l]=0;
      }
    }
  }
  int icl=0;
  for(im1=0;im1<nmaps;im1++) {
    for(im2=im1;im2<nmaps;im2++) {
      cells_pass[icl]=cells_in[im2+nmaps*im1];
      icl++;
    }
  }

  //Count number of modes per bandpower
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

  //Generate maps
  flouble **maps=nmt_synfast_flat(fsk->nx,fsk->ny,fsk->lx,fsk->ly,nfields,field_spins,
				  nbpw+1,larr,beams,nbpw+1,larr,cells_pass,1234);

  //Compute power spectra
  im1=0;
  for(if1=0;if1<nfields;if1++) {
    im2=0;
    for(if2=0;if2<nfields;if2++) {
      int ncls_here=field_nmaps[if1]*field_nmaps[if2];
      double **cells_here=my_malloc(ncls_here*sizeof(double *));
      for(ii=0;ii<ncls_here;ii++)
	cells_here[ii]=my_malloc(nbpw*sizeof(double));
      
      fs_anafast(fsk,bpw,&(maps[im1]),&(maps[im2]),field_spins[if1],field_spins[if2],
		 cells_here);
      int i1;
      for(i1=0;i1<field_nmaps[if1];i1++) {
      	int i2;
      	for(i2=0;i2<field_nmaps[if2];i2++) {
	  int index_here=i2+field_nmaps[if2]*i1;
	  int index_out=im2+i2+nmaps*(im1+i1);
	  for(l=0;l<nbpw;l++)
	    cells_out[index_out][l]=cells_here[index_here][l];
	}
      }

      for(ii=0;ii<ncls_here;ii++)
      	free(cells_here[ii]);
      free(cells_here);
      im2+=field_nmaps[if2];
    }
    im1+=field_nmaps[if1];
  }

  //Compare with input and check for >5-sigma deviations
  for(l=0;l<nbpw;l++) {
    for(im1=0;im1<nmaps;im1++) {
      for(im2=0;im2<nmaps;im2++) {
	double c11=0.5*(cells_in[im1+nmaps*im1][l]+cells_in[im1+nmaps*im1][l+1]);
	double c12=0.5*(cells_in[im2+nmaps*im1][l]+cells_in[im2+nmaps*im1][l+1]);
	double c21=0.5*(cells_in[im1+nmaps*im2][l]+cells_in[im1+nmaps*im2][l+1]);
	double c22=0.5*(cells_in[im2+nmaps*im2][l]+cells_in[im2+nmaps*im2][l+1]);
	double sig=sqrt((c11*c22+c12*c21)/npixls[ii]);
	double diff=fabs(cells_out[im2+nmaps*im1][l]-c12);
	//Check that there are no >5-sigma fluctuations around input power spectrum
	ASSERT_TRUE((int)(diff<5*sig));
      }
    }
  }

  nmt_flatsky_info_free(fsk);
  for(im1=0;im1<nmaps;im1++)
    dftw_free(maps[im1]);
  free(maps);
  nmt_bins_flat_free(bpw);
  for(ii=0;ii<nfields;ii++)
    free(beams[ii]);
  for(ii=0;ii<ncls;ii++) {
    free(cells_in[ii]);
    free(cells_out[ii]);
  }
  free(larr);
  free(cells_in);
  free(cells_out);
  free(cells_pass);
  free(beams);
  free(npixls);
}
