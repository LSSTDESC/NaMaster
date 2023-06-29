#include "config.h"
#include "utils.h"

nmt_k_function *nmt_k_function_alloc(int nk,flouble *karr,flouble *farr,flouble y0,flouble yf,int is_const)
{
  nmt_k_function *f=my_malloc(sizeof(nmt_k_function));
  f->is_const=is_const;
  f->y0=y0;
  if(!is_const) {
    f->x0=karr[0];
    f->xf=karr[nk-1];
    f->yf=yf;
    f->spl=gsl_spline_alloc(gsl_interp_linear,nk);
    gsl_spline_init(f->spl,karr,farr,nk);
  }
  return f;
}

void nmt_k_function_free(nmt_k_function *f)
{
  if(!(f->is_const))
    gsl_spline_free(f->spl);
  free(f);
}

flouble nmt_k_function_eval(nmt_k_function *f,flouble k,gsl_interp_accel *intacc)
{
  if((f->is_const) || (k<=f->x0))
    return f->y0;
  else if(k>=f->xf)
    return f->yf;
  else
    return gsl_spline_eval(f->spl,k,intacc);
}

#define N_DELL 1
nmt_flatsky_info *nmt_flatsky_info_alloc(int nx,int ny,flouble lx,flouble ly)
{
  nmt_flatsky_info *fs=my_malloc(sizeof(nmt_flatsky_info));
  fs->nx=nx;
  fs->ny=ny;
  fs->npix=nx*((long)ny);
  fs->lx=lx;
  fs->ly=ly;
  fs->pixsize=lx*ly/(nx*ny);

  int ii;
  flouble dkx=2*M_PI/lx;
  flouble dky=2*M_PI/ly;
  flouble kmax_x=dkx*(nx/2);
  flouble kmax_y=dky*(ny/2);
  double dk=NMT_MIN(dkx,dky);
  double kmax=sqrt(kmax_y*kmax_y+kmax_x*kmax_x);
  fs->dell=N_DELL*dk;
  fs->i_dell=1./fs->dell;
  fs->n_ell=0;
  while((fs->n_ell+1)*fs->dell<=kmax)
    fs->n_ell++;
  fs->ell_min=my_malloc(fs->n_ell*sizeof(flouble));
  for(ii=0;ii<fs->n_ell;ii++)
    fs->ell_min[ii]=ii*fs->dell;

  return fs;
}

void nmt_flatsky_info_free(nmt_flatsky_info *fs)
{
  free(fs->ell_min);
  free(fs);
}

void nmt_field_flat_free(nmt_field_flat *fl)
{
  int imap,itemp;
  nmt_k_function_free(fl->beam);

  nmt_flatsky_info_free(fl->fs);
  dftw_free(fl->mask);
  if(!(fl->mask_only)) {
    for(imap=0;imap<fl->nmaps;imap++)
      dftw_free(fl->alms[imap]);
    free(fl->alms);
    if(fl->ntemp>0)
      gsl_matrix_free(fl->matrix_M);
  }

  if(!(fl->lite)) {
    for(imap=0;imap<fl->nmaps;imap++)
      dftw_free(fl->maps[imap]);
    free(fl->maps);
    if(fl->ntemp>0) {
      for(itemp=0;itemp<fl->ntemp;itemp++) {
        for(imap=0;imap<fl->nmaps;imap++) {
          dftw_free(fl->temp[itemp][imap]);
          dftw_free(fl->a_temp[itemp][imap]);
        }
        free(fl->temp[itemp]);
        free(fl->a_temp[itemp]);
      }
      free(fl->temp);
      free(fl->a_temp);
    }

    if(fl->a_mask!=NULL) {
      for(imap=0;imap<fl->nmaps;imap++)
        dftw_free(fl->a_mask[imap]);
      free(fl->a_mask);
    }
  }
  free(fl);
}

static void walm_x_lpower(nmt_flatsky_info *fs,fcomplex **walm_in,fcomplex **walm_out,int power)
{
#pragma omp parallel default(none) \
  shared(fs,walm_in,walm_out,power)
  {
    int iy;
    flouble dkx=2*M_PI/fs->nx;
    flouble dky=2*M_PI/fs->ny;

#pragma omp for   
    for(iy=0;iy<fs->ny;iy++) {
      int ix;
      flouble ky;
      if(2*iy<=fs->ny)
	ky=iy*dky;
      else
	ky=-(fs->ny-iy)*dky;
      for(ix=0;ix<=fs->nx/2;ix++) {
	int ipow;
	flouble kpow=1;
	flouble kx=ix*dkx;
	long index=ix+((long)(fs->nx/2+1))*iy;
	flouble kmod=sqrt(kx*kx+ky*ky);
	for(ipow=0;ipow<power;ipow++)
	  kpow*=kmod;
	walm_out[0][index]=-walm_in[0][index]*kpow;
	walm_out[1][index]=0;
      }
    } //end omp for
  } //end omp parallel
}

void nmt_purify_flat(nmt_field_flat *fl,flouble *mask,fcomplex **walm0,
		     flouble **maps_in,flouble **maps_out,fcomplex **alms)
{
  long ip;
  int imap;
  int purify[2]={0,0};
  long nmodes=fl->fs->ny*((long)(fl->fs->nx/2+1));
  flouble  **pmap=my_malloc(fl->nmaps*sizeof(flouble *));
  flouble  **wmap=my_malloc(fl->nmaps*sizeof(flouble *));
  fcomplex **walm=my_malloc(fl->nmaps*sizeof(fcomplex *));
  fcomplex **palm=my_malloc(fl->nmaps*sizeof(fcomplex *));
  fcomplex **alm_out=my_malloc(fl->nmaps*sizeof(fcomplex *));
  for(imap=0;imap<fl->nmaps;imap++) {
    pmap[imap]=dftw_malloc(fl->npix*sizeof(flouble));
    wmap[imap]=dftw_malloc(fl->npix*sizeof(flouble));
    walm[imap]=dftw_malloc(nmodes*sizeof(fcomplex));
    palm[imap]=dftw_malloc(nmodes*sizeof(fcomplex));
    for(ip=0;ip<nmodes;ip++)
      walm[imap][ip]=walm0[imap][ip];
    alm_out[imap]=dftw_malloc(nmodes*sizeof(fcomplex)); 
  }

  if(fl->pure_e)
    purify[0]=1;
  if(fl->pure_b)
    purify[1]=1;

  //Product with spin-0 mask
  for(imap=0;imap<fl->nmaps;imap++)
    fs_map_product(fl->fs,maps_in[imap],mask,pmap[imap]);
  //Compute SHT and store in alm_out
  fs_map2alm(fl->fs,1,2,pmap,alm_out);

  //Compute spin-1 mask
  walm_x_lpower(fl->fs,walm0,walm,1);
  fs_alm2map(fl->fs,1,1,wmap,walm);
  //Product with spin-1 mask
  for(ip=0;ip<fl->npix;ip++) {
    pmap[0][ip]=wmap[0][ip]*maps_in[0][ip]+wmap[1][ip]*maps_in[1][ip];
    pmap[1][ip]=wmap[0][ip]*maps_in[1][ip]-wmap[1][ip]*maps_in[0][ip];
  }
  //Compute DFT, multiply by 2/l and add to alm_out
  fs_map2alm(fl->fs,1,1,pmap,palm);
  for(imap=0;imap<fl->nmaps;imap++) {
    if(purify[imap]) {
#pragma omp parallel default(none) shared(fl,imap,alm_out,palm)
      {
	int iy;
	flouble dkx=2*M_PI/fl->fs->nx;
	flouble dky=2*M_PI/fl->fs->ny;
	
#pragma omp for
	for(iy=0;iy<fl->fs->ny;iy++) {
	  int ix;
	  flouble ky;
	  if(2*iy<=fl->fs->ny)
	    ky=iy*dky;
	  else
	    ky=-(fl->fs->ny-iy)*dky;
	  for(ix=0;ix<=fl->fs->nx/2;ix++) {
	    flouble kx=ix*dkx;
	    long index=ix+((long)(fl->fs->nx/2+1))*iy;
	    flouble kmod=sqrt(kx*kx+ky*ky);
	    if(kmod>0)
	      alm_out[imap][index]+=2*palm[imap][index]/kmod;
	  }
	} //end omp for
      } //end omp parallel
    }
  }

  //Compute spin-2 mask
  walm_x_lpower(fl->fs,walm0,walm,2);
  fs_alm2map(fl->fs,1,2,wmap,walm);
  //Product with spin-2 mask
  for(ip=0;ip<fl->npix;ip++) { //Extra minus sign because of the scalar SHT below
    pmap[0][ip]=-1*(wmap[0][ip]*maps_in[0][ip]+wmap[1][ip]*maps_in[1][ip]);
    pmap[1][ip]=-1*(wmap[0][ip]*maps_in[1][ip]-wmap[1][ip]*maps_in[0][ip]);
  }
  //Compute DFT, multiply by 1/l^2 and add to alm_out
  fs_map2alm(fl->fs,2,0,pmap,palm);
  for(imap=0;imap<fl->nmaps;imap++) {
    if(purify[imap]) {
#pragma omp parallel default(none) shared(fl,imap,alm_out,palm)
      {
	int iy;
	flouble dkx=2*M_PI/fl->fs->nx;
	flouble dky=2*M_PI/fl->fs->ny;
	
#pragma omp for
	for(iy=0;iy<fl->fs->ny;iy++) {
	  int ix;
	  flouble ky;
	  if(2*iy<=fl->fs->ny)
	    ky=iy*dky;
	  else
	    ky=-(fl->fs->ny-iy)*dky;
	  for(ix=0;ix<=fl->fs->nx/2;ix++) {
	    flouble kx=ix*dkx;
	    long index=ix+((long)(fl->fs->nx/2+1))*iy;
	    flouble kmod2=kx*kx+ky*ky;
	    if(kmod2>0)
	      alm_out[imap][index]+=palm[imap][index]/kmod2;
	  }
	} //end omp for
      } //end omp parallel
    }
  }

  for(imap=0;imap<fl->nmaps;imap++) {
    for(ip=0;ip<nmodes;ip++)
      alms[imap][ip]=alm_out[imap][ip];
  }
  fs_alm2map(fl->fs,1,2,maps_out,alm_out);

  for(imap=0;imap<fl->nmaps;imap++) {
    dftw_free(pmap[imap]);
    dftw_free(wmap[imap]);
    dftw_free(palm[imap]);
    dftw_free(walm[imap]);
    dftw_free(alm_out[imap]);
  }
  free(pmap);
  free(wmap);
  free(palm);
  free(walm);
  free(alm_out);
}

nmt_field_flat *nmt_field_flat_alloc(int nx,int ny,flouble lx,flouble ly,
				     flouble *mask,int spin,flouble **maps,int ntemp,flouble ***temp,
				     int nl_beam,flouble *l_beam,flouble *beam,
				     int pure_e,int pure_b,double tol_pinv,int masked_input,
                                     int is_lite,int mask_only)
{
  long ip;
  int ii,itemp,itemp2,imap;
  nmt_field_flat *fl=my_malloc(sizeof(nmt_field_flat));
  fl->fs=nmt_flatsky_info_alloc(nx,ny,lx,ly);
  fl->npix=nx*((long)ny);
  fl->spin=spin;
  if(spin) fl->nmaps=2;
  else fl->nmaps=1;
  fl->ntemp=ntemp;
  fl->mask_only=mask_only;
  if(mask_only)
    is_lite=1;
  fl->lite=is_lite;

  if((pure_e || pure_b) && (spin!=2))
    report_error(NMT_ERROR_VALUE,"Purification only implemented for spin-2 fields\n");
  fl->pure_e=0;
  fl->pure_b=0;
  if(spin==2) {
    if(pure_e)
      fl->pure_e=1;
    if(pure_b)
      fl->pure_b=1;
  }

  if(beam==NULL)
    fl->beam=nmt_k_function_alloc(-1,NULL,NULL,1.,1.,1);
  else
    fl->beam=nmt_k_function_alloc(nl_beam,l_beam,beam,beam[0],0.,0);

  fl->mask=dftw_malloc(fl->npix*sizeof(flouble));
  for(ip=0;ip<fl->npix;ip++)
    fl->mask[ip]=mask[ip];

  fl->maps=NULL;
  fl->temp=NULL;
  if(mask_only) {
    fl->a_mask=NULL;
    fl->alms=NULL;
    fl->a_temp=NULL;
    return fl;
  }

  //Store unmasked fields if purifying
  // Instead of doing this we could just unmask the maps
  // before purifying, but it turns out that the floating point
  // errors incurred while doing that are pretty nasty around
  // the mask edges, so this is safer.
  flouble **maps_unmasked;
  flouble ***temp_unmasked;
  if(fl->spin && (fl->pure_e || fl->pure_b)) {
    maps_unmasked=my_malloc(fl->nmaps*sizeof(flouble *));
    for(imap=0;imap<fl->nmaps;imap++) {
      maps_unmasked[imap]=dftw_malloc(fl->npix*sizeof(flouble));
      fs_mapcpy(fl->fs,maps_unmasked[imap],maps[imap]);
      //Unmask if masked
      if(masked_input) {
        for(ip=0;ip<fl->npix;ip++) {
          if(fl->mask[ip]>0)
            maps_unmasked[imap][ip]/=fl->mask[ip];
          else
            maps_unmasked[imap][ip]=0;
        }
      }
    }
    if(fl->ntemp>0) {
      temp_unmasked=my_malloc(fl->ntemp*sizeof(flouble **));
      for(itemp=0;itemp<fl->ntemp;itemp++) {
        temp_unmasked[itemp]=my_malloc(fl->nmaps*sizeof(flouble *));
        for(imap=0;imap<fl->nmaps;imap++) {
          temp_unmasked[itemp][imap]=dftw_malloc(fl->npix*sizeof(flouble));
          fs_mapcpy(fl->fs,temp_unmasked[itemp][imap],temp[itemp][imap]);
          //Unmask if masked
          if(masked_input) {
            for(ip=0;ip<fl->npix;ip++) {
              if(mask[ip]>0)
                temp_unmasked[itemp][imap][ip]/=fl->mask[ip];
              else
                temp_unmasked[itemp][imap][ip]=0;
            }
          }
        }
      }
    }
  }

  //Mask if unmasked
  if(!masked_input) {
    for(imap=0;imap<fl->nmaps;imap++)
      fs_map_product(fl->fs,maps[imap],fl->mask,maps[imap]);

    if(fl->ntemp>0) {
      for(itemp=0;itemp<fl->ntemp;itemp++) {
        for(imap=0;imap<fl->nmaps;imap++)
          fs_map_product(fl->fs,temp[itemp][imap],fl->mask,temp[itemp][imap]);
      }
    }
  }

  if(fl->ntemp>0) {
    //Compute normalization matrix
    fl->matrix_M=gsl_matrix_alloc(fl->ntemp,fl->ntemp);
    for(itemp=0;itemp<fl->ntemp;itemp++) {
      for(itemp2=itemp;itemp2<fl->ntemp;itemp2++) {
	flouble matrix_element=0;
	for(imap=0;imap<fl->nmaps;imap++)
	  matrix_element+=fs_map_dot(fl->fs,temp[itemp][imap],temp[itemp2][imap]);
	gsl_matrix_set(fl->matrix_M,itemp,itemp2,matrix_element);
	if(itemp2!=itemp)
	  gsl_matrix_set(fl->matrix_M,itemp2,itemp,matrix_element);
      }
    }
    moore_penrose_pinv(fl->matrix_M,tol_pinv);

    //Deproject
    flouble *prods=my_calloc(fl->ntemp,sizeof(flouble));
    for(itemp=0;itemp<fl->ntemp;itemp++) {
      for(imap=0;imap<fl->nmaps;imap++) 
	prods[itemp]+=fs_map_dot(fl->fs,temp[itemp][imap],maps[imap]);
    }
    for(itemp=0;itemp<fl->ntemp;itemp++) {
      flouble alpha=0;
      for(itemp2=0;itemp2<fl->ntemp;itemp2++) {
	double mij=gsl_matrix_get(fl->matrix_M,itemp,itemp2);
	alpha+=mij*prods[itemp2];
      }
#ifdef _DEBUG
      printf("alpha_%d = %lE\n",itemp,alpha);
#endif //_DEBUG
      for(imap=0;imap<fl->nmaps;imap++) {
	long ip;
	for(ip=0;ip<fl->npix;ip++) {
	  maps[imap][ip]-=alpha*temp[itemp][imap][ip];
          //Unmasked fields too if purifying!
          if(fl->spin && (fl->pure_e || fl->pure_b))
            maps_unmasked[imap][ip]-=alpha*temp_unmasked[itemp][imap][ip];
        }
      }
    }
    free(prods);
  }

  //Allocate Fourier coefficients
  long nmodes=fl->fs->ny*((long)(fl->fs->nx/2+1));
  fl->alms=my_malloc(fl->nmaps*sizeof(fcomplex *));
  for(ii=0;ii<fl->nmaps;ii++)
    fl->alms[ii]=dftw_malloc(nmodes*sizeof(fcomplex));
  if(is_lite)
    fl->a_temp=NULL;
  else {
    if(fl->ntemp>0) {
      fl->a_temp=my_malloc(fl->ntemp*sizeof(fcomplex **));
      for(itemp=0;itemp<fl->ntemp;itemp++) {
        fl->a_temp[itemp]=my_malloc(fl->nmaps*sizeof(fcomplex *));
        for(imap=0;imap<fl->nmaps;imap++)
          fl->a_temp[itemp][imap]=dftw_malloc(nmodes*sizeof(fcomplex));
      }
    }
  }
  fl->a_mask=NULL;

  if(fl->spin && (fl->pure_e || fl->pure_b)) {
    //If purification is needed:
    // 1- Compute mask alms
    // 2- Purify de-contaminated map
    // 3- Compute purified contaminats

    //Compute mask DFT (store in fl->a_mask
    fcomplex **a_mask=my_malloc(fl->nmaps*sizeof(fcomplex *));
    for(imap=0;imap<fl->nmaps;imap++)
      a_mask[imap]=dftw_malloc(nmodes*sizeof(fcomplex));
    fs_map2alm(fl->fs,1,0,&(fl->mask),a_mask);

    //Purify map
    nmt_purify_flat(fl,fl->mask,a_mask,maps_unmasked,maps,fl->alms);

    if((!is_lite) && (fl->ntemp>0)) {
      //Compute purified contaminant DFTs
      for(itemp=0;itemp<fl->ntemp;itemp++) {
	nmt_purify_flat(fl,fl->mask,a_mask,temp_unmasked[itemp],temp[itemp],fl->a_temp[itemp]);
	for(imap=0;imap<fl->nmaps;imap++)//Store non-pure map
	  fs_map_product(fl->fs,temp_unmasked[itemp][imap],fl->mask,temp[itemp][imap]);
      }
      //IMPORTANT: at this stage, fl->maps and fl->alms contain the purified map and SH coefficients
      //           However, although fl->a_temp contains the purified SH coefficients,
      //           fl->temp contains the ***non-purified*** maps. This is to speed up the calculation
      //           of the deprojection bias.
    }

    for(imap=0;imap<fl->nmaps;imap++)
      dftw_free(maps_unmasked[imap]);
    free(maps_unmasked);
    if(fl->ntemp>0) {
      for(itemp=0;itemp<fl->ntemp;itemp++) {
	for(imap=0;imap<fl->nmaps;imap++)
          dftw_free(temp_unmasked[itemp][imap]);
        free(temp_unmasked[itemp]);
      }
      free(temp_unmasked);
    }

    if(is_lite) {
      for(imap=0;imap<fl->nmaps;imap++)
        dftw_free(a_mask[imap]);
      free(a_mask);
    }
    else
      fl->a_mask=a_mask;
  }
  else {
    //If no purification, just and SHT
    //Masked map and spherical harmonic coefficients
    fs_map2alm(fl->fs,1,fl->spin,maps,fl->alms);

    //Compute template DFT too
    if((!is_lite) && (fl->ntemp>0)) {
      for(itemp=0;itemp<fl->ntemp;itemp++)
	fs_map2alm(fl->fs,1,fl->spin,temp[itemp],fl->a_temp[itemp]);
    }
  }

  if(!is_lite) {
    fl->maps=my_malloc(fl->nmaps*sizeof(flouble *));
    for(imap=0;imap<fl->nmaps;imap++) {
      fl->maps[imap]=dftw_malloc(fl->npix*sizeof(flouble));
      fs_mapcpy(fl->fs,fl->maps[imap],maps[imap]);
    }
    if(fl->ntemp>0) {
      fl->temp=my_malloc(fl->ntemp*sizeof(flouble **));
      for(itemp=0;itemp<fl->ntemp;itemp++) {
        fl->temp[itemp]=my_malloc(fl->nmaps*sizeof(flouble *));
        for(imap=0;imap<fl->nmaps;imap++) {
          fl->temp[itemp][imap]=dftw_malloc(fl->npix*sizeof(flouble));
          fs_mapcpy(fl->fs,fl->temp[itemp][imap],temp[itemp][imap]);
        }
      }
    }
  }

  return fl;
}

flouble **nmt_synfast_flat(int nx,int ny,flouble lx,flouble ly,int nfields,int *spin_arr,
			   int nl_beam,flouble *l_beam,flouble **beam_fields,
			   int nl_cell,flouble *l_cell,flouble **cell_fields,
			   int seed)
{
  int ifield,imap;
  int nmaps=0,ncls=0;
  long npix=nx*ny;
  nmt_k_function **beam,**cell;
  flouble **maps;
  fcomplex **alms;
  nmt_flatsky_info *fs=nmt_flatsky_info_alloc(nx,ny,lx,ly);
  for(ifield=0;ifield<nfields;ifield++) {
    int nmp=1;
    if(spin_arr[ifield]) nmp=2;
    nmaps+=nmp;
  }

  imap=0;
  beam=my_malloc(nmaps*sizeof(nmt_k_function *));
  maps=my_malloc(nmaps*sizeof(flouble *));
  for(ifield=0;ifield<nfields;ifield++) {
    int imp,nmp=1;
    if(spin_arr[ifield]) nmp=2;
    for(imp=0;imp<nmp;imp++) {
      beam[imap+imp]=nmt_k_function_alloc(nl_beam,l_beam,beam_fields[ifield],beam_fields[ifield][0],0.,0);
      maps[imap+imp]=dftw_malloc(npix*sizeof(flouble));
    }
    imap+=nmp;
  }

  ncls=(nmaps*(nmaps+1))/2;
  cell=my_malloc(ncls*sizeof(nmt_k_function *));
  for(imap=0;imap<ncls;imap++)
    cell[imap]=nmt_k_function_alloc(nl_cell,l_cell,cell_fields[imap],cell_fields[imap][0],0.,0);

  alms=fs_synalm(nx,ny,lx,ly,nmaps,cell,beam,seed);

  for(imap=0;imap<nmaps;imap++)
    nmt_k_function_free(beam[imap]);
  free(beam);
  for(imap=0;imap<ncls;imap++)
    nmt_k_function_free(cell[imap]);
  free(cell);

  imap=0;
  for(ifield=0;ifield<nfields;ifield++) {
    int imp,nmp=1;
    if(spin_arr[ifield]) nmp=2;
    fs_alm2map(fs,1,spin_arr[ifield],&(maps[imap]),&(alms[imap]));
    for(imp=0;imp<nmp;imp++)
      dftw_free(alms[imap+imp]);
    imap+=nmp;
  }
  free(alms);
  nmt_flatsky_info_free(fs);

  return maps;
}
