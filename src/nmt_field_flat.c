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
    //    f->spl=gsl_spline_alloc(gsl_interp_cspline,nk);
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
  fs->npix=nx*ny;
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
  //  fs->n_cells=my_calloc(fs->n_ell,sizeof(int));
  for(ii=0;ii<fs->n_ell;ii++)
    fs->ell_min[ii]=ii*fs->dell;
  /*
#pragma omp parallel default(none) \
  shared(fs,dkx,dky)
  {
    int iy;

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
	flouble kmod=sqrt(kx*kx+ky*ky);
	int ik=(int)(kmod*fs->i_dell);
	if(ik<fs->n_ell) {
#pragma omp atomic
	  fs->n_cells[ik]++;
	}
      }
    } //end omp for
  } //end omp parallel
  */
  return fs;
}

void nmt_flatsky_info_free(nmt_flatsky_info *fs)
{
  free(fs->ell_min);
  //  free(fs->n_cells);
  free(fs);
}

void nmt_field_flat_free(nmt_field_flat *fl)
{
  int imap,itemp;
  nmt_k_function_free(fl->beam);

  nmt_flatsky_info_free(fl->fs);
  for(imap=0;imap<fl->nmaps;imap++) {
    dftw_free(fl->maps[imap]);
    dftw_free(fl->alms[imap]);
  }
  dftw_free(fl->mask);
  if(fl->ntemp>0) {
    for(itemp=0;itemp<fl->ntemp;itemp++) {
      for(imap=0;imap<fl->nmaps;imap++)
	dftw_free(fl->temp[itemp][imap]);
    }
  }

  free(fl->alms);
  free(fl->maps);
  if(fl->ntemp>0) {
    for(itemp=0;itemp<fl->ntemp;itemp++) {
      free(fl->temp[itemp]);
      free(fl->a_temp[itemp]);
    }
    free(fl->temp);
    free(fl->a_temp);
    gsl_matrix_free(fl->matrix_M);
  }
  if(fl->a_mask!=NULL) {
    for(imap=0;imap<fl->nmaps;imap++)
      dftw_free(fl->a_mask[imap]);
    free(fl->a_mask);
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
	long index=ix+(fs->nx/2+1)*iy;
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
  flouble  **pmap=my_malloc(fl->nmaps*sizeof(flouble *));
  flouble  **wmap=my_malloc(fl->nmaps*sizeof(flouble *));
  fcomplex **walm=my_malloc(fl->nmaps*sizeof(fcomplex *));
  fcomplex **palm=my_malloc(fl->nmaps*sizeof(fcomplex *));
  fcomplex **alm_out=my_malloc(fl->nmaps*sizeof(fcomplex *));
  for(imap=0;imap<fl->nmaps;imap++) {
    pmap[imap]=dftw_malloc(fl->npix*sizeof(flouble));
    wmap[imap]=dftw_malloc(fl->npix*sizeof(flouble));
    walm[imap]=dftw_malloc(fl->fs->ny*(fl->fs->nx/2+1)*sizeof(fcomplex));
    palm[imap]=dftw_malloc(fl->fs->ny*(fl->fs->nx/2+1)*sizeof(fcomplex));
    for(ip=0;ip<fl->fs->ny*(fl->fs->nx/2+1);ip++)
      walm[imap][ip]=walm0[imap][ip];
    alm_out[imap]=dftw_malloc(fl->fs->ny*(fl->fs->nx/2+1)*sizeof(fcomplex)); 
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
	    long index=ix+(fl->fs->nx/2+1)*iy;
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
	    long index=ix+(fl->fs->nx/2+1)*iy;
	    flouble kmod2=kx*kx+ky*ky;
	    if(kmod2>0)
	      alm_out[imap][index]+=palm[imap][index]/kmod2;
	  }
	} //end omp for
      } //end omp parallel
    }
  }

  for(imap=0;imap<fl->nmaps;imap++) {
    for(ip=0;ip<fl->fs->ny*(fl->fs->nx/2+1);ip++)
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
				     flouble *mask,int pol,flouble **maps,int ntemp,flouble ***temp,
				     int nl_beam,flouble *l_beam,flouble *beam,
				     int pure_e,int pure_b,double tol_pinv)
{
  long ip;
  int ii,itemp,itemp2,imap;
  nmt_field_flat *fl=my_malloc(sizeof(nmt_field_flat));
  fl->fs=nmt_flatsky_info_alloc(nx,ny,lx,ly);
  fl->npix=nx*ny;
  fl->pol=pol;
  if(pol) fl->nmaps=2;
  else fl->nmaps=1;
  fl->ntemp=ntemp;

  fl->pure_e=0;
  fl->pure_b=0;
  if(pol) {
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

  fl->maps=my_malloc(fl->nmaps*sizeof(flouble *));
  for(ii=0;ii<fl->nmaps;ii++)
    fl->maps[ii]=dftw_malloc(fl->npix*sizeof(flouble));

  if(fl->ntemp>0) {
    fl->temp=my_malloc(fl->ntemp*sizeof(flouble **));
    for(itemp=0;itemp<fl->ntemp;itemp++) {
      fl->temp[itemp]=my_malloc(fl->nmaps*sizeof(flouble *));
      for(imap=0;imap<fl->nmaps;imap++) {
	fl->temp[itemp][imap]=dftw_malloc(fl->npix*sizeof(flouble));
	fs_map_product(fl->fs,temp[itemp][imap],fl->mask,fl->temp[itemp][imap]); //Multiply by mask
      }
    }
    
    //Compute normalization matrix
    fl->matrix_M=gsl_matrix_alloc(fl->ntemp,fl->ntemp);
    for(itemp=0;itemp<fl->ntemp;itemp++) {
      for(itemp2=itemp;itemp2<fl->ntemp;itemp2++) {
	flouble matrix_element=0;
	for(imap=0;imap<fl->nmaps;imap++)
	  matrix_element+=fs_map_dot(fl->fs,fl->temp[itemp][imap],fl->temp[itemp2][imap]);
	gsl_matrix_set(fl->matrix_M,itemp,itemp2,matrix_element);
	if(itemp2!=itemp)
	  gsl_matrix_set(fl->matrix_M,itemp2,itemp,matrix_element);
      }
    }
    moore_penrose_pinv(fl->matrix_M,tol_pinv);

    //Deproject
    for(ii=0;ii<fl->nmaps;ii++)
      fs_map_product(fl->fs,maps[ii],fl->mask,fl->maps[ii]);
    flouble *prods=my_calloc(fl->ntemp,sizeof(flouble));
    for(itemp=0;itemp<fl->ntemp;itemp++) {
      for(imap=0;imap<fl->nmaps;imap++) 
	prods[itemp]+=fs_map_dot(fl->fs,fl->temp[itemp][imap],fl->maps[imap]);
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
	for(ip=0;ip<fl->npix;ip++)
	  maps[imap][ip]-=alpha*temp[itemp][imap][ip]; //Correct unmasked field (in case of purification)
      }
    }
    free(prods);
  }

  fl->alms=my_malloc(fl->nmaps*sizeof(fcomplex *));
  for(ii=0;ii<fl->nmaps;ii++)
    fl->alms[ii]=dftw_malloc(fl->fs->ny*(fl->fs->nx/2+1)*sizeof(fcomplex));
  if(fl->ntemp>0) {
    fl->a_temp=my_malloc(fl->ntemp*sizeof(fcomplex **));
    for(itemp=0;itemp<fl->ntemp;itemp++) {
      fl->a_temp[itemp]=my_malloc(fl->nmaps*sizeof(fcomplex *));
      for(imap=0;imap<fl->nmaps;imap++)
	fl->a_temp[itemp][imap]=dftw_malloc(fl->fs->ny*(fl->fs->nx/2+1)*sizeof(fcomplex));
    }
  }


  if(fl->pol && (fl->pure_e || fl->pure_b)) {
    //If purification is needed:
    // 1- Compute mask alms
    // 2- Purify de-contaminated map
    // 3- Compute purified contaminats

    //Compute mask DFT (store in fl->a_mask
    fl->a_mask=my_malloc(fl->nmaps*sizeof(fcomplex *));
    for(imap=0;imap<fl->nmaps;imap++)
      fl->a_mask[imap]=dftw_malloc(fl->fs->ny*(fl->fs->nx/2+1)*sizeof(fcomplex));
    fs_map2alm(fl->fs,1,0,&(fl->mask),fl->a_mask);

    //Purify map
    nmt_purify_flat(fl,fl->mask,fl->a_mask,maps,fl->maps,fl->alms);

    //Compute purified contaminant DFTs
    if(fl->ntemp>0) {
      for(itemp=0;itemp<fl->ntemp;itemp++) {
	nmt_purify_flat(fl,fl->mask,fl->a_mask,temp[itemp],fl->temp[itemp],fl->a_temp[itemp]);
	for(imap=0;imap<fl->nmaps;imap++) //Store non-pure map
	  fs_map_product(fl->fs,temp[itemp][imap],fl->mask,fl->temp[itemp][imap]);
      }    
      //IMPORTANT: at this stage, fl->maps and fl->alms contain the purified map and SH coefficients
      //           However, although fl->a_temp contains the purified SH coefficients,
      //           fl->temp contains the ***non-purified*** maps. This is to speed up the calculation
      //           of the deprojection bias.
    }
  }
  else {
    //If no purification, just multiply by mask and SHT
    fl->a_mask=NULL; //No need to store extra-pure mask harmonic coefficients

    //Masked map and spherical harmonic coefficients
    for(imap=0;imap<fl->nmaps;imap++)
      fs_map_product(fl->fs,maps[imap],fl->mask,fl->maps[imap]);
    fs_map2alm(fl->fs,1,2*fl->pol,fl->maps,fl->alms);

    //Compute template DFT too
    if(fl->ntemp>0) {
      for(itemp=0;itemp<fl->ntemp;itemp++)
	fs_map2alm(fl->fs,1,2*fl->pol,fl->temp[itemp],fl->a_temp[itemp]);
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
