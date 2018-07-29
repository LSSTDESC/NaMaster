#include "utils.h"

void nmt_field_free(nmt_field *fl)
{
  int imap,itemp;
  free(fl->beam);

  for(imap=0;imap<fl->nmaps;imap++) {
    free(fl->maps[imap]);
    free(fl->alms[imap]);
  }
  free(fl->mask);
  if(fl->ntemp>0) {
    for(itemp=0;itemp<fl->ntemp;itemp++) {
      for(imap=0;imap<fl->nmaps;imap++) {
	free(fl->temp[itemp][imap]);
	free(fl->a_temp[itemp][imap]);
      }
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
      free(fl->a_mask[imap]);
    free(fl->a_mask);
  }
  free(fl);
}

void nmt_purify(nmt_field *fl,flouble *mask,fcomplex **walm0,
		flouble **maps_in,flouble **maps_out,fcomplex **alms)
{
  long ip;
  int imap,mm,ll;
  int purify[2]={0,0};
  flouble *f_l=my_malloc((fl->lmax+1)*sizeof(flouble));
  flouble  **pmap=my_malloc(fl->nmaps*sizeof(flouble *));
  flouble  **wmap=my_malloc(fl->nmaps*sizeof(flouble *));
  fcomplex **walm=my_malloc(fl->nmaps*sizeof(fcomplex *));
  fcomplex **palm=my_malloc(fl->nmaps*sizeof(fcomplex *));
  fcomplex **alm_out=my_malloc(fl->nmaps*sizeof(fcomplex *));
  for(imap=0;imap<fl->nmaps;imap++) {
    walm[imap]=my_calloc(he_nalms(fl->lmax),sizeof(fcomplex));
    palm[imap]=my_calloc(he_nalms(fl->lmax),sizeof(fcomplex));
    pmap[imap]=my_calloc(fl->npix,sizeof(flouble));
    wmap[imap]=my_calloc(fl->npix,sizeof(flouble));
    memcpy(walm[imap],walm0[imap],he_nalms(fl->lmax)*sizeof(fcomplex));
    alm_out[imap]=my_calloc(he_nalms(fl->lmax),sizeof(fcomplex));
  }

  if(fl->pure_e)
    purify[0]=1;
  if(fl->pure_b)
    purify[1]=1;

  //Product with spin-0 mask
  for(imap=0;imap<fl->nmaps;imap++)
    he_map_product(fl->nside,maps_in[imap],mask,pmap[imap]);
  //Compute SHT and store in alm_out
  he_map2alm(fl->nside,fl->lmax,1,2,pmap      ,alm_out,HE_NITER_DEFAULT);

  //Compute spin-1 mask
  for(ll=0;ll<=fl->lmax;ll++) //The minus sign is because of the definition of E-modes
    f_l[ll]=-sqrt((ll+1.)*ll);
  he_alter_alm(fl->lmax,-1.,walm[0],walm[0],f_l,0); //TODO: There may be a -1 sign here
  he_alm2map(fl->nside,fl->lmax,1,1,wmap,walm);
  //Product with spin-1 mask
  for(ip=0;ip<fl->npix;ip++) {
    pmap[0][ip]=wmap[0][ip]*maps_in[0][ip]+wmap[1][ip]*maps_in[1][ip];
    pmap[1][ip]=wmap[0][ip]*maps_in[1][ip]-wmap[1][ip]*maps_in[0][ip];
  }
  //Compute SHT, multiply by 2*sqrt((l+1)!(l-2)!/((l-1)!(l+2)!)) and add to alm_out
  he_map2alm(fl->nside,fl->lmax,1,1,pmap       ,palm  ,HE_NITER_DEFAULT);
  for(ll=0;ll<=fl->lmax;ll++) {
    if(ll>1)
      f_l[ll]=2./sqrt((ll+2.)*(ll-1.));
    else
      f_l[ll]=0;
  }
  for(imap=0;imap<fl->nmaps;imap++) {
    if(purify[imap]) {
      for(mm=0;mm<=fl->lmax;mm++) {
	for(ll=mm;ll<=fl->lmax;ll++) {
	  long index=he_indexlm(ll,mm,fl->lmax);
	  alm_out[imap][index]+=f_l[ll]*palm[imap][index];
	}
      }
    }
  }

  //Compute spin-2 mask
  for(ll=0;ll<=fl->lmax;ll++) { //The extra minus sign is because of the scalar SHT below (E-mode def for s=0)
    if(ll>1)
      f_l[ll]=-sqrt((ll+2.)*(ll-1.));
    else
      f_l[ll]=0;
  }
  he_alter_alm(fl->lmax,-1.,walm[0],walm[0],f_l,0); //TODO: There may be a -1 sign here
  he_alm2map(fl->nside,fl->lmax,1,2,wmap,walm);
  //Product with spin-2 mask
  for(ip=0;ip<fl->npix;ip++) {
    pmap[0][ip]=wmap[0][ip]*maps_in[0][ip]+wmap[1][ip]*maps_in[1][ip];
    pmap[1][ip]=wmap[0][ip]*maps_in[1][ip]-wmap[1][ip]*maps_in[0][ip];
  }
  //Compute SHT, multiply by sqrt((l-2)!/(l+2)!) and add to alm_out
  he_map2alm(fl->nside,fl->lmax,2,0,pmap,palm,HE_NITER_DEFAULT);
  for(ll=0;ll<=fl->lmax;ll++) {
    if(ll>1)
      f_l[ll]=1./sqrt((ll+2.)*(ll+1.)*ll*(ll-1.));
    else
      f_l[ll]=0;
  }
  for(imap=0;imap<fl->nmaps;imap++) {
    if(purify[imap]) {
      for(mm=0;mm<=fl->lmax;mm++) {
	for(ll=mm;ll<=fl->lmax;ll++) {
	  long index=he_indexlm(ll,mm,fl->lmax);
	  alm_out[imap][index]+=f_l[ll]*palm[imap][index];
	}
      }
    }
  }

  for(imap=0;imap<fl->nmaps;imap++)
    memcpy(alms[imap],alm_out[imap],he_nalms(fl->lmax)*sizeof(fcomplex));
  he_alm2map(fl->nside,fl->lmax,1,2,maps_out,alms);

  for(imap=0;imap<fl->nmaps;imap++) {
    free(pmap[imap]);
    free(wmap[imap]);
    free(palm[imap]);
    free(walm[imap]);
    free(alm_out[imap]);
  }
  free(pmap);
  free(wmap);
  free(palm);
  free(walm);
  free(alm_out);
}

nmt_field *nmt_field_alloc_sph(long nside,flouble *mask,int pol,flouble **maps,
			       int ntemp,flouble ***temp,flouble *beam,
			       int pure_e,int pure_b,int n_iter_mask_purify,double tol_pinv)
{
  int ii,itemp,itemp2,imap;
  nmt_field *fl=my_malloc(sizeof(nmt_field));
  fl->nside=nside;
  fl->lmax=3*fl->nside-1;
  fl->npix=12*fl->nside*fl->nside;
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

  fl->beam=my_malloc(3*fl->nside*sizeof(flouble));
  if(beam==NULL) {
    for(ii=0;ii<3*fl->nside;ii++)
      fl->beam[ii]=1.;
  }
  else
    memcpy(fl->beam,beam,3*fl->nside*sizeof(flouble));

  fl->mask=my_malloc(fl->npix*sizeof(flouble));
  memcpy(fl->mask,mask,fl->npix*sizeof(flouble));

  fl->maps=my_malloc(fl->nmaps*sizeof(flouble *));
  for(ii=0;ii<fl->nmaps;ii++)
    fl->maps[ii]=my_malloc(fl->npix*sizeof(flouble));

  if(fl->ntemp>0) {
    fl->temp=my_malloc(fl->ntemp*sizeof(flouble **));
    for(itemp=0;itemp<fl->ntemp;itemp++) {
      fl->temp[itemp]=my_malloc(fl->nmaps*sizeof(flouble *));
      for(imap=0;imap<fl->nmaps;imap++) {
	fl->temp[itemp][imap]=my_malloc(fl->npix*sizeof(flouble));
	he_map_product(fl->nside,temp[itemp][imap],fl->mask,fl->temp[itemp][imap]); //Multiply by mask
      }
    }

    //Compute normalization matrix
    fl->matrix_M=gsl_matrix_alloc(fl->ntemp,fl->ntemp);
    for(itemp=0;itemp<fl->ntemp;itemp++) {
      for(itemp2=itemp;itemp2<fl->ntemp;itemp2++) {
	flouble matrix_element=0;
	for(imap=0;imap<fl->nmaps;imap++)
	  matrix_element+=he_map_dot(fl->nside,fl->temp[itemp][imap],fl->temp[itemp2][imap]);
	gsl_matrix_set(fl->matrix_M,itemp,itemp2,matrix_element);
	if(itemp2!=itemp)
	  gsl_matrix_set(fl->matrix_M,itemp2,itemp,matrix_element);
      }
    }
    moore_penrose_pinv(fl->matrix_M,tol_pinv);

    //Deproject
    for(ii=0;ii<fl->nmaps;ii++) //Need to multiply observed map by mask first
      he_map_product(fl->nside,maps[ii],fl->mask,fl->maps[ii]);
    flouble *prods=my_calloc(fl->ntemp,sizeof(flouble));
    for(itemp=0;itemp<fl->ntemp;itemp++) {
      for(imap=0;imap<fl->nmaps;imap++) 
	prods[itemp]+=he_map_dot(fl->nside,fl->temp[itemp][imap],fl->maps[imap]);
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

  //Allocate harmonic coefficients
  fl->alms=my_malloc(fl->nmaps*sizeof(fcomplex *));
  for(ii=0;ii<fl->nmaps;ii++)
    fl->alms[ii]=my_malloc(he_nalms(fl->lmax)*sizeof(fcomplex));
  if(fl->ntemp>0) {
    fl->a_temp=my_malloc(fl->ntemp*sizeof(fcomplex **));
    for(itemp=0;itemp<fl->ntemp;itemp++) {
      fl->a_temp[itemp]=my_malloc(fl->nmaps*sizeof(fcomplex *));
      for(imap=0;imap<fl->nmaps;imap++)
	fl->a_temp[itemp][imap]=my_malloc(he_nalms(fl->lmax)*sizeof(fcomplex));
    }
  }

  if(fl->pol && (fl->pure_e || fl->pure_b)) {
    //If purification is needed:
    // 1- Compute mask alms
    // 2- Purify de-contaminated map
    // 3- Compute purified contaminants

    //Compute mask SHT (store in fl->a_mask)
    fl->a_mask=my_malloc(fl->nmaps*sizeof(fcomplex *));
    for(imap=0;imap<fl->nmaps;imap++)
      fl->a_mask[imap]=my_calloc(he_nalms(fl->lmax),sizeof(fcomplex));
    he_map2alm(fl->nside,fl->lmax,1,0,&(fl->mask),fl->a_mask,n_iter_mask_purify);

    //Purify map
    nmt_purify(fl,fl->mask,fl->a_mask,maps,fl->maps,fl->alms);

    //Compute purified contaminant SHTs
    if(fl->ntemp>0) {
      for(itemp=0;itemp<fl->ntemp;itemp++) {
	nmt_purify(fl,fl->mask,fl->a_mask,temp[itemp],fl->temp[itemp],fl->a_temp[itemp]);
	for(imap=0;imap<fl->nmaps;imap++) //Store non-pure map
	  he_map_product(fl->nside,temp[itemp][imap],fl->mask,fl->temp[itemp][imap]);
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

    //Masked map and harmonic coefficients
    for(imap=0;imap<fl->nmaps;imap++)
      he_map_product(fl->nside,maps[imap],fl->mask,fl->maps[imap]);
    he_map2alm(fl->nside,fl->lmax,1,2*fl->pol,fl->maps,fl->alms,HE_NITER_DEFAULT);

    //Compute template harmonic coefficients too
    if(fl->ntemp>0) {
      for(itemp=0;itemp<fl->ntemp;itemp++)
	he_map2alm(fl->nside,fl->lmax,1,2*fl->pol,fl->temp[itemp],fl->a_temp[itemp],HE_NITER_DEFAULT);
    }
  }
  
  return fl;
}

flouble **nmt_synfast_sph(int nside,int nfields,int *spin_arr,int lmax,
			  flouble **cells,flouble **beam_fields,int seed)
{
  int ifield,imap;
  int nmaps=0;
  long npix=12*((long)nside)*nside;
  flouble **beam,**maps;
  fcomplex **alms;
  for(ifield=0;ifield<nfields;ifield++) {
    int nmp=1;
    if(spin_arr[ifield]) nmp=2;
    nmaps+=nmp;
  }

  imap=0;
  beam=my_malloc(nmaps*sizeof(flouble *));
  maps=my_malloc(nmaps*sizeof(flouble *));
  for(ifield=0;ifield<nfields;ifield++) {
    int imp,nmp=1;
    if(spin_arr[ifield]) nmp=2;
    for(imp=0;imp<nmp;imp++) {
      beam[imap+imp]=my_malloc((lmax+1)*sizeof(flouble));
      maps[imap+imp]=my_malloc(npix*sizeof(flouble));
      memcpy(beam[imap+imp],beam_fields[ifield],(lmax+1)*sizeof(flouble));
    }
    imap+=nmp;
  }

  imap=0;
  alms=he_synalm(nside,nmaps,lmax,cells,beam,seed);
  for(ifield=0;ifield<nfields;ifield++) {
    int imp,nmp=1;
    if(spin_arr[ifield]) nmp=2;
    he_alm2map(nside,3*nside-1,1,spin_arr[ifield],&(maps[imap]),&(alms[imap]));
    for(imp=0;imp<nmp;imp++) {
      free(alms[imap+imp]);
      free(beam[imap+imp]);
    }
    imap+=nmp;
  }
  free(alms);
  free(beam);
  return maps;
}

nmt_field *nmt_field_read(char *fname_mask,char *fname_maps,char *fname_temp,char *fname_beam,
			  int pol,int pure_e,int pure_b,int n_iter_mask_purify,double tol_pinv)
{
  long nside,nside_dum;
  nmt_field *fl;
  flouble *beam;
  flouble *mask;
  flouble **maps;
  flouble ***temp;
  int ii,ntemp,itemp,imap,nmaps=1;
  if(pol) nmaps=2;

  //Read mask and compute nside, lmax etc.
  mask=he_read_healpix_map(fname_mask,&nside,0);

  //Read beam
  if(!strcmp(fname_beam,"none"))
    beam=NULL;
  else {
    FILE *fi=my_fopen(fname_beam,"r");
    int nlines=my_linecount(fi); rewind(fi);
    if(nlines!=3*nside)
      report_error(1,"Beam file must have 3*nside rows and two columns\n");
    beam=my_malloc(3*nside*sizeof(flouble));
    for(ii=0;ii<3*nside;ii++) {
      int l;
      double b;
      int stat=fscanf(fi,"%d %lf\n",&l,&b);
      if(stat!=2)
	report_error(1,"Error reading file %s, line %d\n",fname_beam,ii+1);
      if((l>3*nside-1) || (l<0))
	report_error(1,"Wrong multipole %d\n",l);
      beam[l]=b;
    }
  }

  //Read data maps
  maps=my_malloc(nmaps*sizeof(flouble *));
  for(ii=0;ii<nmaps;ii++) {
    maps[ii]=he_read_healpix_map(fname_maps,&(nside_dum),ii);
    if(nside_dum!=nside)
      report_error(1,"Wrong nside %ld\n",nside_dum);
  }

  //Read templates and deproject
  if(strcmp(fname_temp,"none")) {
    int ncols,isnest;
    he_get_file_params(fname_temp,&nside_dum,&ncols,&isnest);
    if(nside_dum!=nside)
      report_error(1,"Wrong nside %ld\n",nside_dum);
    if((ncols==0) || (ncols%nmaps!=0))
      report_error(1,"Not enough templates in file %s\n",fname_temp);
    ntemp=ncols/nmaps;
    temp=my_malloc(ntemp*sizeof(flouble **));
    for(itemp=0;itemp<ntemp;itemp++) {
      temp[itemp]=my_malloc(nmaps*sizeof(flouble *));
      for(imap=0;imap<nmaps;imap++)
	temp[itemp][imap]=he_read_healpix_map(fname_temp,&(nside_dum),itemp*nmaps+imap);
    }
  }
  else {
    ntemp=0;
    temp=NULL;
  }

  fl=nmt_field_alloc_sph(nside,mask,pol,maps,ntemp,temp,beam,pure_e,pure_b,n_iter_mask_purify,tol_pinv);

  if(beam!=NULL)
    free(beam);
  free(mask);
  for(imap=0;imap<nmaps;imap++)
    free(maps[imap]);
  free(maps);
  if(ntemp>0) {
    for(itemp=0;itemp<ntemp;itemp++) {
      for(imap=0;imap<nmaps;imap++)
	free(temp[itemp][imap]);
      free(temp[itemp]);
    }
    free(temp);
  }

  return fl;
}
