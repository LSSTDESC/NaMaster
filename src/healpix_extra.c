
#include "utils.h"
#include <fitsio.h>
#include <chealpix.h>
#include <sharp_almhelpers.h>
#include <sharp_geomhelpers.h>
#include <sharp.h>

#include <stdlib.h>
#include <stdio.h>
#include <namaster.h>
#include <math.h>
#include "utils.h"
#include <fitsio.h>

// libsharp includes for map2alm and alm2map
#include <sharp.h>
#include <sharp_almhelpers.h>
#include <sharp_geomhelpers.h>
#include "sharp_legendre_roots.h"
#include "c_utils.h"
#include "ls_fft.h"

#include <omp.h>


#define MAX_SHT 32

void he_write_healpix_map(flouble **tmap,int nfields,long nside,char *fname)
{
  fitsfile *fptr;
  int ii,status=0;
  char **ttype,**tform,**tunit;
  float *map_dum=my_malloc(he_nside2npix(nside)*sizeof(float));

  ttype=my_malloc(nfields*sizeof(char *));
  tform=my_malloc(nfields*sizeof(char *));
  tunit=my_malloc(nfields*sizeof(char *));
  for(ii=0;ii<nfields;ii++) {
    ttype[ii]=my_malloc(256);
    tform[ii]=my_malloc(256);
    tunit[ii]=my_malloc(256);
    sprintf(ttype[ii],"map %d",ii+1);
    sprintf(tform[ii],"1E");
    sprintf(tunit[ii],"  ");
  }

  fits_create_file(&fptr,fname,&status);
  fits_create_tbl(fptr,BINARY_TBL,0,nfields,ttype,tform,
		  tunit,"BINTABLE",&status);
  fits_write_key(fptr,TSTRING,"PIXTYPE","HEALPIX","HEALPIX Pixelisation",
		 &status);

  fits_write_key(fptr,TSTRING,"ORDERING","RING",
		 "Pixel ordering scheme, either RING or NESTED",&status);
  fits_write_key(fptr,TLONG,"NSIDE",&nside,
		 "Resolution parameter for HEALPIX",&status);
  fits_write_key(fptr,TSTRING,"COORDSYS","G",
		 "Pixelisation coordinate system",&status);
  fits_write_comment(fptr,
		     "G = Galactic, E = ecliptic, C = celestial = equatorial",
		     &status);
  for(ii=0;ii<nfields;ii++) {
    long ip;
    for(ip=0;ip<he_nside2npix(nside);ip++)
      map_dum[ip]=(float)(tmap[ii][ip]);
    fits_write_col(fptr,TFLOAT,ii+1,1,1,he_nside2npix(nside),map_dum,&status);
  }
  fits_close_file(fptr, &status);

  for(ii=0;ii<nfields;ii++) {
    free(ttype[ii]);
    free(tform[ii]);
    free(tunit[ii]);
  }
  free(ttype);
  free(tform);
  free(tunit);
}

flouble *he_read_healpix_map(char *fname,long *nside,int nfield)
{
  //////
  // Reads a healpix map from file fname. The map will be
  // read from column #nfield. It also returns the map's nside.
  int status=0,hdutype,nfound,anynul,ncols;
  long naxes,*naxis,npix;
  fitsfile *fptr;
  flouble *map,nulval;
  char order_in_file[32];
  int nested_in_file=0;

  fits_open_file(&fptr,fname,READONLY,&status);
  fits_movabs_hdu(fptr,2,&hdutype,&status);
  fits_read_key_lng(fptr,"NAXIS",&naxes,NULL,&status);
  naxis=my_malloc(naxes*sizeof(long));
  fits_read_keys_lng(fptr,"NAXIS",1,naxes,naxis,&nfound,&status);
  fits_read_key_lng(fptr,"NSIDE",nside,NULL,&status);
  npix=12*(*nside)*(*nside);
  if(npix%naxis[1]!=0)
    report_error(NMT_ERROR_READ,"FITS file %s corrupt\n",fname);

  if (fits_read_key(fptr, TSTRING, "ORDERING", order_in_file, NULL, &status))
    report_error(NMT_ERROR_READ,"WARNING: Could not find %s keyword in in file %s\n","ORDERING",fname);
  if(!strncmp(order_in_file,"NEST",4))
    nested_in_file=1;

  map=my_malloc(npix*sizeof(flouble));
  fits_get_num_cols(fptr,&ncols,&status);
  if(nfield>=ncols)
    report_error(NMT_ERROR_READ,"Not enough columns in FITS file\n");
#ifdef _SPREC
  fits_read_col(fptr,TFLOAT,nfield+1,1,1,npix,&nulval,map,&anynul,&status);
#else //_SPREC
  fits_read_col(fptr,TDOUBLE,nfield+1,1,1,npix,&nulval,map,&anynul,&status);
#endif //_SPREC
  free(naxis);

  fits_close_file(fptr,&status);

  flouble *map_ring;
  if(nested_in_file) {
    long ipring,ipnest;

    printf("read_healpix_map: input is nested. Transforming to ring.\n");
    map_ring=my_malloc(npix*sizeof(flouble));
    for(ipnest=0;ipnest<npix;ipnest++) {
      nest2ring(*nside,ipnest,&ipring);
      map_ring[ipring]=map[ipnest];
    }
    free(map);
  }
  else
    map_ring=map;

  return map_ring;
}

void he_get_file_params(char *fname,long *nside,int *nfields,int *isnest)
{
  //////
  // Reads a healpix map from file fname. The map will be
  // read from column #nfield. It also returns the map's nside.
  int status=0,hdutype,nfound;
  long naxes,*naxis;
  fitsfile *fptr;
  char order_in_file[32];

  fits_open_file(&fptr,fname,READONLY,&status);
  fits_movabs_hdu(fptr,2,&hdutype,&status);
  fits_read_key_lng(fptr,"NAXIS",&naxes,NULL,&status);
  naxis=my_malloc(naxes*sizeof(long));
  fits_read_keys_lng(fptr,"NAXIS",1,naxes,naxis,&nfound,&status);
  fits_read_key_lng(fptr,"NSIDE",nside,NULL,&status);

  if (fits_read_key(fptr, TSTRING, "ORDERING", order_in_file, NULL, &status)) {
    report_error(NMT_ERROR_READ,"WARNING: Could not find %s keyword in in file %s\n",
		 "ORDERING",fname);
  }

  if(!strncmp(order_in_file,"NEST",4))
    *isnest=1;
  else
    *isnest=0;
  fits_get_num_cols(fptr,nfields,&status);
  free(naxis);
  fits_close_file(fptr,&status);
}


int he_ring_num(long nside,double z)
{
  //Returns ring index for normalized height z
  int iring;

  iring=(int)(nside*(2-1.5*z)+0.5);
  if(z>0.66666666) {
    iring=(int)(nside*sqrt(3*(1-z))+0.5);
    if(iring==0) iring=1;
  }

  if(z<-0.66666666) {
    iring=(int)(nside*sqrt(3*(1+z))+0.5);
    if(iring==0) iring=1;
    iring=4*nside-iring;
  }

  return iring;
}

static void get_ring_limits(long nside,int iz,long *ip_lo,long *ip_hi)
{
  long ir;
  long ipix1,ipix2;
  long npix=12*nside*nside;
  long ncap=2*nside*(nside-1);

  if((iz>=nside)&&(iz<=3*nside)) { //eqt
    ir=iz-nside+1;
    ipix1=ncap+4*nside*(ir-1);
    ipix2=ipix1+4*nside-1;
  }
  else {
    if(iz<nside) { //north
      ir=iz;
      ipix1=2*ir*(ir-1);
      ipix2=ipix1+4*ir-1;
    }
    else { //south
      ir=4*nside-iz;
      ipix1=npix-2*ir*(ir+1);
      ipix2=ipix1+4*ir-1;
    }
  }

  *ip_lo=ipix1;
  *ip_hi=ipix2;
}

void he_query_strip(long nside,double theta1,double theta2,
		    int *pixlist,long *npix_strip)
{
  double z_hi=cos(theta1);
  double z_lo=cos(theta2);
  int irmin,irmax;

  if((theta2<=theta1)||
     (theta1<0)||(theta1>M_PI)||
     (theta2<0)||(theta2>M_PI)) {
    report_error(NMT_ERROR_HPX,"Wrong strip boundaries\n");
  }

  irmin=he_ring_num(nside,z_hi);
  irmax=he_ring_num(nside,z_lo);

  //Count number of pixels in strip
  int iz;
  long npix_in_strip=0;
  for(iz=irmin;iz<=irmax;iz++) {
    long ipix1,ipix2;
    get_ring_limits(nside,iz,&ipix1,&ipix2);
    npix_in_strip+=ipix2-ipix1+1;
  }
  if(*npix_strip<npix_in_strip)
    report_error(NMT_ERROR_MEMORY,"Not enough memory in pixlist\n");
  else
    *npix_strip=npix_in_strip;

  //Count number of pixels in strip
  long i_list=0;
  for(iz=irmin;iz<=irmax;iz++) {
    long ipix1,ipix2,ip;
    get_ring_limits(nside,iz,&ipix1,&ipix2);
    for(ip=ipix1;ip<=ipix2;ip++) {
      pixlist[i_list]=ip;
      i_list++;
    }
  }
}

void he_ring2nest_inplace(flouble *map_in,long nside)
{
  long npix=12*nside*nside;
  flouble *map_out=my_malloc(npix*sizeof(flouble));

#pragma omp parallel default(none)		\
  shared(map_in,nside,npix,map_out)
  {
    long ip;

#pragma omp for
    for(ip=0;ip<npix;ip++) {
      long inest;
      ring2nest(nside,ip,&inest);

      map_out[inest]=map_in[ip];
    } //end omp for
  } //end omp parallel
  memcpy(map_in,map_out,npix*sizeof(flouble));

  free(map_out);
}

void he_nest2ring_inplace(flouble *map_in,long nside)
{
  long npix=12*nside*nside;
  flouble *map_out=my_malloc(npix*sizeof(flouble));

#pragma omp parallel default(none)		\
  shared(map_in,nside,npix,map_out)
  {
    long ip;

#pragma omp for
    for(ip=0;ip<npix;ip++) {
      long iring;
      nest2ring(nside,ip,&iring);

      map_out[iring]=map_in[ip];
    } //end omp for
  } //end omp parallel
  memcpy(map_in,map_out,npix*sizeof(flouble));

  free(map_out);
}

void he_udgrade(flouble *map_in,long nside_in,
		flouble *map_out,long nside_out,
		int nest)
{
  long npix_in=he_nside2npix(nside_in);
  long npix_out=he_nside2npix(nside_out);

  if(nside_in>nside_out) {
    long ii;
    long np_ratio=npix_in/npix_out;
    double i_np_ratio=1./((double)np_ratio);

    for(ii=0;ii<npix_out;ii++) {
      int jj;
      double tot=0;

      if(nest) {
	for(jj=0;jj<np_ratio;jj++)
	  tot+=map_in[jj+ii*np_ratio];
	map_out[ii]=tot*i_np_ratio;
      }
      else {
	long inest_out;

	ring2nest(nside_out,ii,&inest_out);
	for(jj=0;jj<np_ratio;jj++) {
	  long iring_in;

	  nest2ring(nside_in,jj+np_ratio*inest_out,&iring_in);
	  tot+=map_in[iring_in];
	}
	map_out[ii]=tot*i_np_ratio;
      }
    }
  }
  else {
    long ii;
    long np_ratio=npix_out/npix_in;

    for(ii=0;ii<npix_in;ii++) {
      int jj;

      if(nest) {
	flouble value=map_in[ii];

	for(jj=0;jj<np_ratio;jj++)
	  map_out[jj+ii*np_ratio]=value;
      }
      else {
	long inest_in;
	flouble value=map_in[ii];
	ring2nest(nside_in,ii,&inest_in);

	for(jj=0;jj<np_ratio;jj++) {
	  long iring_out;

	  nest2ring(nside_out,jj+inest_in*np_ratio,&iring_out);
	  map_out[iring_out]=value;
	}
      }
    }
  }
}

long he_nside2npix(long nside)
{
  return 12*nside*nside;
}

void he_pix2vec_ring(long nside, long ipix, double *vec)
{
  pix2vec_ring(nside,ipix,vec);
}

static double fmodulo (double v1, double v2)
{
  if (v1>=0)
    return (v1<v2) ? v1 : fmod(v1,v2);
  double tmp=fmod(v1,v2)+v2;
  return (tmp==v2) ? 0. : tmp;
}

static int imodulo (int v1, int v2)
{ int v=v1%v2; return (v>=0) ? v : v+v2; }

static const double twopi=6.283185307179586476925286766559005768394;
static const double twothird=2.0/3.0;
static const double inv_halfpi=0.6366197723675813430755350534900574;
long he_ang2pix(long nside,double cth,double phi)
{
  double ctha=fabs(cth);
  double tt=fmodulo(phi,twopi)*inv_halfpi; /* in [0,4) */

  if (ctha<=twothird) {/* Equatorial region */
    double temp1=nside*(0.5+tt);
    double temp2=nside*cth*0.75;
    int jp=(int)(temp1-temp2); /* index of  ascending edge line */
    int jm=(int)(temp1+temp2); /* index of descending edge line */
    int ir=nside+1+jp-jm; /* ring number counted from cth=2/3 */ /* in {1,2n+1} */
    int kshift=1-(ir&1); /* kshift=1 if ir even, 0 otherwise */
    int ip=(jp+jm-nside+kshift+1)/2; /* in {0,4n-1} */
    ip=imodulo(ip,4*nside);

    return nside*(nside-1)*2 + (ir-1)*4*nside + ip;
  }
  else {  /* North & South polar caps */
    double tp=tt-(int)(tt);
    double tmp=nside*sqrt(3*(1-ctha));
    int jp=(int)(tp*tmp); /* increasing edge line index */
    int jm=(int)((1.0-tp)*tmp); /* decreasing edge line index */
    int ir=jp+jm+1; /* ring number counted from the closest pole */
    int ip=(int)(tt*ir); /* in {0,4*ir-1} */
    ip = imodulo(ip,4*ir);

    if (cth>0)
      return 2*ir*(ir-1)+ip;
    else
      return 12*nside*nside-2*ir*(ir+1)+ip;
  }
}

static int nint_he(double n)
{
  if(n>0) return (int)(n+0.5);
  else if(n<0) return (int)(n-0.5);
  else return 0;
}

static int modu_he(int a,int n)
{
  int moda=a%n;
  if(moda<0)
    moda+=n;
  return moda;
}

void he_in_ring(int nside,int iz,flouble phi0,flouble dphi,
		int *listir,int *nir)
{
  int take_all,conservative,to_top;
  int npix;
  int ncap;
  int diff,jj;
  int nr,nir1,nir2,ir,kshift;
  int ipix1,ipix2;
  int ip_low,ip_hi;
  int nir_here;
  flouble phi_low,phi_hi,shift;

  conservative=1;//Do we take intersected pixels whose
                 //centers do not fall within range?
  take_all=0;//Take all pixels in ring?
  to_top=0;
  npix=(12*nside)*nside;
  ncap=2*nside*(nside-1); //#pixels in north cap
  nir_here=*nir;
  *nir=0;

  phi_low=phi0-dphi-(int)((phi0-dphi)/(2*M_PI))*2*M_PI;
  phi_hi=phi0+dphi-(int)((phi0+dphi)/(2*M_PI))*2*M_PI;
  if(fabs(dphi-M_PI)<1E-6) take_all=1;

  //Identifies ring number
  if((iz>=nside)&&(iz<=3*nside)) {//equatorial
    ir=iz-nside+1;
    ipix1=ncap+4*nside*(ir-1); //Lowest pixel number
    ipix2=ipix1+4*nside-1; //Highest pixel number
    kshift=modu_he(ir,2);
    nr=4*nside;
  }
  else {
    if(iz<nside) {//North pole
      ir=iz;
      ipix1=2*ir*(ir-1);
      ipix2=ipix1+4*ir-1;
    }
    else {//South pole
      ir=4*nside-iz;
      ipix1=npix-2*ir*(ir+1);
      ipix2=ipix1+4*ir-1;
    }
    nr=4*ir;
    kshift=1;
  }

  //Constructs the pixel list
  if(take_all) {
    *nir=ipix2-ipix1+1;
    if(*nir>nir_here)
      report_error(NMT_ERROR_MEMORY,"Not enough memory in listir\n");
    for(jj=0;jj<(*nir);jj++)
      listir[jj]=ipix1+jj;

    return;
  }

  shift=0.5*kshift;
  if(conservative) {
    ip_low=nint_he(nr*phi_low/(2*M_PI)-shift);
    ip_hi=nint_he(nr*phi_hi/(2*M_PI)-shift);
    ip_low=modu_he(ip_low,nr);
    ip_hi=modu_he(ip_hi,nr);
  }
  else {
    ip_low=(int)(nr*phi_low/(2*M_PI)-shift)+1;
    ip_hi=(int)(nr*phi_hi/(2*M_PI)-shift);
    diff=modu_he((ip_low-ip_hi),nr);
    if(diff<0) diff+=nr;
    if((diff==1)&&(dphi*nr<M_PI)) {
      *nir=0;
      return;
    }
    if(ip_low>=nr) ip_low-=nr;
    if(ip_hi<0) ip_hi+=nr;
  }

  if(ip_low>ip_hi) to_top=1;
  ip_low+=ipix1;
  ip_hi+=ipix1;

  if(to_top) {
    nir1=ipix2-ip_low+1;
    nir2=ip_hi-ipix1+1;
    (*nir)=nir1+nir2;

    if(*nir>nir_here)
      report_error(NMT_ERROR_MEMORY,"Not enough memory in listir\n");
    for(jj=0;jj<nir1;jj++)
      listir[jj]=ip_low+jj;
    for(jj=nir1;jj<(*nir);jj++)
      listir[jj]=ipix1+jj-nir1;
  }
  else {
    (*nir)=ip_hi-ip_low+1;

    if(*nir>nir_here)
      report_error(NMT_ERROR_MEMORY,"Not enough memory in listir\n");
    for(jj=0;jj<(*nir);jj++)
      listir[jj]=ip_low+jj;
  }

  return;
}

static double wrap_phi(double phi)
{
  if(phi>2*M_PI)
    return wrap_phi(phi-2*M_PI);
  else if(phi<0)
    return wrap_phi(phi+2*M_PI);
  else
    return phi;
}

void he_query_disc(int nside,double cth0,double phi,flouble radius,
		   int *listtot,int *nlist,int inclusive)
{
  double phi0;
  int irmin,irmax,iz,nir,ilist;
  flouble radius_eff,a,b,c,cosang;
  flouble dth1,dth2,cosphi0,cosdphi,dphi;
  flouble rlat0,rlat1,rlat2,zmin,zmax,z;
  int *listir;

  phi0=wrap_phi(phi);
  listir=&(listtot[*nlist]);

  if((radius<0)||(radius>M_PI))
    report_error(NMT_ERROR_HPX,"The angular radius is in RADIAN, and should lie in [0,M_PI]!");

  dth1=1/(3*((flouble)(nside*nside)));
  dth2=2/(3*((flouble)nside));

  if(inclusive)
    radius_eff=radius+1.071*M_PI/(4*((flouble)nside)); //TODO:check this
  else
    radius_eff=radius;
  cosang=cos(radius_eff);

  //Circle center
  cosphi0=cos(phi0);
  a=1-cth0*cth0;

  //Coord z of highest and lowest points in the disc
  rlat0=asin(cth0); //TODO:check
  rlat1=rlat0+radius_eff;
  rlat2=rlat0-radius_eff;

  if(rlat1>=0.5*M_PI)
    zmax=1;
  else
    zmax=sin(rlat1);
  irmin=he_ring_num(nside,zmax);
  if(irmin<2)
    irmin=1;
  else
    irmin--;

  if(rlat2<=-0.5*M_PI)
    zmin=-1;
  else
    zmin=sin(rlat2);
  irmax=he_ring_num(nside,zmin);
  if(irmax>(4*nside-2))
    irmax=4*nside-1;
  else
    irmax++;

  ilist=0;
  //Loop on ring number
  for(iz=irmin;iz<=irmax;iz++) {
    int kk;

    if(iz<=nside-1) //North polar cap
      z=1-iz*iz*dth1;
    else if(iz<=3*nside) //Tropical band + equator
      z=(2*nside-iz)*dth2;
    else
      z=-1+dth1*(4*nside-iz)*(4*nside-iz);

    //phi range in the disc for each z
    b=cosang-z*cth0;
    c=1-z*z;
    if(cth0==1) {
      dphi=M_PI;
      if(b>0) continue; //Out of the disc
    }
    else {
      cosdphi=b/sqrt(a*c);
      if(fabs(cosdphi)<=1)
	dphi=acos(cosdphi);
      else {
	if(cosphi0<cosdphi) continue; //Out of the disc
	dphi=M_PI;
      }
    }

    //Find pixels in the disc
    nir=*nlist;
    he_in_ring(nside,iz,phi0,dphi,listir,&nir);

    if(*nlist<ilist+nir) {
      report_error(NMT_ERROR_MEMORY,"Not enough memory in listtot %d %d %lf %lf %lf %d\n",
		   *nlist,ilist+nir,radius,cth0,phi,nside);
    }
    for(kk=0;kk<nir;kk++) {
      listtot[ilist]=listir[kk];
      ilist++;
    }
  }

  *nlist=ilist;
}


#define MAX_SHT 32
long he_nalms(int lmax)
{
  return ((lmax+1)*(lmax+2))/2;
}

long he_indexlm(int l,int m,int lmax)
{
  if(m>0)
    return l+m*lmax-(m*(m-1))/2;
  else
    return l;
}

static void sht_wrapper(int spin,int lmax,int nside,int ntrans,flouble **maps,fcomplex **alms,int alm2map)
{
  double time=0;
  sharp_alm_info *alm_info;
  sharp_geom_info *geom_info;
#ifdef _SPREC
  int flags=0;
#else //_SPREC
  int flags=SHARP_DP;
#endif //_SPREC

  sharp_make_triangular_alm_info(lmax,lmax,1,&alm_info);
  sharp_make_weighted_healpix_geom_info(nside,1,NULL,&geom_info);
#ifdef _NEW_SHARP
  sharp_execute(alm2map,spin,0,alm,map,geom_info,alm_info,ntrans,flags,0,&time,NULL);
#else //_NEW_SHARP
  sharp_execute(alm2map,spin,alms,maps,geom_info,alm_info,ntrans,flags,&time,NULL);
#endif //_NEW_SHARP
  sharp_destroy_geom_info(geom_info);
  sharp_destroy_alm_info(alm_info);
}

void he_alm2map(int nside,int lmax,int ntrans,int spin,flouble **maps,fcomplex **alms)
{
  int nbatches,nodd,itrans,nmaps=1;
  if(spin)
    nmaps=2;
  nbatches=ntrans/MAX_SHT;
  nodd=ntrans%MAX_SHT;

  for(itrans=0;itrans<nbatches;itrans++) {
    sht_wrapper(spin,lmax,nside,MAX_SHT,&(maps[itrans*nmaps*MAX_SHT]),
		&(alms[itrans*nmaps*MAX_SHT]),SHARP_ALM2MAP);
  }
  if(nodd>0) {
    sht_wrapper(spin,lmax,nside,nodd,&(maps[nbatches*nmaps*MAX_SHT]),
		&(alms[nbatches*nmaps*MAX_SHT]),SHARP_ALM2MAP);
  }
}

void he_map2alm(int nside,int lmax,int ntrans,int spin,flouble **maps,fcomplex **alms,int niter)
{
  int nbatches,nodd,itrans,nmaps=1;
  if(spin)
    nmaps=2;
  nbatches=ntrans/MAX_SHT;
  nodd=ntrans%MAX_SHT;

  for(itrans=0;itrans<nbatches;itrans++) {
    sht_wrapper(spin,lmax,nside,MAX_SHT,&(maps[itrans*nmaps*MAX_SHT]),
		&(alms[itrans*nmaps*MAX_SHT]),SHARP_MAP2ALM);
  }
  if(nodd>0) {
    sht_wrapper(spin,lmax,nside,nodd,&(maps[nbatches*nmaps*MAX_SHT]),
		&(alms[nbatches*nmaps*MAX_SHT]),SHARP_MAP2ALM);
  }

  if(niter) {
    int ii,iter;
    int npix=12*nside*nside;
    int nalm=he_nalms(lmax);
    flouble **maps_2=my_malloc(ntrans*nmaps*sizeof(flouble *));
    fcomplex **alms_2=my_malloc(ntrans*nmaps*sizeof(complex *));

    for(ii=0;ii<ntrans*nmaps;ii++) {
      maps_2[ii]=my_malloc(npix*sizeof(flouble));
      alms_2[ii]=my_malloc(nalm*sizeof(fcomplex));
    }

    for(iter=0;iter<niter;iter++) {
      //Get new map
      for(itrans=0;itrans<nbatches;itrans++) {
	sht_wrapper(spin,lmax,nside,MAX_SHT,&(maps_2[itrans*nmaps*MAX_SHT]),
		    &(alms[itrans*nmaps*MAX_SHT]),SHARP_ALM2MAP);
      }
      if(nodd>0) {
	sht_wrapper(spin,lmax,nside,nodd,&(maps_2[nbatches*nmaps*MAX_SHT]),
		    &(alms[nbatches*nmaps*MAX_SHT]),SHARP_ALM2MAP);
      }

      //Subtract from original map
      for(ii=0;ii<ntrans*nmaps;ii++) {
	int ip;
	for(ip=0;ip<npix;ip++)
	  maps_2[ii][ip]=maps[ii][ip]-maps_2[ii][ip];
      }

      //Get alms of difference
      for(itrans=0;itrans<nbatches;itrans++) {
	sht_wrapper(spin,lmax,nside,MAX_SHT,&(maps_2[itrans*nmaps*MAX_SHT]),
		    &(alms_2[itrans*nmaps*MAX_SHT]),SHARP_MAP2ALM);
      }
      if(nodd>0) {
	sht_wrapper(spin,lmax,nside,nodd,&(maps_2[nbatches*nmaps*MAX_SHT]),
		    &(alms_2[nbatches*nmaps*MAX_SHT]),SHARP_MAP2ALM);
      }

      //Add to original alm
      for(ii=0;ii<ntrans*nmaps;ii++) {
	int ilm;
	for(ilm=0;ilm<nalm;ilm++)
	  alms[ii][ilm]+=alms_2[ii][ilm];
      }
    }

    for(ii=0;ii<ntrans*nmaps;ii++) {
      free(maps_2[ii]);
      free(alms_2[ii]);
    }
    free(maps_2);
    free(alms_2);
  }
}

void he_alm2cl(fcomplex **alms_1,fcomplex **alms_2,
	       int pol_1,int pol_2,
	       flouble **cls,int lmax)
{
  int i1,index_cl;
  int nmaps_1=1,nmaps_2=1;
  if(pol_1) nmaps_1=2;
  if(pol_2) nmaps_2=2;

  index_cl=0;
  for(i1=0;i1<nmaps_1;i1++) {
    int i2;
    fcomplex *alm1=alms_1[i1];
    for(i2=0;i2<nmaps_2;i2++) {
      int l;
      fcomplex *alm2=alms_2[i2];
      for(l=0;l<=lmax;l++) {
	int m;
	cls[index_cl][l]=creal(alm1[he_indexlm(l,0,lmax)])*creal(alm2[he_indexlm(l,0,lmax)]);

	for(m=1;m<=l;m++) {
	  long index_lm=he_indexlm(l,m,lmax);
	  cls[index_cl][l]+=2*(creal(alm1[index_lm])*creal(alm2[index_lm])+
			       cimag(alm1[index_lm])*cimag(alm2[index_lm]));
	}
	cls[index_cl][l]/=(2*l+1.);
      }
      index_cl++;
    }
  }
}

void he_anafast(flouble **maps_1,flouble **maps_2,
		int pol_1,int pol_2,flouble **cls,
		int nside,int lmax,int iter)
{
  fcomplex **alms_1,**alms_2;
  int i1,lmax_here=NMT_MAX(lmax,3*nside-1);
  int nmaps_1=1, nmaps_2=1;
  if(pol_1) nmaps_1=2;
  if(pol_2) nmaps_2=2;

  alms_1=my_malloc(nmaps_1*sizeof(fcomplex *));
  for(i1=0;i1<nmaps_1;i1++)
    alms_1[i1]=my_malloc(he_nalms(lmax_here)*sizeof(fcomplex));
  he_map2alm(nside,lmax,1,2*pol_1,maps_1,alms_1,iter);

  if(maps_1==maps_2)
    alms_2=alms_1;
  else {
    alms_2=my_malloc(nmaps_2*sizeof(fcomplex *));
    for(i1=0;i1<nmaps_2;i1++)
      alms_2[i1]=my_malloc(he_nalms(lmax_here)*sizeof(fcomplex));
    he_map2alm(nside,lmax,1,2*pol_2,maps_2,alms_2,iter);
  }

  he_alm2cl(alms_1,alms_2,pol_1,pol_2,cls,lmax);

  for(i1=0;i1<nmaps_1;i1++)
    free(alms_1[i1]);
  free(alms_1);
  if(alms_1!=alms_2) {
    for(i1=0;i1<nmaps_2;i1++)
      free(alms_2[i1]);
    free(alms_2);
  }
}

//Transforms FWHM in arcmin to sigma_G in rad:
//         pi/(60*180*sqrt(8*log(2))
#define FWHM2SIGMA 0.00012352884853326381
flouble *he_generate_beam_window(int lmax,double fwhm_amin)
{
  long l;
  double sigma=FWHM2SIGMA*fwhm_amin;
  flouble *beam=my_malloc((lmax+1)*sizeof(flouble));

  for(l=0;l<=lmax;l++)
    beam[l]=exp(-0.5*l*(l+1)*sigma*sigma);

  return beam;
}

void he_zero_alm(int lmax,fcomplex *alm)
{
#pragma omp parallel default(none) \
  shared(lmax,alm)
  {
    long lm,nalm=he_nalms(lmax);
#pragma omp for
    for(lm=0;lm<nalm;lm++) {
      alm[lm]=0;
    } //end omp for
  } //end omp parallel
}

void he_alter_alm(int lmax,double fwhm_amin,fcomplex *alm_in,fcomplex *alm_out,
		  flouble *window,int add_to_out)
{
  flouble *beam;
  int mm;

  if(window==NULL) beam=he_generate_beam_window(lmax,fwhm_amin);
  else beam=window;

  for(mm=0;mm<=lmax;mm++) {
    int ll;
    for(ll=mm;ll<=lmax;ll++) {
      long index=he_indexlm(ll,mm,lmax);
      if(add_to_out)
	alm_out[index]+=alm_in[index]*beam[ll];
      else
	alm_out[index]=alm_in[index]*beam[ll];
    }
  }

  if(window==NULL)
    free(beam);
}

void he_map_product(int nside,flouble *mp1,flouble *mp2,flouble *mp_out)
{
#pragma omp parallel default(none)		\
  shared(nside,mp1,mp2,mp_out)
  {
    long ip;
    long npix=12*nside*nside;

#pragma omp for
    for(ip=0;ip<npix;ip++) {
      mp_out[ip]=mp1[ip]*mp2[ip];
    } //end omp for
  } //end omp parallel
}

flouble he_map_dot(int nside,flouble *mp1,flouble *mp2)
{
  double sum=0;
  long npix=12*nside*nside;
  double pixsize=4*M_PI/npix;

#pragma omp parallel default(none)		\
  shared(nside,mp1,mp2,sum,npix)
  {
    long ip;
    double sum_thr=0;

#pragma omp for
    for(ip=0;ip<npix;ip++) {
      sum_thr+=mp1[ip]*mp2[ip];
    } //end omp for

#pragma omp critical
    {
      sum+=sum_thr;
    } //end omp critical
  } //end omp parallel

  return (flouble)(sum*pixsize);
}

fcomplex **he_synalm(int nside,int nmaps,int lmax,flouble **cells,flouble **beam,int seed)
{
  int imap;
  fcomplex **alms;
  int lmax_here=3*nside-1;
  alms=my_malloc(nmaps*sizeof(fcomplex *));
  for(imap=0;imap<nmaps;imap++)
    alms[imap]=my_malloc(he_nalms(lmax_here)*sizeof(fcomplex));
  if(lmax>lmax_here) lmax=lmax_here;

  //Switch off error handler for Cholesky decomposition
  gsl_error_handler_t *geh=gsl_set_error_handler_off();

  int numthr=0;

#pragma omp parallel default(none)					\
  shared(nside,nmaps,lmax,cells,beam,seed,alms,lmax_here,numthr)
  {
    //This is to avoid using the omp.h library
    int ithr;
#pragma omp critical
    {
      ithr=numthr;
      numthr++;
    }

    int ll;
    flouble *bms=my_malloc(nmaps*sizeof(flouble));
    gsl_vector *rv1  =gsl_vector_alloc(nmaps);
    gsl_vector *iv1  =gsl_vector_alloc(nmaps);
    gsl_vector *rv2  =gsl_vector_alloc(nmaps);
    gsl_vector *iv2  =gsl_vector_alloc(nmaps);
    gsl_matrix *clmat=gsl_matrix_alloc(nmaps,nmaps);
    gsl_vector *eval =gsl_vector_alloc(nmaps);
    gsl_matrix *evec =gsl_matrix_alloc(nmaps,nmaps);
    gsl_eigen_symmv_workspace *wsym=gsl_eigen_symmv_alloc(nmaps);
    //int ithr=omp_get_thread_num();
    unsigned int seed_thr=(unsigned int)(seed+ithr);
    gsl_rng *rng=init_rng(seed_thr);

#pragma omp for
    for(ll=0;ll<=lmax_here;ll++) {
      int mm,imp1,imp2;

      if(ll>lmax) {
	for(imp1=0;imp1<nmaps;imp1++) {
	  for(mm=0;mm<=ll;mm++) {
	    long index=he_indexlm(ll,mm,lmax_here);
	    alms[imp1][index]=0;
	  }
	}
      }
      else {
	//Get power spectrum
	int icl=0;
	for(imp1=0;imp1<nmaps;imp1++) {
	  for(imp2=imp1;imp2<nmaps;imp2++) {//Fill up only lower triangular part
	    gsl_matrix_set(clmat,imp1,imp2,cells[icl][ll]*0.5);
	    if(imp2!=imp1)
	      gsl_matrix_set(clmat,imp2,imp1,cells[icl][ll]*0.5);
	    icl++;
	  }
	  bms[imp1]=beam[imp1][ll];
	}

	//Take square root
	gsl_eigen_symmv(clmat,eval,evec,wsym);
	for(imp1=0;imp1<nmaps;imp1++) {
	  for(imp2=0;imp2<nmaps;imp2++) {
	    double oij=gsl_matrix_get(evec,imp1,imp2);
	    double lambda=gsl_vector_get(eval,imp2);
	    if(lambda<=0) lambda=0;
	    else lambda=sqrt(lambda);
	    gsl_matrix_set(clmat,imp1,imp2,oij*lambda);
	  }
	}

	//Random alm for m=0
	for(imp1=0;imp1<nmaps;imp1++) {
	  double dr,di;
	  rng_gauss(rng,&dr,&di);
	  gsl_vector_set(rv1,imp1,dr);
	}
	gsl_blas_dgemv(CblasNoTrans,1.,clmat,rv1,0,rv2);
	for(imp1=0;imp1<nmaps;imp1++)
	  alms[imp1][he_indexlm(ll,0,lmax_here)]=bms[imp1]*((fcomplex)(M_SQRT2*gsl_vector_get(rv2,imp1)));

	//Random alms for m>0
	for(mm=1;mm<=ll;mm++) {
	  long index=he_indexlm(ll,mm,lmax_here);
	  for(imp1=0;imp1<nmaps;imp1++) {
	    double dr,di;
	    rng_gauss(rng,&dr,&di);
	    gsl_vector_set(rv1,imp1,dr);
	    gsl_vector_set(iv1,imp1,di);
	  }
	  gsl_blas_dgemv(CblasNoTrans,1.,clmat,rv1,0,rv2);
	  gsl_blas_dgemv(CblasNoTrans,1.,clmat,iv1,0,iv2);
	  for(imp1=0;imp1<nmaps;imp1++)
	    alms[imp1][index]=bms[imp1]*((fcomplex)(gsl_vector_get(rv2,imp1)+I*gsl_vector_get(iv2,imp1)));
	}
      }
    } //end omp for
    free(bms);
    gsl_vector_free(rv1);
    gsl_vector_free(rv2);
    gsl_vector_free(iv1);
    gsl_vector_free(iv2);
    gsl_matrix_free(clmat);
    gsl_vector_free(eval);
    gsl_matrix_free(evec);
    gsl_eigen_symmv_free(wsym);
    end_rng(rng);
  } //end omp parallel

  //Restore error handler
  gsl_set_error_handler(geh);

  return alms;
}

// rectangular helper functions

/* Weights from Waldvogel 2006: BIT Numerical Mathematics 46, p. 195 */
void sharp_make_cc_geom_info_stripe (int nrings, int ppring, double phi0,
  int stride_lon, int stride_lat, sharp_geom_info **geom_info,
  int nsubrings, int i0)
  {
  const double pi=3.141592653589793238462643383279502884197;

  double *theta=RALLOC(double,nrings);
  double *weight=RALLOC(double,nrings);
  int *nph=RALLOC(int,nrings);
  double *phi0_=RALLOC(double,nrings);
  ptrdiff_t *ofs=RALLOC(ptrdiff_t,nrings);
  int *stride_=RALLOC(int,nrings);

  int n=nrings-1;
  SET_ARRAY(weight,0,nrings,0.);
  double dw=-1./(n*n-1.+(n&1));
  weight[0]=2.+dw;
  for (int k=1; k<=(n/2-1); ++k)
    weight[2*k-1]=2./(1.-4.*k*k) + dw;
  weight[2*(n/2)-1]=(n-3.)/(2*(n/2)-1) -1. -dw*((2-(n&1))*n-1);
  real_plan plan = make_real_plan(n);
  real_plan_backward_fftpack(plan,weight);
  kill_real_plan(plan);
  weight[n]=weight[0];

  for (int m=0; m<(nrings+1)/2; ++m)
    {
    theta[m]=pi*m/(nrings-1.);
    if (theta[m]<1e-15) theta[m]=1e-15;
    theta[nrings-1-m]=pi-theta[m];
    nph[m]=nph[nrings-1-m]=ppring;
    phi0_[m]=phi0_[nrings-1-m]=phi0;
    ofs[m]=(ptrdiff_t)m*stride_lat;
    ofs[nrings-1-m]=(ptrdiff_t)((nrings-1-m)*stride_lat);
    stride_[m]=stride_[nrings-1-m]=stride_lon;
    weight[m]=weight[nrings-1-m]=weight[m]*2*pi/(n*nph[m]);
    }

  // remove subrings which are not in the map
  double *subtheta=RALLOC(double,nsubrings);
  double *subweight=RALLOC(double,nsubrings);
  int *subnph=RALLOC(int,nsubrings);
  double *subphi0_=RALLOC(double,nsubrings);
  ptrdiff_t *subofs=RALLOC(ptrdiff_t,nsubrings);
  int *substride_=RALLOC(int,nsubrings);
  for (int m=0; m<nsubrings; ++m)
  {
    // subtheta[m]=theta[nrings-(m+i0)];
    // subweight[m]=weight[nrings-(m+i0)];
    subtheta[m]=theta[(m+i0)];
    subweight[m]=weight[(m+i0)];
    subnph[m]=ppring;
    subphi0_[m]=phi0;
    subofs[m]=(ptrdiff_t)m*stride_lat;
    substride_[m]=stride_lon;
  }
  // printf("%f %f\n", subtheta[0], subtheta[nsubrings-1]);
  sharp_make_geom_info (nsubrings, subnph, subofs, substride_,
    subphi0_, subtheta, subweight, geom_info);

  DEALLOC(theta);
  DEALLOC(weight);
  DEALLOC(nph);
  DEALLOC(phi0_);
  DEALLOC(ofs);
  DEALLOC(stride_);

  DEALLOC(subtheta);
  DEALLOC(subweight);
  DEALLOC(subnph);
  DEALLOC(subphi0_);
  DEALLOC(subofs);
  DEALLOC(substride_);
}


flouble *rect_read_CAR_map(char *fname, nmt_curvedsky_info *sky_info, int nfield)
{
  int status=0, hdutype;
  fitsfile *fptr;

  int nkeys, ii;
  char card[FLEN_CARD], value[FLEN_VALUE];
  char *comment;

  int anynul;
  long *axes;
  int naxes;
  long nelements;
  long fpixel[] = {1,1,1};

  flouble *map, nulval;
  fits_open_file(&fptr, fname, READONLY, &status);
  // fits_get_hdu_num(&fptr, hdutype);
  fits_movabs_hdu(fptr, nfield+1, &hdutype, &status); //change hdu
  fits_get_hdrspace(fptr, &nkeys, NULL, &status);

  for (ii = 1; ii <= nkeys; ii++) {
    fits_read_record(fptr, ii, card, &status);
    // printf("%s, %s\n", card, value);
  }

  fits_get_img_dim(fptr, &naxes, &status);
  axes = (long*) malloc(naxes * sizeof(long));
  fits_get_img_size(fptr, 3, axes, &status);
  // printf("\n\n%i, (%li, %li, %li)\n", naxes, axes[0], axes[1], axes[2]);

  nelements = axes[0] * axes[1] * axes[2];

  double Delta_phi, Delta_theta, phi0, theta0;
  fits_read_key(fptr, TDOUBLE, "CDELT1", &Delta_phi, comment, &status);
  fits_read_key(fptr, TDOUBLE, "CDELT2", &Delta_theta, comment, &status);
  fits_read_key(fptr, TDOUBLE, "CRVAL1", &phi0, comment, &status);
  fits_read_key(fptr, TDOUBLE, "CRVAL2", &theta0, comment, &status);

  sky_info->nx = axes[0];
  sky_info->ny = axes[1];
  sky_info->npix = nelements;
  sky_info->Delta_phi = Delta_phi * M_PI / 180.0;
  sky_info->Delta_theta = Delta_theta * M_PI / 180.0;
  sky_info->phi0 = phi0 * M_PI / 180.0;
  sky_info->theta0 = (90 - theta0) * M_PI / 180.0;
  sky_info->n_eq = fmin(fabs(180.0/Delta_theta), fabs(180.0/Delta_phi)) / 2;

  // printf("nx: %i, ny: %i, npix: %li\n",
  //   sky_info.nx, sky_info.ny, sky_info.npix );
  // printf("Delta_phi: %e, Delta_theta: %e\nphi0: %e, theta0: %e\n",
  //   sky_info.Delta_phi, sky_info.Delta_theta,
  // sky_info.phi0, sky_info.theta0);

  map = (flouble*) malloc(nelements* sizeof(flouble));
  fits_read_pix(fptr, TDOUBLE, fpixel, nelements, 0, map, &anynul, &status);

  if (status)
    fits_report_error(stderr, status);

  fits_close_file(fptr,&status);

  return map;
}


void rect_get_file_params(char *fname,nmt_curvedsky_info *sky_info,int *nfields)
{
  int status=0, hdutype;
  int nfield = 0;
  fitsfile *fptr;

  int nkeys, ii;
  char card[FLEN_CARD], value[FLEN_VALUE];
  char *comment;

  int anynul;
  long *axes;
  int naxes;
  long nelements;
  long fpixel[] = {1,1,1};

  flouble *map, nulval;
  fits_open_file(&fptr, fname, READONLY, &status);
  // fits_get_hdu_num(&fptr, hdutype);
  fits_movabs_hdu(fptr, nfield, &hdutype, &status); //change hdu
  fits_get_hdrspace(fptr, &nkeys, NULL, &status);

  for (ii = 1; ii <= nkeys; ii++) {
    fits_read_record(fptr, ii, card, &status);
  }

  fits_get_img_dim(fptr, &naxes, &status);
  axes = (long*) malloc(naxes * sizeof(long));
  fits_get_img_size(fptr, 3, axes, &status);

  nelements = axes[0] * axes[1] * axes[2];

  double Delta_phi, Delta_theta, phi0, theta0;
  fits_read_key(fptr, TDOUBLE, "CDELT1", &Delta_phi, comment, &status);
  fits_read_key(fptr, TDOUBLE, "CDELT2", &Delta_theta, comment, &status);
  fits_read_key(fptr, TDOUBLE, "CRVAL1", &phi0, comment, &status);
  fits_read_key(fptr, TDOUBLE, "CRVAL2", &theta0, comment, &status);

  sky_info->nx = axes[0];
  sky_info->ny = axes[1];
  sky_info->npix = nelements;
  sky_info->Delta_phi = Delta_phi * M_PI / 180.0;
  sky_info->Delta_theta = Delta_theta * M_PI / 180.0;
  sky_info->phi0 = phi0 * M_PI / 180.0;
  sky_info->theta0 = (90 - theta0) * M_PI / 180.0;
  sky_info->n_eq = fmin(fabs(180.0/Delta_theta), fabs(180.0/Delta_phi)) / 2;

  fits_close_file(fptr,&status);
}


void rect_sht_wrapper(int spin, int lmax, nmt_curvedsky_info *sky_info,
  int ntrans, flouble **maps,fcomplex **alms,int alm2map)
{
  double time=0;
  sharp_alm_info *alm_info;
  sharp_geom_info *geom_info;
#ifdef _SPREC
  int flags=0;
#else //_SPREC
  int flags=SHARP_DP;
#endif //_SPREC

  sharp_make_triangular_alm_info(lmax,lmax,1,&alm_info);

  // first nrings is total
  flouble Delta_theta = fabs(sky_info->Delta_theta);
  int tot_sphere_rings = round(M_PI / Delta_theta) + 1;
  flouble theta1 = sky_info->theta0 - sky_info->ny * Delta_theta;
  int first_ring = round(theta1 / Delta_theta) + 1;
  // printf("total sphere rings: %i\nfirst ring: %i\n",
  //   tot_sphere_rings, first_ring);

  // sharp_make_weighted_healpix_geom_info(nside,1,NULL,&geom_info);
  sharp_make_cc_geom_info_stripe( tot_sphere_rings,
    sky_info->nx, sky_info->phi0, // DEGREES OR RADIANS??
    1, sky_info->nx, //stride_lon ,stride_lat
    &geom_info,
    sky_info->ny, first_ring ); //nsubrings, start_index of ring

#ifdef _NEW_SHARP
  sharp_execute(alm2map,spin,0,alm,map,geom_info,alm_info,ntrans,flags,0,
    &time,NULL);
#else //_NEW_SHARP
  sharp_execute(alm2map,spin,alms,maps,geom_info,alm_info,ntrans,flags,
    &time,NULL);
#endif //_NEW_SHARP
  sharp_destroy_geom_info(geom_info);
  sharp_destroy_alm_info(alm_info);
}


void rect_alm2map(nmt_curvedsky_info *sky_info,int lmax,int ntrans,
  int spin,flouble **maps,fcomplex **alms)
{
  int nbatches,nodd,itrans,nmaps=1;
  if(spin)
    nmaps=2;
  nbatches=ntrans/MAX_SHT;
  nodd=ntrans%MAX_SHT;

  for(itrans=0;itrans<nbatches;itrans++) {
    rect_sht_wrapper(spin,lmax,sky_info,MAX_SHT,&(maps[itrans*nmaps*MAX_SHT]),
		&(alms[itrans*nmaps*MAX_SHT]),SHARP_ALM2MAP);
  }
  if(nodd>0) {
    rect_sht_wrapper(spin,lmax,sky_info,nodd,&(maps[nbatches*nmaps*MAX_SHT]),
		&(alms[nbatches*nmaps*MAX_SHT]),SHARP_ALM2MAP);
  }
}

flouble rect_get_pix_area(nmt_curvedsky_info *sky_info, int i)
{
  flouble theta1 = (int)(i / sky_info->nx) * sky_info->Delta_theta
    + sky_info->theta0;
  return sin(theta1) * sky_info->Delta_phi * sky_info->Delta_theta;
}

void rect_map2alm(nmt_curvedsky_info *sky_info,int lmax,int ntrans,
  int spin,flouble **maps,fcomplex **alms,int niter)
{
  int nbatches,nodd,itrans,nmaps=1;
  if(spin)
    nmaps=2;
  nbatches=ntrans/MAX_SHT;
  nodd=ntrans%MAX_SHT;

  for(itrans=0;itrans<nbatches;itrans++) {
    rect_sht_wrapper(spin,lmax,sky_info,MAX_SHT,&(maps[itrans*nmaps*MAX_SHT]),
		&(alms[itrans*nmaps*MAX_SHT]),SHARP_MAP2ALM);
  }
  if(nodd>0) {
    rect_sht_wrapper(spin,lmax,sky_info,nodd,&(maps[nbatches*nmaps*MAX_SHT]),
		&(alms[nbatches*nmaps*MAX_SHT]),SHARP_MAP2ALM);
  }

  if(niter) {
    int ii,iter;
    int npix=sky_info->npix;
    int nalm=he_nalms(lmax);
    flouble **maps_2=my_malloc(ntrans*nmaps*sizeof(flouble *));
    fcomplex **alms_2=my_malloc(ntrans*nmaps*sizeof(complex *));

    for(ii=0;ii<ntrans*nmaps;ii++) {
      maps_2[ii]=my_malloc(npix*sizeof(flouble));
      alms_2[ii]=my_malloc(nalm*sizeof(fcomplex));
    }

    for(iter=0;iter<niter;iter++) {
      //Get new map
      for(itrans=0;itrans<nbatches;itrans++) {
	rect_sht_wrapper(spin,lmax,sky_info,MAX_SHT,&(maps_2[itrans*nmaps*MAX_SHT]),
		    &(alms[itrans*nmaps*MAX_SHT]),SHARP_ALM2MAP);
      }
      if(nodd>0) {
	rect_sht_wrapper(spin,lmax,sky_info,nodd,&(maps_2[nbatches*nmaps*MAX_SHT]),
		    &(alms[nbatches*nmaps*MAX_SHT]),SHARP_ALM2MAP);
      }

      //Subtract from original map
      for(ii=0;ii<ntrans*nmaps;ii++) {
	int ip;
	for(ip=0;ip<npix;ip++)
	  maps_2[ii][ip]=maps[ii][ip]-maps_2[ii][ip];
      }

      //Get alms of difference
      for(itrans=0;itrans<nbatches;itrans++) {
	rect_sht_wrapper(spin,lmax,sky_info,MAX_SHT,&(maps_2[itrans*nmaps*MAX_SHT]),
		    &(alms_2[itrans*nmaps*MAX_SHT]),SHARP_MAP2ALM);
      }
      if(nodd>0) {
	rect_sht_wrapper(spin,lmax,sky_info,nodd,&(maps_2[nbatches*nmaps*MAX_SHT]),
		    &(alms_2[nbatches*nmaps*MAX_SHT]),SHARP_MAP2ALM);
      }

      //Add to original alm
      for(ii=0;ii<ntrans*nmaps;ii++) {
	int ilm;
	for(ilm=0;ilm<nalm;ilm++)
	  alms[ii][ilm]+=alms_2[ii][ilm];
      }
    }

    for(ii=0;ii<ntrans*nmaps;ii++) {
      free(maps_2[ii]);
      free(alms_2[ii]);
    }
    free(maps_2);
    free(alms_2);
  }
}


void rect_map_product(nmt_curvedsky_info *sky_info,flouble *mp1,flouble *mp2,flouble *mp_out)
{
#pragma omp parallel default(none)		\
  shared(sky_info,mp1,mp2,mp_out)
  {
    long ip;
    long npix=sky_info->npix;

#pragma omp for
    for(ip=0;ip<npix;ip++) {
      mp_out[ip]=mp1[ip]*mp2[ip];
    } //end omp for
  } //end omp parallel
}

flouble rect_map_dot(nmt_curvedsky_info *sky_info,flouble *mp1,flouble *mp2)
{
  double sum=0;
  long npix=sky_info->npix;
  // double pixsize=4*M_PI/npix;

#pragma omp parallel default(none)		\
  shared(sky_info,mp1,mp2,sum,npix)
  {
    long ip;
    double sum_thr=0;

#pragma omp for
    for(ip=0;ip<npix;ip++) {
      sum_thr+=mp1[ip]*mp2[ip]*rect_get_pix_area(sky_info, ip);
    } //end omp for

#pragma omp critical
    {
      sum+=sum_thr;
    } //end omp critical
  } //end omp parallel

  return (flouble)(sum);
}

void rect_anafast(flouble **maps_1,flouble **maps_2,
		int pol_1,int pol_2,flouble **cls,
		nmt_curvedsky_info *cs,int lmax,int iter)
{
  fcomplex **alms_1,**alms_2;
  int i1,lmax_here=NMT_MAX(lmax,3*cs->n_eq-1);
  int nmaps_1=1, nmaps_2=1;
  if(pol_1) nmaps_1=2;
  if(pol_2) nmaps_2=2;

  alms_1=my_malloc(nmaps_1*sizeof(fcomplex *));
  for(i1=0;i1<nmaps_1;i1++)
    alms_1[i1]=my_malloc(he_nalms(lmax_here)*sizeof(fcomplex));
  rect_map2alm(cs,lmax,1,2*pol_1,maps_1,alms_1,iter);

  if(maps_1==maps_2)
    alms_2=alms_1;
  else {
    alms_2=my_malloc(nmaps_2*sizeof(fcomplex *));
    for(i1=0;i1<nmaps_2;i1++)
      alms_2[i1]=my_malloc(he_nalms(lmax_here)*sizeof(fcomplex));
    rect_map2alm(cs,lmax,1,2*pol_2,maps_2,alms_2,iter);
  }

  he_alm2cl(alms_1,alms_2,pol_1,pol_2,cls,lmax);

  for(i1=0;i1<nmaps_1;i1++)
    free(alms_1[i1]);
  free(alms_1);
  if(alms_1!=alms_2) {
    for(i1=0;i1<nmaps_2;i1++)
      free(alms_2[i1]);
    free(alms_2);
  }
}
