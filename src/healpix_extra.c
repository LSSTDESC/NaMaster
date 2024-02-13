#include "config.h"
#include "utils.h"
#include <fitsio.h>
#include <chealpix.h>

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
//TODO: generalize to CAR;
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

flouble he_get_pix_area(long nside,long iy)
{
  return M_PI/(3*nside*nside);
}

void he_map_product(long nside,flouble *mp1,flouble *mp2,flouble *mp_out)
{
#pragma omp parallel default(none)		\
  shared(nside,mp1,mp2,mp_out)
  {
    long ip;
    long npix=he_nside2npix(nside);

#pragma omp for
    for(ip=0;ip<npix;ip++) {
      mp_out[ip]=mp1[ip]*mp2[ip];
    } //end omp for
  } //end omp parallel
}

flouble he_map_dot(long nside,flouble *mp1,flouble *mp2)
{
  double sum=0;

#pragma omp parallel default(none)		\
  shared(nside,mp1,mp2,sum)
  {
    long ip;
    long npix=he_nside2npix(nside);
    double sum_thr=0;
    double pixsize=he_get_pix_area(nside,0);

#pragma omp for
    for(ip=0;ip<npix;ip++) {
      sum_thr+=mp1[ip]*mp2[ip];
    } //end omp for

#pragma omp critical
    {
      sum+=sum_thr*pixsize;
    } //end omp critical
  } //end omp parallel

  return (flouble)(sum);
}
