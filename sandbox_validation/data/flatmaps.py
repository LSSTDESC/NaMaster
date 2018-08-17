import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pymaster as nmt
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import interp1d

class FlatMapInfo(object) :
    def __init__(self,wcs,nx=None,ny=None,lx=None,ly=None) :
        """
        Creates a flat map
        wcs : WCS object containing information about reference point and resolution
        nx,ny : Number of pixels in the x/y axes. If None, dx/dy must be provided
        lx,ly : Extent of themap in the x/y axes. If None, nx/ny must be provided
        """
        self.wcs=wcs.copy()

        if nx is None and lx is None :
            raise ValueError("Must provide either nx or lx")

        if ny is None and ly is None :
            raise ValueError("Must provide either ny or ly")

        if nx is None :
            self.lx=lx
            self.nx=int(self.lx/np.abs(self.wcs.wcs.cdelt[0]))+1
        else :
            self.nx=nx
            self.lx=np.fabs(nx*self.wcs.wcs.cdelt[0])
        self.dx=self.lx/self.nx

        if ny is None :
            self.ly=ly
            self.ny=int(self.ly/np.abs(self.wcs.wcs.cdelt[1]))+1
        else :
            self.ny=ny
            self.ly=np.fabs(ny*self.wcs.wcs.cdelt[1])
        self.dy=self.ly/self.ny

        self.lx_rad=self.lx*np.pi/180
        self.ly_rad=self.ly*np.pi/180

        self.npix=self.nx*self.ny

    def get_dims(self) :
        """
        Returns map size
        """
        return [self.ny,self.nx]
        
    def get_size(self) :
        """
        Returns map size
        """
        return self.npix

    def pos2pix(self,ra,dec) :
        """
        Returns pixel indices for arrays of x and y coordinates.
        Will return -1 if (x,y) lies outside the map
        """
        ra=np.asarray(ra)
        scalar_input=False
        if ra.ndim==0 :
            ra=x[None]
            scalar_input=True

        dec=np.asarray(dec)
        if dec.ndim==0 :
            dec=dec[None]

        if len(ra)!=len(dec) :
            raise ValueError("ra and dec must have the same size!")

        ix,iy=np.transpose(self.wcs.wcs_world2pix(np.transpose(np.array([ra,dec])),0))
        ix=ix.astype(int); iy=iy.astype(int);
        ix_out=np.where(np.logical_or(ix<0,ix>=self.nx))[0]
        iy_out=np.where(np.logical_or(iy<0,iy>=self.ny))[0]

        ipix=ix+self.nx*iy
        ipix[ix_out]=-1
        ipix[iy_out]=-1

        if scalar_input :
            return np.squeeze(ipix)
        return ipix

    def pos2pix2d(self,ra,dec) :
        """
        Returns pixel indices for arrays of x and y coordinates.
        """
        ra=np.asarray(ra)
        scalar_input=False
        if ra.ndim==0 :
            ra=x[None]
            scalar_input=True

        dec=np.asarray(dec)
        if dec.ndim==0 :
            dec=dec[None]

        if len(ra)!=len(dec) :
            raise ValueError("ra and dec must have the same size!")

        ix,iy=np.transpose(self.wcs.wcs_world2pix(np.transpose(np.array([ra,dec])),0))
        ix_out=np.where(np.logical_or(ix<-self.nx,ix>=2*self.nx))[0]
        iy_out=np.where(np.logical_or(iy<-self.ny,iy>=2*self.ny))[0]
        
        is_in=np.ones(len(ix),dtype=bool)
        is_in[ix_out]=False
        is_in[iy_out]=False
        is_in[np.isnan(ix)]=False
        is_in[np.isnan(iy)]=False

        if scalar_input :
            return np.squeeze(ix),np.squeeze(iy),np.squeeze(is_in)
        return ix,iy,is_in

    def pix2pos(self,ipix) :
        """
        Returns x,y coordinates of pixel centres for a set of pixel indices.
        """
        ipix=np.asarray(ipix)
        scalar_input=False
        if ipix.ndim==0 :
            ipix=ipix[None]
            scalar_input=True

        i_out=np.where(np.logical_or(ipix<0,ipix>=self.npix))[0]
        if len(i_out)>0 :
            print(ipix[i_out])
            raise ValueError("Pixels outside of range")

        ix=ipix%self.nx
        ioff=np.array(ipix-ix)
        iy=ioff.astype(int)/(int(self.nx))
        ix=ix.astype(np.float_)
        iy=iy.astype(np.float_)

        ra,dec=np.transpose(self.wcs.wcs_pix2world(np.transpose(np.array([ix,iy])),0))

        if scalar_input :
            return np.squeeze(ra),np.squeeze(dec)
        return ra,dec

    def get_empty_map(self) :
        """
        Returns a map full of zeros
        """
        return np.zeros(self.npix,dtype=float)

    def view_map(self,map_in,ax=None, xlabel='RA', ylabel='Dec',fs=12, tfs=12,
		 title=None, addColorbar=True,posColorbar= False, cmap = cm.viridis,
                 colorMax= None, colorMin= None,fnameOut=None):
        """
        Plots a 2D map (passed as a flattened array)
        """
        if len(map_in)!=self.npix :
            raise ValueError("Input map doesn't have the correct size")

	# set up the colorbar if min, max not given.
        #if colorMax is None or colorMin is None:
        #    if posColorbar:
        #        ind= np.where(map_in>0)[0]
        #        colorMin= np.percentile(map_in[ind], 15)
        #        colorMax= np.percentile(map_in[ind], 95)
        #    else:
        #        colorMin= np.percentile(map_in, 15)
        #        colorMax= np.percentile(map_in, 95)

        if ax is None :
            fig=plt.figure()
            ax=fig.add_subplot(111,projection=self.wcs)
        if title is not None :
            ax.set_title(title,fontsize=tfs)
        image= ax.imshow(map_in.reshape([self.ny,self.nx]),vmin=colorMin,vmax=colorMax,
			 origin='lower', interpolation='nearest')
        if addColorbar :
            plt.colorbar(image)
        ax.set_xlabel(xlabel,fontsize=fs)
        ax.set_ylabel(ylabel,fontsize=fs)
        if fnameOut is not None :
            plt.savefig(fnameOut,bbox_inches='tight')

    def write_flat_map(self,filename,maps,descript=None) :
        """
        Saves a set of maps in FITS format wit WCS.
        """

        if maps.ndim<1 :
            raise ValueError("Must supply at least one map")
        if maps.ndim==1 :
            maps=np.array([maps])
        if len(maps[0])!=self.npix :
            raise ValueError("Map doesn't conform to this pixelization")
        if descript is not None :
            if len(maps)==1 :
                descript=[descript]
            if len(maps)!=len(descript) :
                raise ValueError("Need one description per map")

        header=self.wcs.to_header()
        hdus=[]
        for im,m in enumerate(maps) :
            head=header.copy()
            if descript is not None :
                head['DESCR']=(descript[im],'Description')
            if im==0 :
                hdu=fits.PrimaryHDU(data=m.reshape([self.ny,self.nx]),header=head)
            else :
                hdu=fits.ImageHDU(data=m.reshape([self.ny,self.nx]),header=head)
            hdus.append(hdu)
        hdulist=fits.HDUList(hdus)
        hdulist.writeto(filename,overwrite=True)

    def compute_power_spectrum(self,map1,mask1,map2=None,mask2=None,l_bpw=None,
                               return_bpw=False,wsp=None,return_wsp=False,
                               temp1=None,temp2=None) :
        """
        Computes power spectrum between two maps.
        map1 : first map to correlate
        mask1 : mask for the first map
        map2 : second map to correlate. If None map2==map1.
        mask2 : mask for the second map. If None mask2==mask1.
        l_bpw : bandpowers on which to calculate the power spectrum. Should be an [2,N_ell] array, where
                the first and second columns list the edges of each bandpower. If None, the function will
                create bandpowers of its own taylored to the properties of your map.
        return_bpw : if True, the bandpowers will also be returned
        wsp : NmtWorkspaceFlat object to accelerate the calculation. If None, this will be precomputed.
        return_wsp : if True, the workspace will also be returned 
        temp1 : if not None, set of contaminants to remove from map1
        temp2 : if not None, set of contaminants to remove from map2
        """
        same_map=False
        if map2 is None :
            map2=map1
            same_map=True
        
        same_mask=False
        if mask2 is None :
            mask2=mask1
            same_mask=False

        if len(map1)!=self.npix :
            raise ValueError("Input map has the wrong size")
        if (len(map1)!=len(map2)) or (len(map1)!=len(mask1)) or (len(map1)!=len(mask2)) :
            raise ValueError("Sizes of all maps and masks don't match")
            
        if l_bpw is None :
            ell_min=max(2*np.pi/self.lx_rad,2*np.pi/self.ly_rad)
            ell_max=min(self.nx*np.pi/self.lx_rad,self.ny*np.pi/self.ly_rad)
            d_ell=2*ell_min
            n_ell=int((ell_max-ell_min)/d_ell)-1
            l_bpw=np.zeros([2,n_ell])
            l_bpw[0,:]=ell_min+np.arange(n_ell)*d_ell
            l_bpw[1,:]=l_bpw[0,:]+d_ell
            return_bpw=True
        
        #Generate binning scheme
        b=nmt.NmtBinFlat(l_bpw[0,:],l_bpw[1,:])

        if temp1 is not None :
            tmp1=np.array([[t.reshape([self.ny,self.nx])] for t in temp1])
        else :
            tmp1=None
        if temp2 is not None :
            tmp2=np.array([[t.reshape([self.ny,self.nx])] for t in temp2])
        else :
            tmp2=None

        #Generate fields
        f1=nmt.NmtFieldFlat(self.lx_rad,self.ly_rad,mask1.reshape([self.ny,self.nx]),
                            [map1.reshape([self.ny,self.nx])],templates=tmp1)
        if same_map and same_mask :
            f2=f1
        else :
            f2=nmt.NmtFieldFlat(self.lx_rad,self.ly_rad,mask2.reshape([self.ny,self.nx]),
                                [map2.reshape([self.ny,self.nx])],templates=tmp2)

        #Compute workspace if needed
        if wsp is None :
            wsp=nmt.NmtWorkspaceFlat();
            wsp.compute_coupling_matrix(f1,f2,b)
            return_wsp=True

        #Compute power spectrum
        cl_coupled=nmt.compute_coupled_cell_flat(f1,f2,b)
        cl_uncoupled=wsp.decouple_cell(cl_coupled)[0]

        #Return
        if return_bpw and return_wsp :
            return cl_uncoupled,l_bpw,wsp
        else :
            if return_bpw :
                return cl_uncoupled,l_bpw
            elif return_wsp :
                return cl_uncoupled,wsp
            else :
                return cl_uncoupled

    def u_grade(self,mp,x_fac,y_fac=None) :
        """
        Up-grades the resolution of a map and returns the associated FlatSkyInfo object.
        mp : input map
        x_fac : the new map will be sub-divided into x_fac*nx pixels in the x direction
        y_fac : the new map will be sub-divided into y_fac*ny pixels in the y direction
                if y_fac=None, then y_fac=x_fac
        """
        if y_fac is None :
            y_fac=x_fac
        if len(mp)!=self.npix :
            raise ValueError("Input map has a wrong size")

        w=WCS(naxis=2)
        w.wcs.cdelt=[self.wcs.wcs.cdelt[0]/int(x_fac),self.wcs.wcs.cdelt[1]/int(y_fac)]
        w.wcs.crval=self.wcs.wcs.crval
        w.wcs.ctype=self.wcs.wcs.ctype
        w.wcs.crpix=[self.wcs.wcs.crpix[0]*int(x_fac),self.wcs.wcs.crpix[1]*int(y_fac)]

        fm_ug=FlatMapInfo(w,nx=self.nx*int(x_fac),ny=self.ny*int(y_fac))
        mp_ug=np.repeat(np.repeat(mp.reshape([self.ny,self.nx]),int(y_fac),axis=0),
                        int(x_fac),axis=1).flatten()
        
        return fm_ug,mp_ug

    def d_grade(self,mp,x_fac,y_fac=None) :
        """
        Down-grades the resolution of a map and returns the associated FlatSkyInfo object.
        mp : input map
        x_fac : the new map will be sub-divided into floor(nx/x_fac) pixels in the x direction
        y_fac : the new map will be sub-divided into floor(ny/y_fac) pixels in the y direction
                if y_fac=None, then y_fac=x_fac.
        Note that if nx/ny is not a multiple of x_fac/y_fac, the remainder pixels will be lost.
        """
        if y_fac is None :
            y_fac=x_fac
        if len(mp)!=self.npix :
            raise ValueError("Input map has a wrong size")
        print(x_fac,y_fac)
        print(int(x_fac),int(y_fac))

        w=WCS(naxis=2)
        w.wcs.cdelt=[self.wcs.wcs.cdelt[0]*int(x_fac),self.wcs.wcs.cdelt[1]*int(y_fac)]
        w.wcs.crval=self.wcs.wcs.crval
        w.wcs.ctype=self.wcs.wcs.ctype
        w.wcs.crpix=[self.wcs.wcs.crpix[0]/int(x_fac),self.wcs.wcs.crpix[1]/int(y_fac)]
        
        nx_new=self.nx/int(x_fac); ix_max=nx_new*int(x_fac)
        ny_new=self.ny/int(y_fac); iy_max=ny_new*int(y_fac)
        
        mp2d=mp.reshape([self.ny,self.nx])[:iy_max,:][:,:ix_max]
        fm_dg=FlatMapInfo(w,nx=nx_new,ny=ny_new)
        mp_dg=np.mean(np.mean(np.reshape(mp2d.flatten(),[ny_new,int(y_fac),nx_new,int(x_fac)]),axis=-1),axis=-2).flatten()
        
        return fm_dg,mp_dg

    @classmethod
    def from_coords(FlatMapInfo,ra_arr,dec_arr,reso,pad=None) :
        """
        Generates a FlatMapInfo object that can encompass all points with coordinates
        given by ra_arr (R.A.) and dec_arr (dec.) with pixel resolution reso.
        The parameter pad should correspond to the number of pixel sizes you want
        to leave as padding around the edges of the map. If None, it will default to 20.
        """

        if len(ra_arr.flatten())!=len(dec_arr.flatten()) :
            raise ValueError("ra_arr and dec_arr must have the same size")

        if pad==None :
            pad=20.
        elif pad<0 :
            raise ValueError("We need a positive padding")

        # Find median coordinates
        ramean=0.5*(np.amax(ra_arr)+np.amin(ra_arr))
        decmean=0.5*(np.amax(dec_arr)+np.amin(dec_arr))

        #Compute projection on the tangent plane
        w=WCS(naxis=2)
        w.wcs.crpix=[0,0]
        w.wcs.cdelt=[-reso,reso]
        w.wcs.crval=[ramean,decmean]
        w.wcs.ctype=['RA---TAN','DEC--TAN']
        ix,iy=np.transpose(w.wcs_world2pix(np.transpose(np.array([ra_arr,dec_arr])),0))
        #Estimate map size
        nsidex=int(np.amax(ix))-int(np.amin(ix))+1+2*int(pad)
        nsidey=int(np.amax(iy))-int(np.amin(iy))+1+2*int(pad)
        #Off-set to make sure every pixel has positive coordinates
        # TODO: worry about 2pi wrapping
        offx=-np.amin(ix)+pad
        offy=-np.amin(iy)+pad
        w.wcs.crpix=[offx,offy]

        return FlatMapInfo(w,nx=nsidex,ny=nsidey)

    def smooth_map(self,map_in,larr,beam) :
        k_x=np.fft.rfftfreq(self.nx,self.lx_rad/(2*np.pi*self.nx))
        k_y=np.fft.fftfreq(self.ny,self.ly_rad/(2*np.pi*self.ny))
        k_mod=np.sqrt(k_x[None,:]**2+k_y[:,None]**2)
        beamf=interp1d(larr,beam,kind='linear',bounds_error=False,fill_value=0)
        beam_map=beamf(k_mod)
        return np.fft.irfft2(beam_map*np.fft.rfft2(map_in.reshape([self.ny,self.nx])),s=[self.ny,self.nx]).flatten()
        
    def clbin(self,larr,cls) :
        if cls.ndim==1 :
            scalar_input=True
        else :
            scalar_input=False

        k_x=np.fft.rfftfreq(self.nx,self.lx_rad/(2*np.pi*self.nx))
        k_y=np.fft.fftfreq(self.ny,self.ly_rad/(2*np.pi*self.ny))
        k_mod=np.sqrt(k_x[None,:]**2+k_y[:,None]**2)

        krange=[0,np.amax(k_mod)]
        kbins=max(self.nx,self.ny)/2
        nk,bk=np.histogram(k_mod.flatten(),range=krange,bins=kbins)
        kk,bk=np.histogram(k_mod.flatten(),range=krange,bins=kbins,weights=k_mod.flatten())
        kmean=kk/nk

        def bin_cl(cl) :
            clf=interp1d(larr,cl,kind='linear',bounds_error=False,fill_value=0)
            clmap=clf(k_mod)
            pk,bk=np.histogram(k_mod.flatten(),range=krange,bins=kbins,weights=clmap.flatten())
            return pk/nk
            
        if scalar_input :
            clout=bin_cl(cls)
        else :
            clout=np.array([bin_cl(c) for c in cls])

        return kmean,clout
        
    def anafast(self,mps) :
        if mps.ndim==1 :
            scalar_input=True
        else :
            scalar_input=False

        k_x=np.fft.rfftfreq(self.nx,self.lx_rad/(2*np.pi*self.nx))
        k_y=np.fft.fftfreq(self.ny,self.ly_rad/(2*np.pi*self.ny))
        k_mod=np.sqrt(k_x[None,:]**2+k_y[:,None]**2)
        dkvol=(2*np.pi)**2/(self.lx_rad*self.ly_rad)
        fft_norm=self.lx_rad*self.ly_rad/(2*np.pi*self.nx*self.ny)

        krange=[0,np.amax(k_mod)]
        kbins=max(self.nx,self.ny)/2
        nk,bk=np.histogram(k_mod.flatten(),range=krange,bins=kbins)
        kk,bk=np.histogram(k_mod.flatten(),range=krange,bins=kbins,weights=k_mod.flatten())
        kmean=kk/nk

        def compute_cl_single(alm1,alm2) :
            almabs2=(np.real(alm1)*np.real(alm2)+np.imag(alm1)*np.imag(alm2)).flatten()
            pk,bk=np.histogram(k_mod.flatten(),range=krange,bins=kbins,weights=almabs2)
            return pk/nk
        
        if scalar_input :
            alms=np.fft.rfftn(mps.reshape([self.ny,self.nx]))*fft_norm
            cls=compute_cl_single(alms,alms)
        else :
            alms_tqu=np.array([np.fft.rfftn(m.reshape([self.ny,self.nx]))*fft_norm for m in mps])
            
            k_mod[0,0]=1E-16
            cosk=k_x[None,:]/k_mod; cosk[0,0]=1.;
            sink=k_y[:,None]/k_mod; sink[0,0]=0.;
            k_mod[0,0]=0
            cos2k=cosk**2-sink**2; sin2k=2*sink*cosk
            a_t=alms_tqu[0,:,:]
            a_e=cos2k*alms_tqu[1,:,:]-sin2k*alms_tqu[2,:,:]
            a_b=sin2k*alms_tqu[1,:,:]+cos2k*alms_tqu[2,:,:]
            cls=[]
            cls.append(compute_cl_single(a_t,a_t))
            cls.append(compute_cl_single(a_e,a_e))
            cls.append(compute_cl_single(a_b,a_b))
            cls.append(compute_cl_single(a_t,a_e))
            cls.append(compute_cl_single(a_e,a_b))
            cls.append(compute_cl_single(a_t,a_b))
            cls=np.array(cls)

        return kmean,cls*dkvol

    def synfast(self,larr,cls) :

        if cls.ndim==1 :
            scalar_input=True
        else :
            scalar_input=False

        k_x=np.fft.rfftfreq(self.nx,self.lx_rad/(2*np.pi*self.nx))
        k_y=np.fft.fftfreq(self.ny,self.ly_rad/(2*np.pi*self.ny))
        k_mod=np.sqrt(k_x[None,:]**2+k_y[:,None]**2)
        inv_dkvol=self.lx_rad*self.ly_rad/(2*np.pi)**2
        fft_norm=2*np.pi*self.nx*self.ny/(self.lx_rad*self.ly_rad)

        def synalm_single(clmap) :
            sqclmap=np.sqrt(clmap*inv_dkvol/2.)
        
            alm_r=np.random.randn(self.ny,self.nx/2+1)*sqclmap
            alm_i=np.random.randn(self.ny,self.nx/2+1)*sqclmap
            alm=alm_r+1j*alm_i
            alm[0,0]=alm_r[0,0]*np.sqrt(2.)
            alm[:self.ny/2,0]=alm_r[:self.ny/2,0]+1j*alm_i[:self.ny/2,0]
            alm[self.ny/2:,0]=alm_r[self.ny/2:,0]+1j*alm_i[self.ny/2:,0]
            return alm
         
        if scalar_input :
            clf=interp1d(larr,cls,kind='linear',bounds_error=False,fill_value=0)
            alms=synalm_single(clf(k_mod))
            mps=np.fft.irfft2(alms,s=[self.ny,self.nx])*fft_norm
            return mps.flatten()
        else :
            cls_maps=np.zeros([self.ny,self.nx/2+1,3,3])
            def get_cl_map(cl) :
                clf=interp1d(larr,cl,kind='linear',bounds_error=False,fill_value=0)
                return clf(k_mod)
            for i in np.arange(3) :
                cls_maps[:,:,i,i]=get_cl_map(cls[i])
            cls_maps[:,:,0,1]=get_cl_map(cls[3])
            cls_maps[:,:,1,0]=cls_maps[:,:,0,1]

            cls_w,cls_v=np.linalg.eigh(cls_maps)
            alms_teb=np.transpose(np.array([synalm_single(c)
                                            for c in np.transpose(cls_w,axes=[2,0,1])]),axes=[1,2,0])
            alms_teb=np.transpose(np.sum(cls_v[:,:,:,:]*alms_teb[:,:,None,:],axis=3),axes=[2,0,1])

            k_mod[0,0]=1E-16
            cosk=k_x[None,:]/k_mod; cosk[0,0]=1.;
            sink=k_y[:,None]/k_mod; sink[0,0]=0.;
            k_mod[0,0]=0
            cos2k=cosk**2-sink**2; sin2k=2*sink*cosk
            alms_tqu=np.zeros_like(alms_teb)
            alms_tqu[0,:,:]=alms_teb[0,:,:]
            alms_tqu[1,:,:]= cos2k*alms_teb[1,:,:]+sin2k*alms_teb[2,:,:]
            alms_tqu[2,:,:]=-sin2k*alms_teb[1,:,:]+cos2k*alms_teb[2,:,:]
            mps=np.array([np.fft.irfft2(alm,s=[self.ny,self.nx])*fft_norm for alm in alms_tqu])

            return mps.reshape([3,self.ny*self.nx])

####
def read_flat_map(filename,i_map=0) :
    """
    Reads a flat-sky map and the details of its pixelization scheme.
    The latter are returned as a FlatMapInfo object.
    i_map : map to read. If -1, all maps will be read.
    """
    hdul=fits.open(filename)
    w=WCS(hdul[0].header)

    if i_map==-1 :
        maps=np.array([hdu.data for hdu in hdul])
        nm,ny,nx=maps.shape
        maps=maps.reshape([nm,ny*nx])
    else :
        maps=hdul[i_map].data
        ny,nx=maps.shape
        maps=maps.flatten()

    fmi=FlatMapInfo(w,nx=nx,ny=ny)

    return fmi,maps
