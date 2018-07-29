from __future__ import print_function
from optparse import OptionParser
import numpy as np
import matplotlib.pyplot as plt
import os
import healpy as hp
from scipy.special import jv
import pymaster as nmt
from astropy.wcs import WCS
import flatmaps as fm

def opt_callback(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))
parser = OptionParser()
parser.add_option('--plot', dest='plot_stuff', default=False, action='store_true',
                  help='Set if you want to produce plots')
(o, args) = parser.parse_args()

DTOR=np.pi/180

res_pix=2.*DTOR/60 #Pixels of about 2 arcmin
taper_width=10. #20-deg edges
area_rad=500.*DTOR*DTOR #Total area of ~500 sq-deg
d_extra=1.2*taper_width*DTOR
Ly=np.sqrt(area_rad)/1.25+d_extra
Lx=np.sqrt(area_rad)*1.25+d_extra
Nx=int(Lx/res_pix)
Ny=int(Ly/res_pix)
ioff=int(d_extra*0.5/res_pix)

def plot_map(mp) :
    plt.figure(); plt.imshow(mp,origin='lower',interpolation='nearest')

def get_ellmod() :
    ell1dx=np.fft.fftfreq(Nx)*2*np.pi*Nx/Lx
    ell1dy=np.fft.fftfreq(Ny)*2*np.pi*Ny/Ly
    ell2dx,ell2dy=np.meshgrid(ell1dx,ell1dy)
    return np.sqrt(ell2dy**2+ell2dx**2)

def smooth_map(mp,smooth_scale) :
    ellmod=get_ellmod()
    beam_taper=2*jv(1,ellmod*smooth_scale*DTOR)/(ellmod*smooth_scale*DTOR); beam_taper[0,0]=1;
    mp=np.real(np.fft.ifft2(np.fft.fft2(mp+1j*0)*beam_taper)).flatten();
    mp[mp<1E-3]=0;
    mp=mp.reshape([Ny,Nx])
    return mp

#Make mask
mask=np.ones([Ny,Nx])
mask[:ioff,:]=0; mask[Ny-ioff:,:]=0;
mask[:,:ioff]=0; mask[:,Nx-ioff:]=0;
mask=smooth_map(mask,taper_width*0.5)
mask_pure=nmt.mask_apodization_flat(mask,Lx,Ly,2.,apotype='C1');

w=WCS(naxis=2)
w.wcs.cdelt=[-Lx/Nx/DTOR,Ly/Ny/DTOR]
w.wcs.crval=[0.,-50.]
w.wcs.ctype=['RA---TAN','DEC--TAN']
w.wcs.crpix=[Nx*0.5,Ny*0.5]
fsk=fm.FlatMapInfo(w,nx=Nx,ny=Ny)
fsk.write_flat_map("mask_cmb_flat.fits",mask.flatten(),descript='Apodized mask')

if o.plot_stuff :
    plt.figure(); plt.imshow(mask_pure,origin='lower',interpolation='nearest');
    plt.show()
