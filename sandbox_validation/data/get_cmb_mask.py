from __future__ import print_function
from optparse import OptionParser
import numpy as np
import matplotlib.pyplot as plt
import os
import healpy as hp

def opt_callback(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))
parser = OptionParser()
parser.add_option('--plot', dest='plot_stuff', default=False, action='store_true',
                  help='Set if you want to produce plots')
parser.add_option('--nside', dest='nside_out', default=256, type=int,
                  help='Resolution parameter')
(o, args) = parser.parse_args()

DTOR=np.pi/180

def getmaskapoana(ns,aps,fsk=0.1,dec0=-50,ra0=0.) :
    #This generates a correctly-apodized mask
    v0=np.array([np.sin(DTOR*(90-dec0))*np.cos(DTOR*ra0),
                 np.sin(DTOR*(90-dec0))*np.sin(DTOR*ra0),
                 np.cos(DTOR*(90-dec0))])
    vv=np.array(hp.pix2vec(ns,np.arange(hp.nside2npix(ns))))
    cth=np.sum(v0[:,None]*vv,axis=0); th=np.arccos(cth); th0=np.arccos(1-2*fsk); th_apo=aps*DTOR
    id0=np.where(th>=th0)[0]
    id1=np.where(th<=th0-th_apo)[0]
    idb=np.where((th>th0-th_apo) & (th<th0))[0]
    x=np.sqrt((1-np.cos(th[idb]-th0))/(1-np.cos(th_apo)))
    mask_apo=np.zeros(hp.nside2npix(ns))
    mask_apo[id0]=0.
    mask_apo[id1]=1.
    mask_apo[idb]=x-np.sin(2*np.pi*x)/(2*np.pi)
    return mask_apo

if not os.path.isfile('mask_cmb_ns%d.fits'%o.nside_out) :
    msk=getmaskapoana(o.nside_out,10.,0.1,ra0=250.)
    hp.write_map("mask_cmb_ns%d.fits"%o.nside_out,msk,overwrite=True)

if o.plot_stuff :
    msk=hp.read_map("mask_cmb_ns%d.fits"%o.nside_out,verbose=False)
    hp.mollview(msk);
    plt.show()

