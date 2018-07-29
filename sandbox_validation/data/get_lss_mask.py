from __future__ import print_function
from optparse import OptionParser
import numpy as np
import matplotlib.pyplot as plt
import os
import healpy as hp

DTOR=np.pi/180

def opt_callback(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))
parser = OptionParser()
parser.add_option('--plot', dest='plot_stuff', default=False, action='store_true',
                  help='Set if you want to produce plots')
parser.add_option('--nside', dest='nside_out', default=512, type=int,
                  help='Resolution parameter')
parser.add_option('--nholes', dest='nholes', default=100, type=int,
                  help='Number of holes to cut out')
parser.add_option('--rholes', dest='rholes', default=1., type=float,
                  help='Hole radius in degrees')
(o, args) = parser.parse_args()

if not os.path.isfile("mask_lss_ns%d.fits"%o.nside_out) :
    mask0=hp.ud_grade(hp.read_map("mask_lss.fits",verbose=False),nside_out=o.nside_out)
    dec0=0; decf=-57; cth0=np.cos(DTOR*(90-decf)); cthf=np.cos(DTOR*(90-dec0));
    th_r_c=np.arccos(cth0+(cthf-cth0)*np.random.rand(o.nholes))
    phi_r_c=2*np.pi*np.random.rand(o.nholes)
    r=hp.Rotator(coord=['C','G'])
    th_r_g,phi_r_g=r(th_r_c,phi_r_c)
    vs=np.transpose(np.array([np.sin(th_r_g)*np.cos(phi_r_g),
                              np.sin(th_r_g)*np.sin(phi_r_g),
                              np.cos(th_r_g)]))
    mskb=np.ones_like(mask0);
    for v in vs :
        mskb[hp.query_disc(o.nside_out,v,o.rholes*np.pi/180)]=0

    hp.write_map("mask_lss_ns%d.fits"%o.nside_out,mskb*mask0,overwrite=True)

if o.plot_stuff :
    mask=hp.read_map("mask_lss_ns%d.fits"%o.nside_out,verbose=False)
    hp.mollview(mask,coord=['G','C']);
    plt.show();
