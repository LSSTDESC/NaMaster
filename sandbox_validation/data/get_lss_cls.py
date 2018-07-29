from __future__ import print_function
from optparse import OptionParser
import numpy as np
import matplotlib.pyplot as plt
import pyccl as ccl
import os

def opt_callback(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))
plot_stuff=True
parser = OptionParser()
parser.add_option('--plot', dest='plot_stuff', default=False, action='store_true',
                  help='Set if you want to produce plots')
(o, args) = parser.parse_args()

if not os.path.isfile("cls_lss.txt") :
    z,pz=np.loadtxt("nz.txt",unpack=True)
    ndens=np.sum(pz)*np.mean(z[1:]-z[:-1])*(180*60./np.pi)**2

    bz=np.ones_like(z)
    cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2.1e-9, n_s=0.96)
    clust=ccl.ClTracerNumberCounts(cosmo,False,False,z=z,n=pz,bias=bz)
    lens=ccl.ClTracerLensing(cosmo,False,z=z,n=pz)
    ell=np.arange(30000)
    cltt=ccl.angular_cl(cosmo,clust,clust,ell)
    clte=ccl.angular_cl(cosmo,clust,lens,ell)
    clee=ccl.angular_cl(cosmo,lens,lens,ell)
    clbb=np.zeros_like(ell)
    nltt=np.ones_like(ell)/ndens
    nlte=np.zeros_like(ell)
    nlee=0.28**2*np.ones_like(ell)/ndens
    nlbb=nlee.copy()
    np.savetxt("cls_lss.txt",np.transpose([ell,cltt,clee,clbb,clte,nltt,nlee,nlbb,nlte]))
ell,cltt,clee,clbb,clte,nltt,nlee,nlbb,nlte=np.loadtxt("cls_lss.txt",unpack=True)

if o.plot_stuff :
    plt.figure()
    plt.plot(ell,cltt,'r-',label='$\\delta_g-\\delta_g$')
    plt.plot(ell,clte,'g-',label='$\\delta_g-\\gamma_E$')
    plt.plot(ell,clee,'b-',label='$\\gamma_E-\\gamma_E$')
    plt.plot(ell,nltt,'r--',label='$N_{gg}$')
    plt.plot(ell,nlee,'g--',label='$N_{EE}$')
    plt.plot(ell,nlbb,'y--',label='$N_{BB}$')
    plt.loglog()
    plt.xlabel('$\\ell$',fontsize=16)
    plt.ylabel('$C_\ell$',fontsize=16)
    plt.legend(loc='lower left',frameon=False,fontsize=16,labelspacing=0.1,ncol=2)
    plt.show()
