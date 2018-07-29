from __future__ import print_function
from optparse import OptionParser
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os

def opt_callback(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))
parser = OptionParser()
parser.add_option('--plot', dest='plot_stuff', default=False, action='store_true',
                  help='Set if you want to produce plots')
parser.add_option('--nside', dest='nside_out', default=512, type=int,
                  help='Resolution parameter')
(o, args) = parser.parse_args()

def read_cl_camb(fname) :
    data=np.loadtxt(fname,unpack=True)
    ll=np.arange(3*o.nside_out,dtype=float)
    fac=2*np.pi/(ll[2:]*(ll[2:]+1.))
    cl_tt=np.zeros_like(ll); cl_tt[2:]=data[1,:3*o.nside_out-2]*fac
    cl_ee=np.zeros_like(ll); cl_ee[2:]=data[2,:3*o.nside_out-2]*fac
    cl_bb=np.zeros_like(ll); cl_bb[2:]=data[3,:3*o.nside_out-2]*fac
    cl_te=np.zeros_like(ll); cl_te[2:]=data[4,:3*o.nside_out-2]*fac

    return ll,cl_tt,cl_ee,cl_bb,cl_te

l,cltt,clee,clbb,clte=read_cl_camb("cls_cmb.txt")
msk=hp.ud_grade(hp.read_map("mask_cmb_ns%d.fits"%o.nside_out,verbose=False),nside_out=o.nside_out)
mz=np.zeros_like(msk)
fsky=np.mean(msk)
larr=np.arange(3*o.nside_out)

if not os.path.isfile('cont_cmb_ns%d.fits'%o.nside_out) :
    if o.nside_out>256 :
        #Generate random homogeneous realization with the right power spectrum.
        ccee=2E-5*(100./(larr+0.1))**2.5; ccee[:10]=ccee[10]
        ccbb=9E-6*(100./(larr+0.1))**2.3; ccbb[:10]=ccbb[10]
        ratio=np.sqrt(0.1*np.mean(clbb[40:60])/np.mean(ccbb[40:60]))
        
        czero=np.zeros_like(ccee)
        t,q,u=hp.synfast([czero,ccee,ccbb,czero],nside=o.nside_out,pol=True,new=True,verbose=False)
    else :
        #Otherwise use the PySM maps
        q,u=hp.read_map("bmode_fg.fits",field=[0,1],verbose=False)
        q=hp.ud_grade(q,nside_out=o.nside_out)
        u=hp.ud_grade(u,nside_out=o.nside_out)
        cctt,ccee,ccbb,ccte,cceb,cctb=np.array(hp.anafast([mz*msk,q*msk,u*msk],pol=True))/fsky
        ratio=np.sqrt(0.1*np.mean(clbb[40:60])/np.mean(ccbb[40:60]))
        
    q*=ratio; u*=ratio;
    hp.write_map("cont_cmb_ns%d.fits"%o.nside_out,[q,u],overwrite=True)

if o.plot_stuff :
    q,u=hp.read_map("cont_cmb_ns%d.fits"%o.nside_out,field=[0,1],verbose=False)
    cctt,ccee,ccbb,ccte,cceb,cctb=np.array(hp.anafast([mz*msk,q*msk,u*msk],pol=True))/fsky
    dt,dq,du=hp.synfast([cltt,clee,clbb,clte],nside=o.nside_out,pol=True,new=True,verbose=False)
    
    plt.figure()
    plt.plot(l,clee,label='True EE')
    plt.plot(l,clbb,label='True BB')
    plt.plot(larr,ccee,label='Contaminant EE')
    plt.plot(larr,ccbb,label='Contaminant BB')
    plt.plot(larr,ccee+clee[:3*o.nside_out],label='Contaminated EE')
    plt.plot(larr,ccbb+clbb[:3*o.nside_out],label='Contaminated BB')
    plt.loglog()
    plt.legend()

    hp.mollview(dq*msk,title='True Q')
    hp.mollview(q*msk,title='Contaminant Q')
    hp.mollview((q+dq)*msk,title='Contaminated Q')
    plt.show()
