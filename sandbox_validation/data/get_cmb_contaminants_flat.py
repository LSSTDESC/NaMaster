from __future__ import print_function
from optparse import OptionParser
import numpy as np
import flatmaps as fm
import matplotlib.pyplot as plt
import os

def opt_callback(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))
parser = OptionParser()
parser.add_option('--plot', dest='plot_stuff', default=False, action='store_true',
                  help='Set if you want to produce plots')
(o, args) = parser.parse_args()

def read_cl_camb(fname) :
    data=np.loadtxt(fname,unpack=True)
    ll=np.arange(len(data[0])+2)+0.
    fac=2*np.pi/(ll[2:]*(ll[2:]+1.))
    cl_tt=np.zeros_like(ll); cl_tt[2:]=data[1,:]*fac
    cl_ee=np.zeros_like(ll); cl_ee[2:]=data[2,:]*fac
    cl_bb=np.zeros_like(ll); cl_bb[2:]=data[3,:]*fac
    cl_te=np.zeros_like(ll); cl_te[2:]=data[4,:]*fac

    return ll,cl_tt,cl_ee,cl_bb,cl_te

l,cltt,clee,clbb,clte=read_cl_camb("cls_cmb.txt")
fmi,msk=fm.read_flat_map("mask_cmb_flat.fits")

#Generate random homogeneous realization with the right power spectrum.
ccee=2E-5*(100./(l+0.1))**2.5; ccee[:10]=ccee[10]
ccbb=9E-6*(100./(l+0.1))**2.3; ccbb[:10]=ccbb[10]
czero=np.zeros_like(ccee)
ratio=np.sqrt(0.2*np.mean(clbb[500:600])/np.mean(ccbb[500:600]))
t,q,u=fmi.synfast(l,np.array([czero,ccee,ccbb,czero]))
q*=ratio; u*=ratio;

fmi.write_flat_map("cont_cmb_flat.fits",np.array([q,u]))

if o.plot_stuff :
    fdum,[cq,cu]=fm.read_flat_map("cont_cmb_flat.fits",i_map=-1)
    dt,dq,du=fmi.synfast(l,np.array([cltt,clee,clbb,clte]))

    ls_c,[ctt_c,cee_c,cbb_c,cte_c,ceb_c,ctb_c]=fmi.anafast(np.array([0*cq,cq,cu]))
    ls_t,[ctt_t,cee_t,cbb_t,cte_t,ceb_t,ctb_t]=fmi.anafast(np.array([dt,dq,du]))
    ls_d,[ctt_d,cee_d,cbb_d,cte_d,ceb_d,ctb_d]=fmi.anafast(np.array([dt+0*cq,dq+cq,du+cu]))

    plt.figure()
    plt.plot(ls_t,cee_t,label='True EE')
    plt.plot(ls_c,cee_c,label='Contaminat EE')
    plt.plot(ls_d,cee_d,label='Contaminated EE')
    plt.plot(l,clee,label='Input EE')
    plt.plot(ls_t,cbb_t,label='True BB')
    plt.plot(ls_c,cbb_c,label='Contaminat BB')
    plt.plot(ls_d,cbb_d,label='Contaminated BB')
    plt.plot(l,clbb,label='Input BB')
    plt.xlabel('$\\ell$',fontsize=18);
    plt.ylabel('$C_\\ell$',fontsize=18);
    plt.legend(); plt.loglog();
    
    fmi.view_map(dq*msk,title='True Q')
    fmi.view_map(q*msk,title='Contaminant Q')
    fmi.view_map((q+dq)*msk,title='Contaminated Q')
    plt.show()
