from __future__ import print_function
from optparse import OptionParser
import numpy as np
import flatmaps as fm
import matplotlib.pyplot as plt
from scipy.special import jv
import os

def opt_callback(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))
parser = OptionParser()
parser.add_option('--plot', dest='plot_stuff', default=False, action='store_true',
                  help='Set if you want to produce plots')
(o, args) = parser.parse_args()

l,cltt,clee,clbb,clte,nltt,nlee,nlbb,nlte=np.loadtxt("cls_lss.txt",unpack=True)
fmi,msk=fm.read_flat_map("mask_lss_flat.fits")

mpt,mpq,mpu=fmi.synfast(l,np.array([cltt,clee,clbb,clte]))

#Star contaminant (basically something that affects mostly small scales)
if not os.path.isfile("cont_lss_star_flat.fits") :
    loff=500.; ltouch=5000; frac=0.2; plaw=1.2;
    clstar=frac*cltt[ltouch]*((ltouch+loff)/(l+loff))**plaw; clstar[:2]=0;
    mp_star=fmi.synfast(l,clstar)

    if o.plot_stuff :
        ls,cl_c=fmi.anafast(mp_star)
        ls,cl_d=fmi.anafast(mpt)
        ls,cl_t=fmi.anafast(mp_star+mpt)
        plt.figure()
        plt.plot(ls,fmi.clbin(l,cltt)[1],'k-')
        plt.plot(ls,fmi.clbin(l,clstar)[1],'k--')
        plt.plot(ls,cl_c,label='Star contaminant')
        plt.plot(ls,cl_d,label='True $\\delta_g$')
        plt.plot(ls,cl_t,label='True + contaminant')
        plt.xlabel('$\\ell$',fontsize=18);
        plt.ylabel('$C_\\ell$',fontsize=18);
        plt.legend(); plt.yscale('log'); plt.xlim([17,20000])
        
        fmi.view_map(mpt*msk,title='True $\\delta_g$');
        fmi.view_map(mp_star*msk,title='Star contaminant')
        fmi.view_map((mp_star+mpt)*msk,title='True + contaminant')
        plt.show()
        
    fmi.write_flat_map("cont_lss_star_flat.fits",mp_star,descript='Star contaminant')

#Dust contaminant (power spectrum extrapolated from the full-sky map)
if not os.path.isfile("cont_lss_dust_flat.fits") :
    loff=10
    cldust=1E-7*((250+loff)/(l+loff))**2.4;
    mp_dust=fmi.synfast(l,cldust)

    if o.plot_stuff :
        ls,cl_c=fmi.anafast(mp_dust)
        ls,cl_d=fmi.anafast(mpt)
        ls,cl_t=fmi.anafast(mp_dust+mpt)
        plt.figure()
        plt.plot(ls,fmi.clbin(l,cltt)[1],'k-')
        plt.plot(ls,fmi.clbin(l,cldust)[1],'k--')
        plt.plot(ls,cl_c,label='Dust contaminant')
        plt.plot(ls,cl_d,label='True $\\delta_g$')
        plt.plot(ls,cl_t,label='True + contaminant')
        plt.xlabel('$\\ell$',fontsize=18);
        plt.ylabel('$C_\\ell$',fontsize=18);
        plt.legend(); plt.yscale('log'); plt.xlim([17,20000])
        
        fmi.view_map(mpt*msk,title='True $\\delta_g$');
        fmi.view_map(mp_dust*msk,title='Dust contaminant')
        fmi.view_map((mp_dust+mpt)*msk,title='True + contaminant')
        plt.show()
        
    fmi.write_flat_map("cont_lss_dust_flat.fits",mp_dust,descript='Dust contaminant')

#PSF contaminant to lensing
if not os.path.isfile("cont_wl_psf_flat.fits") :
    theta_fov_deg=3.5/2. #3.5-deg diameter FoV
    theta_fov_rad=theta_fov_deg*np.pi/180
    fl=2*jv(1,l*theta_fov_rad)/(l*theta_fov_rad); fl[0]=0
    ratio=np.sqrt(2*np.sum(clee[10:100])/np.sum(fl[10:100]**2))
    clcont=(ratio*fl)**2
    ct,cq,cu=fmi.synfast(l,np.array([clcont*0,clcont,clcont,clcont*0]))

    if o.plot_stuff :
        ls,cl_c=fmi.anafast(np.array([ct,cq,cu]))
        ls,cl_d=fmi.anafast(np.array([mpt,mpq,mpu]))
        ls,cl_t=fmi.anafast(np.array([mpt+ct,mpq+cq,mpu+cu]))

        plt.figure()
        plt.plot(ls,fmi.clbin(l,clee)[1],'k-')
        plt.plot(ls,fmi.clbin(l,clcont)[1],'k--')
        plt.plot(ls,cl_c[1],label='PSF contaminant (EE)')
        plt.plot(ls,cl_d[1],label='True $\\gamma_g$ (EE)')
        plt.plot(ls,cl_t[1],label='True + contaminant (EE)')
        plt.xlabel('$\\ell$',fontsize=18);
        plt.ylabel('$C_\\ell$',fontsize=18);
        plt.legend(); plt.yscale('log'); plt.xlim([17,20000])

        fmi.view_map(mpq*msk,title='True $\\gamma_g$ (EE)');
        fmi.view_map(cq*msk,title='PSF contaminant')
        fmi.view_map((cq+mpq)*msk,title='True + contaminant')
        plt.show()
        
    fmi.write_flat_map("cont_wl_psf_flat.fits",np.array([cq,cu]),
                       descript=np.array(['PSF Q','PSF U']))

#Small-scale contaminant to lensing
if not os.path.isfile("cont_wl_ss_flat.fits") :
    norm=np.mean(clee[10000:15000])*0.25
    cctt=np.zeros_like(l); ccte=np.zeros_like(l);
    ccee=norm*(10000/(l+1000.))**2; ccbb=ccee.copy()
    ct,cq,cu=fmi.synfast(l,np.array([cctt,ccee,ccbb,ccte]))

    if o.plot_stuff :
        ls,cl_c=fmi.anafast(np.array([ct,cq,cu]))
        ls,cl_d=fmi.anafast(np.array([mpt,mpq,mpu]))
        ls,cl_t=fmi.anafast(np.array([mpt+ct,mpq+cq,mpu+cu]))

        plt.figure()
        plt.plot(ls,fmi.clbin(l,clee)[1],'k-')
        plt.plot(ls,fmi.clbin(l,ccee)[1],'k--')
        plt.plot(ls,cl_c[1],label='Small-scale contaminant (EE)')
        plt.plot(ls,cl_d[1],label='True $\\gamma_g$ (EE)')
        plt.plot(ls,cl_t[1],label='True + contaminant (EE)')
        plt.xlabel('$\\ell$',fontsize=18);
        plt.ylabel('$C_\\ell$',fontsize=18);
        plt.legend(); plt.yscale('log'); plt.xlim([17,20000])

        fmi.view_map(mpq*msk,title='True $\\gamma_g$ (EE)');
        fmi.view_map(cq*msk,title='Small-scale contaminant')
        fmi.view_map((cq+mpq)*msk,title='True + contaminant')
        plt.show()
        
    fmi.write_flat_map("cont_wl_ss_flat.fits",np.array([cq,cu]),
                       descript=np.array(['SS Q','SS U']))

if o.plot_stuff :
    fdum,[cwp_q,cwp_u]=fm.read_flat_map("cont_wl_psf_flat.fits",i_map=-1)
    fdum,[cws_q,cws_u]=fm.read_flat_map("cont_wl_ss_flat.fits",i_map=-1)
    fdum,cld=fm.read_flat_map("cont_lss_dust_flat.fits")
    fdum,cls=fm.read_flat_map("cont_lss_star_flat.fits")
    mpz=np.zeros_like(mpt)

    tl=mpt+cld+cls;
    tw_q=mpq+cwp_q+cws_q; tw_u=mpu+cwp_u+cws_u;
    ls,c_l_cdcd=fmi.anafast(cld)
    ls,c_l_cscs=fmi.anafast(cls)
    ls,c_l_dd=fmi.anafast(mpt)
    ls,c_l_tt=fmi.anafast(tl)
    ls,[c_l_dd_tt,c_w_dd_ee,c_w_dd_bb,c_l_dd_te,c_w_dd_eb,c_l_dd_tb]=fmi.anafast(np.array([mpt,mpq,mpu]))
    ls,[c_l_tt_tt,c_w_tt_ee,c_w_tt_bb,c_l_tt_te,c_w_tt_eb,c_l_tt_tb]=fmi.anafast(np.array([tl,tw_q,tw_u]))
    ls,[dum1,c_w_cscs_ee,c_w_cscs_bb,dum2,dum3,dum4]=fmi.anafast(np.array([mpz,cws_q,cws_u]))
    ls,[dum1,c_w_cpcp_ee,c_w_cpcp_bb,dum2,dum3,dum4]=fmi.anafast(np.array([mpz,cwp_q,cwp_u]))

    plt.figure()
    plt.plot(ls,c_l_dd,label='True $\\delta_g$')
    plt.plot(ls,c_l_cdcd,label='Dust contaminant')
    plt.plot(ls,c_l_cscs,label='Star contaminant')
    plt.plot(ls,c_l_tt,label='Contaminated $\\delta_g$')
    plt.xlabel('$\\ell$',fontsize=18);
    plt.ylabel('$C_\\ell$',fontsize=18);
    plt.legend(); plt.loglog(); plt.xlim([17,20000])
    
    plt.figure()
    plt.plot(ls,c_w_dd_ee,label='True $\\gamma_g$ EE')
    plt.plot(ls,c_w_cpcp_ee,label='PSF contaminant EE')
    plt.plot(ls,c_w_cscs_ee,label='Small-scale contaminant EE')
    plt.plot(ls,c_w_tt_ee,label='Contaminated $\\gamma_g$ EE')
    plt.xlabel('$\\ell$',fontsize=18);
    plt.ylabel('$C_\\ell$',fontsize=18);
    plt.legend(); plt.loglog(); plt.xlim([17,20000])
    
    plt.figure()
    plt.plot(ls,c_w_dd_bb,label='True $\\gamma_g$ BB')
    plt.plot(ls,c_w_cpcp_bb,label='PSF contaminant BB')
    plt.plot(ls,c_w_cscs_bb,label='Small-scale contaminant BB')
    plt.plot(ls,c_w_tt_bb,label='Contaminated $\\gamma_g$ BB')
    plt.xlabel('$\\ell$',fontsize=18);
    plt.ylabel('$C_\\ell$',fontsize=18);
    plt.legend(); plt.loglog(); plt.xlim([17,20000])

    fmi.view_map(mpt*msk,title='True $\\delta_g$')
    fmi.view_map(tl*msk,title='Contaminated $\\delta_g$')
    fmi.view_map(cld*msk,title='Dust contaminant')
    fmi.view_map(cls*msk,title='Star contaminant')

    fmi.view_map(mpq*msk,title='True $\\gamma_g$ Q')
    fmi.view_map(tw_q*msk,title='Contaminated $\\gamma_g$ Q')
    fmi.view_map(cwp_q*msk,title='PSF contaminant Q')
    fmi.view_map(cws_q*msk,title='Small-scale contaminant Q')

    plt.show()
