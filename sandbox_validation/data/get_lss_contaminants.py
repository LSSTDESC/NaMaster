from __future__ import print_function
from optparse import OptionParser
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from scipy.special import jv
import os

def opt_callback(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))
parser = OptionParser()
parser.add_option('--plot', dest='plot_stuff', default=False, action='store_true',
                  help='Set if you want to produce plots')
parser.add_option('--nside', dest='nside_out', default=512, type=int,
                  help='Resolution parameter')
(o, args) = parser.parse_args()

l,cltt,clee,clbb,clte,nltt,nlee,nlbb,nlte=np.loadtxt("cls_lss.txt",unpack=True)
msk=hp.ud_grade(hp.read_map("mask_lss.fits",verbose=False),nside_out=o.nside_out); fsky=np.mean(msk)

larr=np.arange(3*o.nside_out)

if not os.path.isfile("cont_lss_nvar_ns%d.fits"%o.nside_out) :
    #First, variations in \bar{n} due to stars
    starmap=hp.read_map("star_template.fits",verbose=False)
    starmap[starmap<=0]=1E-15
    starmap_low=hp.smoothing(hp.ud_grade(hp.ud_grade(starmap,nside_out=64),nside_out=o.nside_out),fwhm=3.5*np.pi/180,verbose=False) #Smooth out the star map to 3.5deg FWHM
    r_starmap=starmap_low/np.amax(starmap_low[msk>0]) #Normalize by maximum value
    shotnoise_factor=(r_starmap-4)/(2*r_starmap-4) #Compute shot noise factor. This interpolates between 1 (no stars) and 1.5 (lots of stars)

    #Now, impose additional depth variations on scales of 3.5 degrees
    cl_extravar=np.exp(-larr*(larr+1)*(3.5*np.pi/180/2.355)**2)
    norm=0.67E-3/np.sum(cl_extravar*(2*larr+1)/4/np.pi); cl_extravar*=norm; #This yields <10% variations
    depth_factor=1+hp.synfast(cl_extravar,o.nside_out,new=True,verbose=False);

    #This total map corresponds to the relative variation in the shot-noise variance
    snvmap=shotnoise_factor*depth_factor*msk

    if o.plot_stuff :
        hp.mollview(snvmap,min=0.9,title='Relative local shot noise variance');
        plt.show()

    hp.write_map("cont_lss_nvar_ns%d.fits"%o.nside_out,snvmap,overwrite=True)

if not os.path.isfile("cont_lss_star_ns%d.fits"%o.nside_out) :
    starmap=hp.ud_grade(hp.read_map("star_template.fits",verbose=False),nside_out=o.nside_out)
    starmap[starmap<=0]=1E-15
    starmap=-np.log(starmap) #Contaminant will be proportional to log(n_star)
    mean=np.sum(starmap*msk)/np.sum(msk)

    clcont=hp.anafast((starmap-mean)*msk)/fsky
    nflat=np.mean(clcont[max(2,o.nside_out//3-50):o.nside_out//3+50])*np.ones_like(clcont); #Add extra fluctuations beyond ell~nside/3 with flat power spectrum
    dstar=hp.synfast(nflat,o.nside_out,new=True,verbose=False)
    starmap_b=np.maximum(starmap+dstar,0)
    clcont_b=hp.anafast((starmap_b-mean)*msk)/fsky
    ratio=-np.sqrt(0.03*np.sum(cltt[max(2,o.nside_out//3-50):o.nside_out//3+50])/np.sum(clcont_b[max(2,o.nside_out//3-50):o.nside_out//3+50])) #3% contamination at ell~nside/3
    starmap_b=(starmap_b-mean)*ratio

    dmap=hp.synfast(cltt[:3*o.nside_out],o.nside_out,new=True,verbose=False);

    if o.plot_stuff :
        cld=hp.anafast(dmap*msk)/fsky
        clc=hp.anafast(starmap_b*msk)/fsky
        clt=hp.anafast((dmap+starmap_b)*msk)/fsky
        plt.figure();
        plt.plot(l,cltt,'k-')
        plt.plot(larr,clc,label='Star contaminant');
        plt.plot(larr,cld,label='True $\\delta_g$');
        plt.plot(larr,clt,label='True + contaminant');
        plt.xlabel('$\\ell$',fontsize=18);
        plt.ylabel('$C_\\ell$',fontsize=18);
        plt.legend(); plt.loglog(); plt.xlim([2,3*o.nside_out])

        hp.mollview(dmap*msk,title='True $\\delta_g$')
        hp.mollview(starmap_b*msk,title='Star contaminant')
        hp.mollview((starmap_b+dmap)*msk,title='True + contaminant')
        plt.show()
    
    hp.write_map("cont_lss_star_ns%d.fits"%o.nside_out,starmap_b,overwrite=True)

if not os.path.isfile("cont_lss_dust_ns%d.fits"%o.nside_out) :
    dust=hp.ud_grade(hp.read_map("lambda_sfd_ebv.fits",verbose=False),nside_out=o.nside_out)
    mean=np.amin(dust*msk)
    dust=mean-dust
    
    cl=hp.anafast(dust*msk)/fsky
    ratio=np.sqrt(0.2*np.sum(cltt[50:200])/np.sum(cl[50:200])) #20% contamination in the power spectrum at ell~100
    dust=dust*ratio

    if o.plot_stuff :
        cl=hp.anafast(dust*msk)/fsky
        plt.figure();
        plt.plot(larr,cl,label='Dust contaminant');
        plt.plot(l,cltt,label='True $\\delta_g$');
        plt.plot(larr,cl+cltt[:3*o.nside_out],label='True + contaminant')
        plt.xlabel('$\\ell$',fontsize=18);
        plt.ylabel('$C_\\ell$',fontsize=18);
        plt.legend(); plt.loglog(); plt.xlim([2,3*o.nside_out])
    
        dmap=hp.synfast(cltt[:3*o.nside_out],o.nside_out,verbose=False);
    
        hp.mollview(dust*msk,title='Dust contaminant')
        hp.mollview(dmap*msk,title='True $\\delta_g$')
        hp.mollview((dmap+dust)*msk,title='True + contaminant')
        plt.show()

    hp.write_map("cont_lss_dust_ns%d.fits"%o.nside_out,dust,overwrite=True)
 
if not os.path.isfile("cont_wl_psf_ns%d.fits"%o.nside_out) :
    theta_fov_deg=3.5/2. #3.5-deg diameter FoV
    theta_fov_rad=theta_fov_deg*np.pi/180
    fl=2*jv(1,l*theta_fov_rad)/(l*theta_fov_rad)

    ratio=np.sqrt(0.3*np.sum(clee[10:100])/np.sum(fl[10:100]**2))

    clcont=(ratio*fl)**2
    ct,cq,cu=hp.synfast([clcont[:3*o.nside_out]*0,clcont[:3*o.nside_out],clcont[:3*o.nside_out],clcont[:3*o.nside_out]*0],o.nside_out,new=True,verbose=False,pol=True)

    if o.plot_stuff :
        plt.figure();
        plt.plot(l,clcont,label='PSF contaminant (EE)');
        plt.plot(l,clee,label='True $\\gamma_g$ (EE)');
        plt.plot(l,clee+clcont,label='True + contaminant (EE)');
        plt.xlabel('$\\ell$',fontsize=18);
        plt.ylabel('$C_\\ell$',fontsize=18);
        plt.legend(); plt.loglog(); plt.xlim([2,3*o.nside_out])

        t,q,u=hp.synfast([cltt[:3*o.nside_out],clee[:3*o.nside_out],clbb[:3*o.nside_out],clte[:3*o.nside_out]],o.nside_out,new=True,verbose=False,pol=True)

        hp.mollview(cq*msk,title='PSF contaminant (Q)')
        hp.mollview(q*msk,title='True $\\gamma_g$ (Q)')
        hp.mollview((q+cq)*msk,title='True + contaminant (Q)')
        plt.show()

    hp.write_map("cont_wl_psf_ns%d.fits"%o.nside_out,[cq,cu],overwrite=True)

if not os.path.isfile("cont_wl_ss_ns%d.fits"%o.nside_out) :
    #Get map with flat power spectrum containing E and B modes
    #We want it to have a 20% contamination at ell~nside. This computes the normalization factor
    norm=0.2*np.mean(clee[max(2,o.nside_out-50):o.nside_out+50])
    cctt=np.zeros_like(larr); ccte=np.zeros_like(larr);
    ccee=norm*np.ones_like(larr); ccbb=norm*np.ones_like(larr);
    ct,cq,cu=hp.synfast([cctt,ccee,ccbb,ccte],o.nside_out,new=True,verbose=False,pol=True)

    if o.plot_stuff :
        plt.figure()
        plt.plot(larr,ccee,label='Small-scale contaminant')
        plt.plot(l,clee,label='True $\\gamma_g$')
        plt.plot(larr,ccee+clee[:3*o.nside_out],label='True + contaminant')
        plt.xlabel('$\\ell$',fontsize=18);
        plt.ylabel('$C_\\ell$',fontsize=18);
        plt.legend(); plt.loglog(); plt.xlim([2,3*o.nside_out])

        t,q,u=hp.synfast([cltt[:3*o.nside_out],clee[:3*o.nside_out],clbb[:3*o.nside_out],clte[:3*o.nside_out]],o.nside_out,new=True,verbose=False,pol=True)

        hp.mollview(cq*msk,title='Small-scale contaminant (Q)')
        hp.mollview(q*msk,title='True $\\gamma_g$ (Q)')
        hp.mollview((q+cq)*msk,title='True + contaminant (Q)')
        plt.show()

    hp.write_map("cont_wl_ss_ns%d.fits"%o.nside_out,[cq,cu],overwrite=True)

if o.plot_stuff :    
    mpzero=np.zeros_like(msk)
    cwp_q,cwp_u=hp.read_map("cont_wl_psf_ns%d.fits"%o.nside_out,field=[0,1],verbose=False)
    cws_q,cws_u=hp.read_map("cont_wl_ss_ns%d.fits"%o.nside_out,field=[0,1],verbose=False)                            
    cld=hp.read_map("cont_lss_dust_ns%d.fits"%o.nside_out,field=0,verbose=False)
    cls=hp.read_map("cont_lss_star_ns%d.fits"%o.nside_out,field=0,verbose=False)
    dl,dw_q,dw_u=hp.synfast([cltt[:3*o.nside_out],clee[:3*o.nside_out],clbb[:3*o.nside_out],clte[:3*o.nside_out]],o.nside_out,new=True,verbose=False,pol=True)
    tl=dl+cls+cld
    tw_q=dw_q+cwp_q+cws_q; tw_u=dw_u+cwp_u+cws_u;
    c_l_cdcd=hp.anafast(cld*msk)/fsky
    c_l_cscs=hp.anafast(cls*msk)/fsky
    c_l_dd=hp.anafast(dl*msk)/fsky
    c_l_tt=hp.anafast(tl*msk)/fsky
    c_l_dd_tt,c_w_dd_ee,c_w_dd_bb,c_l_dd_te,dum1,dum2=hp.anafast([dl,dw_q,dw_u],pol=True)
    c_l_tt_tt,c_w_tt_ee,c_w_tt_bb,c_l_tt_te,dum1,dum2=hp.anafast([tl,tw_q,tw_u],pol=True)
    dum1,c_w_cscs_ee,c_w_cscs_bb,dum2,dum3,dum4=hp.anafast([mpzero,cws_q,cws_u],pol=True)
    dum1,c_w_cpcp_ee,c_w_cpcp_bb,dum2,dum3,dum4=hp.anafast([mpzero,cwp_q,cwp_u],pol=True)
    
    plt.figure();
    plt.plot(larr,c_l_dd,label='True $\\delta_g$')
    plt.plot(larr,c_l_cdcd,label='Dust contaminant')
    plt.plot(larr,c_l_cscs,label='Star contaminant')
    plt.plot(larr,c_l_tt,label='Contaminated $\\delta_g$')
    plt.xlabel('$\\ell$',fontsize=18);
    plt.ylabel('$C_\\ell$',fontsize=18);
    plt.legend(); plt.loglog(); plt.xlim([2,3*o.nside_out])
    
    plt.figure()
    plt.plot(larr,c_w_dd_ee,label='True $\\gamma_g$ EE')
    plt.plot(larr,c_w_cpcp_ee,label='PSF contaminant EE')
    plt.plot(larr,c_w_cscs_ee,label='Small-scale contaminant EE')
    plt.plot(larr,c_w_tt_ee,label='Contaminated $\\gamma_g$ EE')
    plt.xlabel('$\\ell$',fontsize=18);
    plt.ylabel('$C_\\ell$',fontsize=18);
    plt.legend(); plt.loglog(); plt.xlim([2,3*o.nside_out])
    
    plt.figure()
    plt.plot(larr,c_w_dd_bb,label='True $\\gamma_g$ BB')
    plt.plot(larr,c_w_cpcp_bb,label='PSF contaminant BB')
    plt.plot(larr,c_w_cscs_bb,label='Small-scale contaminant BB')
    plt.plot(larr,c_w_tt_bb,label='Contaminated $\\gamma_g$ BB')
    plt.xlabel('$\\ell$',fontsize=18);
    plt.ylabel('$C_\\ell$',fontsize=18);
    plt.legend(); plt.loglog(); plt.xlim([2,3*o.nside_out])
    
    hp.mollview(dl*msk,title='True $\\delta_g$')
    hp.mollview(tl*msk,title='Contaminated $\\delta_g$')
    hp.mollview(cld*msk,title='Dust contaminant')
    hp.mollview(cls*msk,title='Star contaminant')

    hp.mollview(dw_q*msk,title='True $\\gamma_g$ Q')
    hp.mollview(tw_q*msk,title='Contaminated $\\gamma_g$ Q')
    hp.mollview(cwp_q*msk,title='PSF contaminant Q')
    hp.mollview(cws_q*msk,title='Small-scale contaminant Q')
    
    plt.show()
