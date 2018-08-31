from __future__ import print_function
from optparse import OptionParser
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import flatmaps as fm
import os
from matplotlib import rc
import matplotlib
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

def opt_callback(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))
parser = OptionParser()
parser.add_option('--plot', dest='plot_stuff', default=False, action='store_true',
                  help='Set if you want to produce plots')
parser.add_option('--figure',dest='whichfig',default=-1, type=int,
                  help='Figure number. Pass -1 for all figures')
(o, args) = parser.parse_args()

nside_lss=1024
nside_cmb=256

def tickfs(ax,x=True,y=True) :
    if x :
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(12)
    if y :
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(12)

if (o.whichfig==3) or (o.whichfig==-1) :
    #Plotting flat-sky LSS maps
    l,cltt,clee,clbb,clte,nltt,nlee,nlbb,nlte=np.loadtxt("cls_lss.txt",unpack=True)
    fmi,msk_f=fm.read_flat_map("mask_lss_flat.fits")
    mpt_f,mpq_f,mpu_f=fmi.synfast(l,np.array([cltt,clee,clbb,clte]))
    fdum,[cwp_q_f,cwp_u_f]=fm.read_flat_map("cont_wl_psf_flat.fits",i_map=-1)
    fdum,[cws_q_f,cws_u_f]=fm.read_flat_map("cont_wl_ss_flat.fits",i_map=-1)
    fdum,cld_f=fm.read_flat_map("cont_lss_dust_flat.fits")
    fdum,cls_f=fm.read_flat_map("cont_lss_star_flat.fits")
    plt.figure(figsize=(12,9.5))
    tonull=np.where(msk_f==0)
    fs_or=matplotlib.rcParams['font.size']
    ax=plt.subplot(421,projection=fmi.wcs); mpp=msk_f.copy();# mpp[tonull]=np.nan
    fmi.view_map(mpp,ax=ax,addColorbar=False,title='Mask')
    ax=plt.subplot(422,projection=fmi.wcs); mpp=msk_f*mpt_f; mpp[tonull]=np.nan
    fmi.view_map(mpp,ax=ax,addColorbar=False,title='$\\delta$',tfs=14)
    ax=plt.subplot(423,projection=fmi.wcs); mpp=msk_f*mpq_f; mpp[tonull]=np.nan
    fmi.view_map(mpp,ax=ax,addColorbar=False,title='$\\gamma_1$',tfs=14)
    ax=plt.subplot(424,projection=fmi.wcs); mpp=msk_f*mpu_f; mpp[tonull]=np.nan
    fmi.view_map(mpp,ax=ax,addColorbar=False,title='$\\gamma_2$',tfs=14)
    ax=plt.subplot(425,projection=fmi.wcs); mpp=msk_f*cld_f; mpp[tonull]=np.nan
    matplotlib.rcParams.update({'font.size':fs_or})
    fmi.view_map(mpp,ax=ax,addColorbar=False,title='Dust')
    ax=plt.subplot(426,projection=fmi.wcs); mpp=msk_f*cls_f; mpp[tonull]=np.nan
    fmi.view_map(mpp,ax=ax,addColorbar=False,title='Stars')
    ax=plt.subplot(427,projection=fmi.wcs); mpp=msk_f*cwp_q_f; mpp[tonull]=np.nan
    fmi.view_map(mpp,ax=ax,addColorbar=False,title='PSF')
    ax=plt.subplot(428,projection=fmi.wcs); mpp=msk_f*cws_u_f; mpp[tonull]=np.nan
    fmi.view_map(mpp,ax=ax,addColorbar=False,title='Small-scale')
    plt.savefig("../plots_paper/maps_lss_flat.pdf",bbox_inches='tight')
    plt.show()

if (o.whichfig==2) or (o.whichfig==-1) :
    #Plotting full-sky LSS maps
    l,cltt,clee,clbb,clte,nltt,nlee,nlbb,nlte=np.loadtxt("cls_lss.txt",unpack=True)
    depth_nvar=hp.read_map("cont_lss_nvar_ns%d.fits"%nside_lss,verbose=False)
    depth_nvar[depth_nvar<0.8]=0
    depth_ivar=np.zeros_like(depth_nvar); depth_ivar[depth_nvar>0.1]=1./depth_nvar[depth_nvar>0.1]
    msk_s=hp.read_map("mask_lss_ns%d.fits"%nside_lss,verbose=False)
    msk_s*=depth_ivar
    mpt_s,mpq_s,mpu_s=hp.synfast([cltt[:3*nside_lss],clee[:3*nside_lss],
                                  clbb[:3*nside_lss],clte[:3*nside_lss]],
                                 nside_lss,new=True,verbose=False,pol=True)
    cld_s=hp.read_map("cont_lss_dust_ns%d.fits"%nside_lss,field=0,verbose=False)
    cls_s=hp.read_map("cont_lss_star_ns%d.fits"%nside_lss,field=0,verbose=False)
    cwp_q_s,cwp_u_s=hp.read_map("cont_wl_psf_ns%d.fits"%nside_lss,field=[0,1],verbose=False)
    cws_q_s,cws_u_s=hp.read_map("cont_wl_ss_ns%d.fits"%nside_lss,field=[0,1],verbose=False)
    tonull=np.where(msk_s==0)
    plt.figure(figsize=(12,13))
    fs_or=matplotlib.rcParams['font.size']
    ax=plt.subplot(421); mpp=msk_s.copy();# mpp[tonull]=hp.UNSEEN
    hp.mollview(mpp,hold=True,notext=True,coord=['G','C'],title='Mask',cbar=False,min=0,max=1.09)
    ax=plt.subplot(422); mpp=msk_s*mpt_s; mpp[tonull]=hp.UNSEEN
    matplotlib.rcParams.update({'font.size':14})
    hp.mollview(mpp,hold=True,notext=True,coord=['G','C'],title='$\\delta$',cbar=False)
    ax=plt.subplot(423); mpp=msk_s*mpq_s; mpp[tonull]=hp.UNSEEN
    hp.mollview(mpp,hold=True,notext=True,coord=['G','C'],title='$\\gamma_1$',cbar=False)
    ax=plt.subplot(424); mpp=msk_s*mpu_s; mpp[tonull]=hp.UNSEEN
    hp.mollview(mpp,hold=True,notext=True,coord=['G','C'],title='$\\gamma_2$',cbar=False)
    ax=plt.subplot(425); mpp=msk_s*cld_s; mpp[tonull]=hp.UNSEEN
    matplotlib.rcParams.update({'font.size':fs_or})
    hp.mollview(mpp,hold=True,notext=True,coord=['G','C'],title='Dust',cbar=False)
    ax=plt.subplot(426); mpp=msk_s*cls_s; mpp[tonull]=hp.UNSEEN
    hp.mollview(mpp,hold=True,notext=True,coord=['G','C'],title='Stars',cbar=False)
    ax=plt.subplot(427); mpp=msk_s*cwp_q_s; mpp[tonull]=hp.UNSEEN
    hp.mollview(mpp,hold=True,notext=True,coord=['G','C'],title='PSF',cbar=False)
    ax=plt.subplot(428); mpp=msk_s*cws_u_s; mpp[tonull]=hp.UNSEEN
    hp.mollview(mpp,hold=True,notext=True,coord=['G','C'],title='Small-scale',cbar=False)
    plt.savefig("../plots_paper/maps_lss_sph.pdf",bbox_inches='tight')
    plt.show()

if (o.whichfig==1) or (o.whichfig==-1) :
    #Plotting LSS power spectra
    l,cltt,clee,clbb,clte,nltt,nlee,nlbb,nlte=np.loadtxt("cls_lss.txt",unpack=True)

    plt.figure()
    ax=plt.gca()
    ax.plot(l,cltt,'-',c='#000099',lw=2,label='$\\delta\\times\\delta$')
    ax.plot(l,clte,'-',c='#00CCCC',lw=2,label='$\\delta\\times\\gamma^E$')
    ax.plot(l,clee,'-',c='#CC0000',lw=2,label='$\\gamma^E\\times\\gamma^E$')
    ax.plot(l,nltt,'--',c='#000099',lw=2)
    ax.plot(l,nlte,'--',c='#00CCCC',lw=2)
    ax.plot(l,nlee,'--',c='#CC0000',lw=2)
    ax.plot([-1,-1],[-1,-1],'k-',lw=2,label='Signal')
    ax.plot([-1,-1],[-1,-1],'k--',lw=2,label='Noise')
    plt.legend(loc='lower left',frameon=False,ncol=2,fontsize=15)
    ax.set_xlabel("$\\ell$",fontsize=15)
    ax.set_ylabel("$C_\\ell$",fontsize=15)
    ax.set_xlim([2,3E4])
    ax.set_ylim([5E-13,5E-6])
    tickfs(ax)
    plt.loglog()
    plt.savefig("../plots_paper/cls_lss.pdf",bbox_inches='tight')
    plt.show()

if (o.whichfig==4) or (o.whichfig==-1) :
    #Plotting LSS contaminant power spectra
    l,cltt,clee,clbb,clte,nltt,nlee,nlbb,nlte=np.loadtxt("cls_lss.txt",unpack=True)
    ll=l[:3*nside_lss]; cl_x=cltt[:3*nside_lss]; cw_x=clee[:3*nside_lss]
    
    msk_s=hp.read_map("mask_lss_ns%d.fits"%nside_lss,verbose=False); fsky=np.mean(msk_s)
    l_s=hp.read_map("cont_lss_star_ns%d.fits"%nside_lss,verbose=False)
    l_d=hp.read_map("cont_lss_dust_ns%d.fits"%nside_lss,verbose=False)
    w_pq,w_pu=hp.read_map("cont_wl_psf_ns%d.fits"%nside_lss,verbose=False,field=[0,1])
    w_sq,w_su=hp.read_map("cont_wl_ss_ns%d.fits"%nside_lss ,verbose=False,field=[0,1])
    cl_d,cw_pe,cw_pb,_,_,_=hp.anafast([l_d*msk_s,w_pq*msk_s,w_pu*msk_s],pol=True,iter=0)/fsky
    cl_s,cw_se,cw_sb,_,_,_=hp.anafast([l_s*msk_s,w_sq*msk_s,w_su*msk_s],pol=True,iter=0)/fsky

    def rebin_and_plot(x,y,b,lt,c,ax,lb=None) :
        xp=np.mean(x.reshape([len(x)//b,b]),axis=1)
        yp=np.mean(y.reshape([len(x)//b,b]),axis=1)
        if lb is None :
            ax.plot(xp,yp,lt,c=c)
        else :
            ax.plot(xp,yp,lt,c=c,label=lb)
    plt.figure(figsize=(7,8))
    plt.subplots_adjust(hspace=0)
    ax1=plt.subplot(311)
    c_x='#000099'; c_d='#00CCCC'; c_s='#CC0000'; c_t='#FF9933'; rb=8
    rebin_and_plot(ll,cl_x,rb,'-',c_x,ax1,lb='Signal')
    rebin_and_plot(ll,cl_d,rb,'--',c_d,ax1,lb='Dust')
    rebin_and_plot(ll,cl_s,rb,'-.',c_s,ax1,lb='Stars')
    rebin_and_plot(ll,cl_x+cl_d+cl_s,rb,':',c_t,ax1,lb='Contaminated signal')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xticklabels([])
    ax1.set_ylim([2E-9,9E-4])
    ax1.legend(loc='upper right',frameon=False)
    ax1.set_ylabel('$C_\\ell^{\\delta\\delta}$',fontsize=15)
    tickfs(ax1,x=False)
    ax2=plt.subplot(312)
    rebin_and_plot(ll,cw_x ,rb,'-',c_x,ax2,lb='Signal')
    rebin_and_plot(ll,cw_pe,rb,'--',c_d,ax2,lb='PSF')
    rebin_and_plot(ll,cw_se,rb,'-.',c_s,ax2,lb='Small-scale')
    rebin_and_plot(ll,cw_x+cw_pe+cw_se,rb,':',c_t,ax2,lb='Contaminated signal')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xticklabels([])
    ax2.set_ylim([4E-11,9E-8])
    ax2.legend(loc='upper right',frameon=False)
    ax2.set_ylabel('$C_\\ell^{EE}$',fontsize=15)
    tickfs(ax2,x=False)
    ax3=plt.subplot(313)
    rebin_and_plot(ll,cw_pb,rb,'--',c_d,ax3,lb='PSF')
    rebin_and_plot(ll,cw_sb,rb,'-.',c_s,ax3,lb='Small-scale')
    rebin_and_plot(ll,cw_pb+cw_sb,rb,':',c_t,ax3,lb='Contaminated signal')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_ylim([4E-11,9E-8])
    ax3.legend(loc='upper right',frameon=False)
    ax3.set_ylabel('$C_\\ell^{BB}$',fontsize=15)
    ax3.set_xlabel('$\\ell$',fontsize=15)
    tickfs(ax3)
    ax1.set_xlim([rb/2,2*nside_lss])
    ax2.set_xlim([rb/2,2*nside_lss])
    ax3.set_xlim([rb/2,2*nside_lss])
    plt.savefig("../plots_paper/cls_cont_lss.pdf",bbox_inches='tight')
    plt.show()
    
if (o.whichfig==5) or (o.whichfig==-1) :
    #Plotting CMB stuff
    msk_s=hp.read_map("mask_cmb_ns%d.fits"%nside_cmb,verbose=False)
    fmi,msk_f=fm.read_flat_map("mask_cmb_flat.fits")
    fdum,[cq_f,cu_f]=fm.read_flat_map("cont_cmb_flat.fits",i_map=-1)
    cq_s,cu_s=hp.read_map("cont_cmb_ns%d.fits"%nside_cmb,field=[0,1],verbose=False)
    tonull_s=np.where(msk_s==0)
    tonull_f=np.where(msk_f==0)
    plt.figure(figsize=(12,9.75))
    ax=plt.subplot(321); mpp=msk_s.copy();
    hp.mollview(mpp,hold=True,notext=True,coord=['G','C'],title='Mask',cbar=False)
    ax=plt.subplot(322,projection=fmi.wcs); mpp=msk_f.copy();
    fmi.view_map(mpp,ax=ax,addColorbar=False,title="Mask",xlabel="")
    ax=plt.subplot(323); mpp=msk_s*cq_s;
    hp.mollview(mpp,hold=True,notext=True,coord=['G','C'],title='$Q_{\\rm FG}$',cbar=False)
    ax=plt.subplot(324,projection=fmi.wcs); mpp=msk_f*cq_f;
    fmi.view_map(mpp,ax=ax,addColorbar=False,title="$Q_{\\rm FG}$",xlabel="")
    ax=plt.subplot(325); mpp=msk_s*cu_s;
    hp.mollview(mpp,hold=True,notext=True,coord=['G','C'],title='$U_{\\rm FG}$',cbar=False)
    ax=plt.subplot(326,projection=fmi.wcs); mpp=msk_f*cu_f;
    fmi.view_map(mpp,ax=ax,addColorbar=False,title="$U_{\\rm FG}$")
    plt.savefig("../plots_paper/maps_cmb.pdf",bbox_inches='tight')
    plt.show()

if (o.whichfig==6) or (o.whichfig==-1) :
    lmax=7000
    def read_cl_camb(fname) :
        data=np.loadtxt(fname,unpack=True)
        ll=np.arange(lmax,dtype=float)
        fac=2*np.pi/(ll[2:]*(ll[2:]+1.))
        cl_tt=np.zeros_like(ll); cl_tt[2:]=data[1,:lmax-2]*fac
        cl_ee=np.zeros_like(ll); cl_ee[2:]=data[2,:lmax-2]*fac
        cl_bb=np.zeros_like(ll); cl_bb[2:]=data[3,:lmax-2]*fac
        cl_te=np.zeros_like(ll); cl_te[2:]=data[4,:lmax-2]*fac

        return ll,cl_tt,cl_ee,cl_bb,cl_te

    l,cltt,clee,clbb,clte=read_cl_camb("cls_cmb.txt")
    ccee=2E-5*(100./(l+0.1))**2.5;
    ccbb=9E-6*(100./(l+0.1))**2.3;
    nlev_s=1.; nlev_l=0.5; lk_s=25.; lk_l=300.; fwhm_s=20.; fwhm_l=1.4;
    def get_noise(ll,nlev,lk,fwhm) :
        flat=2*(nlev*np.pi/(180*60))**2
        oeff=(np.ones_like(ll)+(lk/(ll+0.1))**2.4)
        beam=np.exp(ll*(ll+1)*(fwhm*np.pi/(180*60*2.355))**2)
        return flat*oeff*beam
    nsee=get_noise(l,nlev_s,lk_s,fwhm_s)
    nlee=get_noise(l,nlev_l,lk_l,fwhm_l)

    ratio_s=np.sqrt(0.1*np.mean(clbb[40:60])/np.mean(ccbb[40:60]))
    ratio_l=np.sqrt(0.2*np.mean(clbb[500:600])/np.mean(ccbb[500:600]))
    print(ratio_l,ratio_s)
    plt.figure()
    ax=plt.gca()
    ax.plot(l,clee,'-',c='b',lw=2,label='Signal $EE$')
    ax.plot(l,clbb,'-',c='y',lw=2,label='Signal $BB$')
    ax.plot(l,nsee,'--',c='k',lw=1.5,label='Noise S.A.')
    ax.plot(l,nlee,'-.',c='k',lw=1.5,label='Noise L.A.')
    ax.plot(l,ccee*ratio_s**2,'--',c='b',lw=2,label='FG, $EE$, S.A.')
    ax.plot(l,ccbb*ratio_s**2,'--',c='y',lw=2,label='FG, $BB$, S.A.')
    ax.plot(l,ccee*ratio_l**2,'-.',c='b',lw=2,label='FG, $EE$, L.A.')
    ax.plot(l,ccbb*ratio_l**2,'-.',c='y',lw=2,label='FG, $BB$, L.A.')
    ax.set_ylim([5E-10,8E-2])
    ax.set_xlim([2,lmax])
    plt.loglog()
    ax.legend(loc='lower left',ncol=2)#,frameon=False)
    tickfs(ax)
    ax.set_xlabel('$\\ell$',fontsize=15)
    ax.set_ylabel('$C_\\ell\\,\\,[\\mu K^2\\,{\\rm srad}]$',fontsize=15)
    plt.savefig("../plots_paper/cls_cmb.pdf",bbox_inches='tight')
    plt.show()
