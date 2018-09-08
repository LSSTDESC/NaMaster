from __future__ import print_function
from optparse import OptionParser
import numpy as np
import matplotlib.pyplot as plt
import pymaster as nmt
import os
import sys
import data.flatmaps as fm


DTOR=np.pi/180

def opt_callback(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))
parser = OptionParser()
parser.add_option('--isim-ini', dest='isim_ini', default=1, type=int,
                  help='Index of first simulation')
parser.add_option('--isim-end', dest='isim_end', default=100, type=int,
                  help='Index of last simulation')
parser.add_option('--wo-contaminants', dest='wo_cont', default=False, action='store_true',
                  help='Set if you don\'t want to use contaminants')
parser.add_option('--plot', dest='plot_stuff', default=False, action='store_true',
                  help='Set if you want to produce plots')
parser.add_option('--wo-pureb', dest='wo_pureb', default=False, action='store_true',
                  help='Set if you don\'t want to purify B-modes')
(o, args) = parser.parse_args()

nsims=o.isim_end-o.isim_ini+1
w_cont=not o.wo_cont
w_pureb=not o.wo_pureb

#Create output directory
predir="tests_flat"
os.system("mkdir -p "+predir)
prefix=predir+"/run_pure0%d_cont%d"%(w_pureb,w_cont)

#Read theory power spectra
def read_cl_camb(fname) :
    data=np.loadtxt(fname,unpack=True)
    ll=np.arange(len(data[0])+2)+0.
    fac=2*np.pi/(ll[2:]*(ll[2:]+1.))
    cl_tt=np.zeros_like(ll); cl_tt[2:]=data[1,:]*fac
    cl_ee=np.zeros_like(ll); cl_ee[2:]=data[2,:]*fac
    cl_bb=np.zeros_like(ll); cl_bb[2:]=data[3,:]*fac
    cl_te=np.zeros_like(ll); cl_te[2:]=data[4,:]*fac

    return ll,cl_tt,cl_ee,cl_bb,cl_te
l,cltt,clee,clbb,clte=read_cl_camb("data/cls_cmb.txt")
#Noise power spectrum
nlev=(0.5*np.pi/(180*60))**2 #1 uK-arcmin noise level
nltt=nlev*(np.ones_like(l)+(300./(l+50.))**2.4) #1/ell noise with a knee scale of ell=300 (optimistic)
nlee=2*nltt; nlbb=2*nltt; nlte=0*nltt
#Beam
fwhm_amin=1.4 #Corresponding to 6m aperture at 90GHz
beam=np.exp(-0.5*l*(l+1)*(fwhm_amin*np.pi/(180*60*2.355))**2)

#Read mask
fmi,mask=fm.read_flat_map("data/mask_cmb_flat.fits")

#Read contaminant maps
if w_cont :
    fgp=np.zeros([1,2,len(mask)])
    dum,[fgp[0,0,:],fgp[0,1,:]]=fm.read_flat_map("data/cont_cmb_flat.fits",i_map=-1) #Foregrounds
    fgp[0,0,:]=fmi.smooth_map(fgp[0,0,:],l,beam)
    fgp[0,1,:]=fmi.smooth_map(fgp[0,1,:],l,beam)

#Binning scheme
ell_min=max(2*np.pi/fmi.lx_rad,2*np.pi/fmi.ly_rad)
ell_max=min(fmi.nx*np.pi/fmi.lx_rad,fmi.ny*np.pi/fmi.ly_rad)
fsky=np.sum(mask)*fmi.lx_rad*fmi.ly_rad/(4*np.pi*fmi.nx*fmi.ny)
d_ell=0.65/fsky
n_ell=int((ell_max-ell_min)/d_ell)-1
l_bpw=np.zeros([2,n_ell])
l_bpw[0,:]=ell_min+np.arange(n_ell)*d_ell
l_bpw[1,:]=l_bpw[0,:]+d_ell
b=nmt.NmtBinFlat(l_bpw[0,:],l_bpw[1,:])

#Generate some initial fields
print(" - Res(x): %.3lf arcmin. Res(y): %.3lf arcmin."%(fmi.lx*60/fmi.nx,fmi.ly*60/fmi.ny))
print(" - lmax = %d, lmin = %d"%(int(ell_max),int(ell_min)))
def get_fields() :
    st,sq,su=nmt.synfast_flat(int(fmi.nx),int(fmi.ny),fmi.lx_rad,fmi.ly_rad,
                              np.array([cltt*beam**2+nltt,clte*beam**2+nlte,0*cltt,
                                        clee*beam**2+nlee,0*clee,
                                        clbb*beam**2+nlbb]),[0,2])
    st=st.flatten(); sq=sq.flatten(); su=su.flatten()
    if w_cont :
        sq+=np.sum(fgp,axis=0)[0,:]; su+=np.sum(fgp,axis=0)[1,:];
        ff2=nmt.NmtFieldFlat(fmi.lx_rad,fmi.ly_rad,mask.reshape([fmi.ny,fmi.nx]),
                             [sq.reshape([fmi.ny,fmi.nx]),su.reshape([fmi.ny,fmi.nx])],
                             templates=fgp.reshape([1,2,fmi.ny,fmi.nx]),beam=[l,beam],
                             purify_e=False,purify_b=w_pureb)
    else :
        ff2=nmt.NmtFieldFlat(fmi.lx_rad,fmi.ly_rad,mask.reshape([fmi.ny,fmi.nx]),
                             [sq.reshape([fmi.ny,fmi.nx]),su.reshape([fmi.ny,fmi.nx])],
                             beam=[l,beam],purify_e=False,purify_b=w_pureb)
    return ff2

np.random.seed(1000)
print("Fielding")
f2=get_fields()

#Use initial fields to generate coupling matrix
w22=nmt.NmtWorkspaceFlat();
if not os.path.isfile(prefix+"_w22.dat") :
    print("Computing 22")
    w22.compute_coupling_matrix(f2,f2,b)
    w22.write_to(prefix+"_w22.dat");
else :
    w22.read_from(prefix+"_w22.dat")

#Generate theory prediction
if not os.path.isfile(prefix+'_cl_th.txt') :
    print("Computing theory prediction")
    cl22_th=w22.decouple_cell(w22.couple_cell(l,np.array([clee,0*clee,0*clbb,clbb])))
    np.savetxt(prefix+"_cl_th.txt",
               np.transpose([b.get_effective_ells(),cl22_th[0],cl22_th[1],cl22_th[2],cl22_th[3]]))
else :
    cl22_th=np.zeros([4,b.get_n_bands()])
    dum,cl22_th[0],cl22_th[1],cl22_th[2],cl22_th[3]=np.loadtxt(prefix+"_cl_th.txt",unpack=True)

#Compute noise and deprojection bias
if not os.path.isfile(prefix+"_clb22.npy") :
    print("Computing deprojection and noise bias 22")
    #Compute noise bias
    clb22=w22.couple_cell(l,np.array([nlee/beam**2,0*nlee,0*nlbb,nlbb/beam**2]))
    #Compute deprojection bias
    print("Deproj")
    if w_cont :
        clb22+=nmt.deprojection_bias_flat(f2,f2,b,l,[clee*beam**2+nlee,0*clee,0*clbb,clbb*beam**2+nlbb])
    np.save(prefix+"_clb22",clb22)
else :
    clb22=np.load(prefix+"_clb22.npy")

#Compute mean and variance over nsims simulations
cl22_all=[]
for i in np.arange(nsims) :
    #if i%100==0 :
    print("%d-th sim"%(i+o.isim_ini))

    if not os.path.isfile(prefix+"_cl_%04d.txt"%(o.isim_ini+i)) :
        np.random.seed(1000+o.isim_ini+i)
        f2=get_fields()
        cl22=w22.decouple_cell(nmt.compute_coupled_cell_flat(f2,f2,b),cl_bias=clb22)
        np.savetxt(prefix+"_cl_%04d.txt"%(o.isim_ini+i),
                   np.transpose([b.get_effective_ells(),cl22[0],cl22[1],cl22[2],cl22[3]]))
    cld=np.loadtxt(prefix+"_cl_%04d.txt"%(o.isim_ini+i),unpack=True)
    cl22_all.append([cld[1],cld[2],cld[3],cld[4]])
cl22_all=np.array(cl22_all)

#Plot results
if o.plot_stuff :
    import scipy.stats as st
    
    def tickfs(ax,x=True,y=True) :
        if x :
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(12)
        if y :
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(12)
    
    l_eff=b.get_effective_ells()
    cols=plt.cm.rainbow(np.linspace(0,1,3))
    hartfac=(nsims-len(l_eff)-2.)/(nsims-1.)
    plt.figure()
    ax=plt.gca()
    mean=np.mean(cl22_all,axis=0)[0]; th=cl22_th[0]
    cov=(np.mean(cl22_all[:,0,:,None]*cl22_all[:,0,None,:],axis=0)-mean[None,:]*mean[:,None])/nsims
    chi2=np.dot(mean-th,np.linalg.solve(cov,mean-th))*hartfac
    print('EE: %.1lf %d %.3lE'%(chi2,len(th),1-st.chi2.cdf(chi2,len(th))))
    std=np.std(cl22_all,axis=0)[0]/np.sqrt(nsims+0.)
    ax.errorbar(l_eff  ,(mean-th)/std,yerr=std/std,
                label='$EE$',fmt='bo')
    mean=np.mean(cl22_all,axis=0)[1]; th=cl22_th[1]
    cov=(np.mean(cl22_all[:,1,:,None]*cl22_all[:,1,None,:],axis=0)-mean[None,:]*mean[:,None])/nsims
    chi2=np.dot(mean-th,np.linalg.solve(cov,mean-th))*hartfac
    print('EB: %.1lf %d %.3lE'%(chi2,len(th),1-st.chi2.cdf(chi2,len(th))))
    std=np.std(cl22_all,axis=0)[1]/np.sqrt(nsims+0.)
    ax.errorbar(l_eff+4,(mean-th)/std,yerr=std/std,
                label='$EB$',fmt='bs')
    mean=np.mean(cl22_all,axis=0)[3]; th=cl22_th[3]
    cov=(np.mean(cl22_all[:,3,:,None]*cl22_all[:,3,None,:],axis=0)-mean[None,:]*mean[:,None])/nsims
    chi2=np.dot(mean-th,np.linalg.solve(cov,mean-th))*hartfac
    print('BB: %.1lf %d %.3lE'%(chi2,len(th),1-st.chi2.cdf(chi2,len(th))))
    std=np.std(cl22_all,axis=0)[3]/np.sqrt(nsims+0.)
    ax.errorbar(l_eff+8,(mean-th)/std,yerr=std/std,
                label='$BB$',fmt='bx')
    ax.set_xlabel('$\\ell$',fontsize=15)
    ax.set_ylabel('$\\Delta C_\\ell/\\sigma_\\ell$',fontsize=15)
    ax.set_ylim([-6,6])
    ax.legend(loc='upper left',frameon=False,fontsize=15,ncol=2,labelspacing=0.1)
    tickfs(ax)
    ax.set_xlim([2,5000])
    plt.savefig(prefix+'_celldiff.png',bbox_inches='tight')
    plt.savefig(prefix+'_celldiff.pdf',bbox_inches='tight')

    import scipy.stats as st
    bins_use=np.where(l_eff<5000)[0]; ndof=len(bins_use)
    #Nsims, ncl, nell
    cl22_mean=np.mean(cl22_all,axis=0)
    dcl=(cl22_all[:,:,bins_use]-cl22_mean[None,:,bins_use]).reshape([nsims,4*ndof])
    res=(cl22_all[:,:,bins_use]-cl22_th[None,:,bins_use]).reshape([nsims,4*ndof])
    covar=np.mean(res[:,:,None]*res[:,None,:],axis=0)
    plt.figure()
    plt.title('BB correlation matrix')
    corr_toplot=(covar/np.sqrt(np.diag(covar)[:,None]*np.diag(covar)[None,:]))[3*ndof:,:][:,3*ndof:]
    plt.imshow(corr_toplot,interpolation='nearest')
    plt.xlabel('$\\ell_1$',fontsize=16)
    plt.ylabel('$\\ell_2$',fontsize=16)
    plt.savefig(prefix+'_covarbb.png',bbox_inches='tight')
    plt.savefig(prefix+'_covarbb.pdf',bbox_inches='tight')
    chi2_22=np.transpose(np.array([np.sum(res[:,i*ndof:(i+1)*ndof]*
                                          np.sum(np.linalg.inv(covar[i*ndof:(i+1)*ndof,:][:,i*ndof:(i+1)*ndof])[None,:,:]*
                                                 res[:,i*ndof:(i+1)*ndof,None],axis=1),axis=1)
                                   for i in np.arange(4)]))

    x=np.linspace(ndof-5*np.sqrt(2.*ndof),ndof+5*np.sqrt(2*ndof),256)
    pdf=st.chi2.pdf(x,ndof)

    plt.figure(figsize=(10,4))
    ax=[plt.subplot(1,3,i+1) for i in range(3)]
    plt.subplots_adjust(wspace=0, hspace=0)

    h,b,p=ax[0].hist(chi2_22[:,0],bins=40,density=True)
    ax[0].text(0.75,0.9,'$EE$',transform=ax[0].transAxes)
    ax[0].set_xlabel('$\\chi^2$')
    ax[0].set_ylabel('$P(\\chi^2)$')

    h,b,p=ax[1].hist(chi2_22[:,1],bins=40,density=True)
    ax[1].text(0.75,0.9,'$EB$',transform=ax[1].transAxes)

    h,b,p=ax[2].hist(chi2_22[:,3],bins=40,density=True)
    ax[2].text(0.75,0.9,'$BB$',transform=ax[2].transAxes)

    for a in ax :
        a.set_xlabel('$\\chi^2$')
    ax[1].set_yticklabels([])
    ax[2].set_yticklabels([])
    for a in ax :
        a.set_xlim([ndof-5*np.sqrt(2.*ndof),ndof+5*np.sqrt(2.*ndof)])
        a.set_ylim([0,1.4*np.amax(pdf)])
        a.plot([ndof,ndof],[0,1.4*np.amax(pdf)],'k--',label='$N_{\\rm dof}$')
        a.plot(x,pdf,'k-',label='$P(\\chi^2,N_{\\rm dof})$')
    ax[0].legend(loc='upper left',frameon=False)
    plt.savefig(prefix+'_distributions.png',bbox_inches='tight')
    plt.savefig(prefix+'_distributions.pdf',bbox_inches='tight')

    ic=0
    plt.figure()
    plt.plot(l_eff,np.mean(cl22_all,axis=0)[0],'.',
             label='$EE$',c=cols[ic]);
    plt.plot(l_eff,cl22_th[0],'--',c=cols[ic]); ic+=1
    plt.plot(l_eff,np.mean(cl22_all,axis=0)[1],'.',
             label='$EB$',c=cols[ic]); ic+=1
    plt.plot(l_eff,np.mean(cl22_all,axis=0)[3],'.',
             label='$BB$',c=cols[ic]);
    plt.plot(l_eff,cl22_th[3],'--',c=cols[ic]); ic+=1
    plt.yscale('log')
    plt.xlim([2,5000])
    plt.xlabel('$\\ell$',fontsize=16)
    plt.ylabel('$C_\\ell$',fontsize=16)
    plt.legend(loc='lower left',frameon=False,fontsize=14,ncol=2)
    plt.savefig(prefix+'_cellfull.png',bbox_inches='tight')
    plt.savefig(prefix+'_cellfull.pdf',bbox_inches='tight')
    plt.show()
