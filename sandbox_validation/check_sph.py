from __future__ import print_function
from optparse import OptionParser
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt
import os
import sys

DTOR=np.pi/180

def opt_callback(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))
parser = OptionParser()
parser.add_option('--nside', dest='nside_out', default=512, type=int,
                  help='Resolution parameter')
parser.add_option('--isim-ini', dest='isim_ini', default=1, type=int,
                  help='Index of first simulation')
parser.add_option('--isim-end', dest='isim_end', default=100, type=int,
                  help='Index of last simulation')
parser.add_option('--wo-mask', dest='wo_mask', default=False, action='store_true',
                  help='Set if you don\'t want to use a mask')
parser.add_option('--wo-contaminants', dest='wo_cont', default=False, action='store_true',
                  help='Set if you don\'t want to use contaminants')
parser.add_option('--wo-varnoise', dest='wo_nvar', default=False, action='store_true',
                  help='Set if you don\'t want to use inhomogeneous noise')
parser.add_option('--plot', dest='plot_stuff', default=False, action='store_true',
                  help='Set if you want to produce plots')
parser.add_option('--aposize', dest='aposize', default=0.0, type=float,
                  help='Mask apodization (in degrees)')
(o, args) = parser.parse_args()

nsims=o.isim_end-o.isim_ini+1
w_mask=not o.wo_mask
w_cont=not o.wo_cont
w_nvar=not o.wo_nvar

#Switch off contaminants and inhomogeneous noiseif there's no mask
if not w_mask :
    w_cont=False
    w_nvar=False

#Create output directory
predir="tests_sph"
os.system("mkdir -p "+predir)
prefix=predir+"/run_ns%d_mask%d_cont%d_nvar%d_apo%.2lf"%(o.nside_out,w_mask,w_cont,w_nvar,o.aposize)
fname_mask=prefix+"_mask"

#Read theory power spectra
l,cltt,clee,clbb,clte,nltt,nlee,nlbb,nlte=np.loadtxt("data/cls_lss.txt",unpack=True)
cltt=cltt[:3*o.nside_out]; clee=clee[:3*o.nside_out]; clbb=clbb[:3*o.nside_out];
clte=clte[:3*o.nside_out]; 
nltt=nltt[:3*o.nside_out]; nlee=nlee[:3*o.nside_out]; nlbb=nlbb[:3*o.nside_out];
nlte=nlte[:3*o.nside_out]; 

#Read noise variance map
if w_nvar :
    depth_nvar=hp.read_map("data/cont_lss_nvar_ns%d.fits"%o.nside_out,verbose=False)
    depth_nvar[depth_nvar<0.8]=0
else :
    depth_nvar=np.ones(hp.nside2npix(o.nside_out))
pixel_area=4*np.pi/hp.nside2npix(o.nside_out)
#Transform variance in sterad to variance in pix
depth_nvar_t=nltt[-1]*depth_nvar/pixel_area
depth_nvar_p=nlee[-1]*depth_nvar/pixel_area

#Generate mask
if not os.path.isfile(fname_mask+'.fits') :    
    if w_mask :
        depth_ivar=np.zeros_like(depth_nvar); depth_ivar[depth_nvar>0.1]=1./depth_nvar[depth_nvar>0.1]
        mask_raw=hp.read_map("data/mask_lss_ns%d.fits"%o.nside_out,verbose=False)
        if o.aposize>0 :
            mask=nmt.mask_apodization(mask_raw,o.aposize,apotype='C1')
        else :
            mask=mask_raw
        mask*=depth_ivar
    else :
        mask=np.ones(hp.nside2npix(o.nside_out))

    hp.write_map(fname_mask+".fits",mask,overwrite=True)
mask=hp.read_map(fname_mask+".fits",verbose=False)
if o.plot_stuff :
    hp.mollview(mask)
fsky=np.mean(mask/np.amax(mask));

#Read contaminant maps
if w_cont :
    fgt=np.zeros([2,1,hp.nside2npix(o.nside_out)])
    fgt[0,0,:]=hp.read_map("data/cont_lss_star_ns%d.fits"%o.nside_out,verbose=False); #Stars
    fgt[1,0,:]=hp.read_map("data/cont_lss_dust_ns%d.fits"%o.nside_out,verbose=False); #Dust
    fgp=np.zeros([2,2,hp.nside2npix(o.nside_out)]);
    fgp[0,0,:],fgp[0,1,:]=hp.read_map("data/cont_wl_psf_ns%d.fits"%o.nside_out,
                                      field=[0,1],verbose=False); #PSF
    fgp[1,0,:],fgp[1,1,:]=hp.read_map("data/cont_wl_ss_ns%d.fits"%o.nside_out,
                                      field=[0,1],verbose=False); #Small-scales
    if o.plot_stuff :
        hp.mollview(np.sum(fgt,axis=0)[0,:]*mask)
        hp.mollview(np.sum(fgp,axis=0)[0,:]*mask)
        hp.mollview(np.sum(fgp,axis=0)[1,:]*mask)

#Binning scheme
d_ell=int(1./fsky)
b=nmt.NmtBin(o.nside_out,nlb=d_ell)

#Generate some initial fields
print(" - Res: %.3lf arcmin. "%(np.sqrt(4*np.pi*(180*60/np.pi)**2/hp.nside2npix(o.nside_out))))
def get_fields() :
    #Signal
    st,sq,su=hp.synfast([cltt,clee,clbb,clte],o.nside_out,new=True,verbose=False,pol=True)
    #Inhomogeneous white noise
    nt=np.sqrt(depth_nvar_t)*np.random.randn(hp.nside2npix(o.nside_out))
    nq=np.sqrt(depth_nvar_p)*np.random.randn(hp.nside2npix(o.nside_out))
    nu=np.sqrt(depth_nvar_p)*np.random.randn(hp.nside2npix(o.nside_out))
    st+=nt; sq+=nq; su+=nu;
    #Contaminants
    if w_cont :
        st+=np.sum(fgt,axis=0)[0,:]; sq+=np.sum(fgp,axis=0)[0,:]; su+=np.sum(fgp,axis=0)[1,:]
        ff0=nmt.NmtField(mask,[st],templates=fgt)
        ff2=nmt.NmtField(mask,[sq,su],templates=fgp)
    else :
        ff0=nmt.NmtField(mask,[st])
        ff2=nmt.NmtField(mask,[sq,su])
    return ff0,ff2
np.random.seed(1000)
f0,f2=get_fields()
    
if o.plot_stuff :
    hp.mollview(f0.get_maps()[0]*mask,title='$\\delta_g$')
    hp.mollview(f2.get_maps()[0]*mask,title='$\\gamma_1$')
    hp.mollview(f2.get_maps()[1]*mask,title='$\\gamma_2$')

#Use initial fields to generate coupling matrix
w00=nmt.NmtWorkspace();
if not os.path.isfile(prefix+"_w00.dat") :
    print("Computing 00")
    w00.compute_coupling_matrix(f0,f0,b)
    w00.write_to(prefix+"_w00.dat");
else :
    w00.read_from(prefix+"_w00.dat")
w02=nmt.NmtWorkspace();
if not os.path.isfile(prefix+"_w02.dat") :
    print("Computing 02")
    w02.compute_coupling_matrix(f0,f2,b)
    w02.write_to(prefix+"_w02.dat");
else :
    w02.read_from(prefix+"_w02.dat")
w22=nmt.NmtWorkspace();
if not os.path.isfile(prefix+"_w22.dat") :
    print("Computing 22")
    w22.compute_coupling_matrix(f2,f2,b)
    w22.write_to(prefix+"_w22.dat");
else :
    w22.read_from(prefix+"_w22.dat")

#Generate theory prediction
cl00_th=w00.decouple_cell(w00.couple_cell([cltt]))
cl02_th=w02.decouple_cell(w02.couple_cell([clte,0*clte]))
cl22_th=w22.decouple_cell(w22.couple_cell([clee,0*clee,0*clbb,clbb]))
np.savetxt(prefix+"_cl_th.txt",
           np.transpose([b.get_effective_ells(),cl00_th[0],cl02_th[0],cl02_th[1],
                         cl22_th[0],cl22_th[1],cl22_th[2],cl22_th[3]]))

#Compute noise and deprojection bias
if not os.path.isfile(prefix+"_clb00.npy") :
    print("Computing deprojection and noise bias 00")
    #Compute noise bias
    clb00=np.sum(depth_nvar_t*mask*mask*pixel_area*pixel_area)/(4*np.pi)*np.ones([1,3*o.nside_out])
    #Compute deprojection bias
    if w_cont :
        #Signal contribution
        clb00+=nmt.deprojection_bias(f0,f0,[cltt])
        #Noise contribution
        clb00+=nmt.uncorr_noise_deprojection_bias(f0,depth_nvar_t*pixel_area)
    np.save(prefix+"_clb00",clb00)
else :
    clb00=np.load(prefix+"_clb00.npy")
if not os.path.isfile(prefix+"_clb02.npy") :
    print("Computing deprojection and noise bias 02")
    clb02=np.zeros([2,3*o.nside_out])
    if w_cont :
        clb02+=nmt.deprojection_bias(f0,f2,[clte,0*clte])
    np.save(prefix+"_clb02",clb02)
else :
    clb02=np.load(prefix+"_clb02.npy")
if not os.path.isfile(prefix+"_clb22.npy") :
    print("Computing deprojection and noise bias 22")
    clb22=np.sum(depth_nvar_p*mask*mask*pixel_area*pixel_area)/(4*np.pi)*np.array([1,0,0,1])[:,None]*np.ones(3*o.nside_out)[None,:]
    if w_cont :
        clb22+=nmt.deprojection_bias(f2,f2,[clee,0*clee,0*clbb,clbb])
        clb22+=nmt.uncorr_noise_deprojection_bias(f2,depth_nvar_p*pixel_area)
    np.save(prefix+"_clb22",clb22)
else :
    clb22=np.load(prefix+"_clb22.npy")
    
#Compute mean and variance over nsims simulations
cl00_all=[]
cl02_all=[]
cl22_all=[]
for i in np.arange(nsims) :
    #if i%100==0 :
    print("%d-th sim"%(i+o.isim_ini))

    if not os.path.isfile(prefix+"_cl_%04d.txt"%(o.isim_ini+i)) :
        np.random.seed(1000+o.isim_ini+i)
        f0,f2=get_fields()
        cl00=w00.decouple_cell(nmt.compute_coupled_cell(f0,f0),cl_bias=clb00)
        cl02=w02.decouple_cell(nmt.compute_coupled_cell(f0,f2),cl_bias=clb02)
        cl22=w22.decouple_cell(nmt.compute_coupled_cell(f2,f2),cl_bias=clb22)
        np.savetxt(prefix+"_cl_%04d.txt"%(o.isim_ini+i),
                   np.transpose([b.get_effective_ells(),cl00[0],cl02[0],cl02[1],
                                 cl22[0],cl22[1],cl22[2],cl22[3]]))
    cld=np.loadtxt(prefix+"_cl_%04d.txt"%(o.isim_ini+i),unpack=True)
    cl00_all.append([cld[1]])
    cl02_all.append([cld[2],cld[3]])
    cl22_all.append([cld[4],cld[5],cld[6],cld[7]])
cl00_all=np.array(cl00_all)
cl02_all=np.array(cl02_all)
cl22_all=np.array(cl22_all)

#Plot results
if o.plot_stuff :
    l_eff=b.get_effective_ells()
    cols=plt.cm.rainbow(np.linspace(0,1,6))
    plt.figure()
    plt.errorbar(l_eff,np.mean(cl00_all,axis=0)[0]/cl00_th[0]-1,
                 yerr=np.std(cl00_all,axis=0)[0]/cl00_th[0]/np.sqrt(nsims+0.),
                 label='$\\delta\\times\\delta$',fmt='ro')
    plt.errorbar(l_eff,np.mean(cl02_all,axis=0)[0]/cl02_th[0]-1,
                 yerr=np.std(cl02_all,axis=0)[0]/cl02_th[0]/np.sqrt(nsims+0.),
                 label='$\\delta\\times\\gamma_E$',fmt='go')
    plt.errorbar(l_eff,np.mean(cl22_all,axis=0)[0]/cl22_th[0]-1,
                 yerr=np.std(cl22_all,axis=0)[0]/cl22_th[0]/np.sqrt(nsims+0.),
                 label='$\\gamma_E\\times\\gamma_E$',fmt='bo')
    plt.xlabel('$\\ell$',fontsize=16)
    plt.ylabel('$\\Delta C_\\ell/C_\\ell$',fontsize=16)
    plt.xlim([2,2*o.nside_out])
    plt.legend(loc='lower right',frameon=False,fontsize=16)
    plt.xscale('log')
    plt.savefig(prefix+'_celldiff.png',bbox_inches='tight')
    plt.savefig(prefix+'_celldiff.pdf',bbox_inches='tight')

    import scipy.stats as st
    bins_use=np.where(l_eff<2*o.nside_out)[0]; ndof=len(bins_use)
    res=(cl00_all[:,:,:]-cl00_th[None,:,:])/np.std(cl00_all,axis=0)[None,:,:]
    chi2_00=np.sum(res[:,:,bins_use]**2,axis=2)
    res=(cl02_all[:,:,:]-cl02_th[None,:,:])/np.std(cl02_all,axis=0)[None,:,:]
    chi2_02=np.sum(res[:,:,bins_use]**2,axis=2)
    res=(cl22_all[:,:,:]-cl22_th[None,:,:])/np.std(cl22_all,axis=0)[None,:,:]
    chi2_22=np.sum(res[:,:,bins_use]**2,axis=2)

    x=np.linspace(ndof-5*np.sqrt(2.*ndof),ndof+5*np.sqrt(2*ndof),256)
    pdf=st.chi2.pdf(x,ndof)

    
    plt.figure(figsize=(10,7))
    ax=[plt.subplot(2,3,i+1) for i in range(6)]
    plt.subplots_adjust(wspace=0, hspace=0)
    
    h,b,p=ax[0].hist(chi2_00[:,0],bins=40,density=True)
    ax[0].text(0.75,0.9,'$\\delta\\times\\delta$'    ,transform=ax[0].transAxes)
    ax[0].set_ylabel('$P(\\chi^2)$')

    h,b,p=ax[1].hist(chi2_02[:,0],bins=40,density=True)
    ax[1].text(0.75,0.9,'$\\delta\\times\\gamma_E$'  ,transform=ax[1].transAxes)

    h,b,p=ax[2].hist(chi2_02[:,1],bins=40,density=True)
    ax[2].text(0.75,0.9,'$\\delta\\times\\gamma_B$'  ,transform=ax[2].transAxes)

    h,b,p=ax[3].hist(chi2_22[:,0],bins=40,density=True)
    ax[3].text(0.75,0.9,'$\\gamma_E\\times\\gamma_E$',transform=ax[3].transAxes)
    ax[3].set_xlabel('$\\chi^2$')
    ax[3].set_ylabel('$P(\\chi^2)$')

    h,b,p=ax[4].hist(chi2_22[:,1],bins=40,density=True)
    ax[4].text(0.75,0.9,'$\\gamma_E\\times\\gamma_B$',transform=ax[4].transAxes)

    h,b,p=ax[5].hist(chi2_22[:,3],bins=40,density=True)
    ax[5].text(0.75,0.9,'$\\gamma_B\\times\\gamma_B$',transform=ax[5].transAxes)

    for a in ax[:3] :
        a.set_xticklabels([])
    for a in ax[3:] :
        a.set_xlabel('$\\chi^2$')
    ax[1].set_yticklabels([])
    ax[2].set_yticklabels([])
    ax[4].set_yticklabels([])
    ax[5].set_yticklabels([])
    for a in ax :
        a.set_xlim([ndof-5*np.sqrt(2.*ndof),ndof+5*np.sqrt(2.*ndof)])
        a.set_ylim([0,1.4*np.amax(pdf)])
        a.plot([ndof,ndof],[0,1.4*np.amax(pdf)],'k--',label='$N_{\\rm dof}$')
        a.plot(x,pdf,'k-',label='$P(\\chi^2,N_{\\rm dof})$')
    ax[3].legend(loc='upper left',frameon=False)
    plt.savefig(prefix+'_distributions.png',bbox_inches='tight')
    plt.savefig(prefix+'_distributions.pdf',bbox_inches='tight')

    ic=0
    plt.figure()
    plt.plot(l_eff,np.mean(cl00_all,axis=0)[0],
             label='$\\delta\\times\\delta$',c=cols[ic])
    plt.plot(l_eff,cl00_th[0],'--',c=cols[ic]); ic+=1
    plt.plot(l_eff,np.mean(cl02_all,axis=0)[0],
             label='$\\delta\\times\\gamma_E$',c=cols[ic]);
    plt.plot(l_eff,cl02_th[0],'--',c=cols[ic]); ic+=1
    plt.plot(l_eff,np.mean(cl02_all,axis=0)[1],
             label='$\\delta\\times\\gamma_B$',c=cols[ic]); ic+=1
    plt.plot(l_eff,np.mean(cl22_all,axis=0)[0],
             label='$\\gamma\\times\\gamma_E$',c=cols[ic]);
    plt.plot(l_eff,cl22_th[0],'--',c=cols[ic]); ic+=1
    plt.plot(l_eff,np.mean(cl22_all,axis=0)[1],
             label='$\\gamma_E\\times\\gamma_B$',c=cols[ic]); ic+=1
    plt.plot(l_eff,np.mean(cl22_all,axis=0)[3],
             label='$\\gamma_B\\times\\gamma_B$',c=cols[ic]); ic+=1
    plt.loglog()
    plt.xlim([2,2*o.nside_out])
    plt.xlabel('$\\ell$',fontsize=16)
    plt.ylabel('$C_\\ell$',fontsize=16)
    plt.legend(loc='lower left',frameon=False,fontsize=14,ncol=2)
    plt.savefig(prefix+'_cellfull.png',bbox_inches='tight')
    plt.savefig(prefix+'_cellfull.pdf',bbox_inches='tight')
    plt.show()
