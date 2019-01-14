import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from matplotlib import rc
import matplotlib
import pymaster as nmt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset,zoomed_inset_axes
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

nside=256
nsims=1000
prefix_pure="tests_sphb/run_pure01_ns%d_cont1"%nside
prefix_nopu="tests_sphb/run_pure00_ns%d_cont1"%nside
prefix_noco="tests_sphb/run_pure01_ns%d_cont0"%nside
prefix_nodb="tests_sph/run_pure01_ns%d_cont1_no_debias"%nside

def tickfs(ax,x=True,y=True) :
    if x :
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(12)
    if y :
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(12)

def read_cls(fname) :
    l,cee,ceb,cbe,cbb=np.loadtxt(fname,unpack=True);
    id_good=np.where(l<2*nside)[0]
    return l[id_good],cee[id_good],ceb[id_good],cbe[id_good],cbb[id_good]
l_th,clEE_th,clEB_th,clBE_th,clBB_th=read_cls(prefix_pure+"_cl_th.txt")
ndof=len(l_th)

print("Reading")
clEE_pure=[]; clEB_pure=[]; clBB_pure=[];
clEE_nopu=[]; clEB_nopu=[]; clBB_nopu=[];
clEE_noco=[]; clEB_noco=[]; clBB_noco=[];
clEE_nodb=[]; clEB_nodb=[]; clBB_nodb=[];
for i in np.arange(nsims) :
    ll,ccee,cceb,ccbe,ccbb=read_cls(prefix_pure+"_cl_%04d.txt"%(i+1))
    clEE_pure.append(ccee); clEB_pure.append(cceb); clBB_pure.append(ccbb);
    ll,ccee,cceb,ccbe,ccbb=read_cls(prefix_nopu+"_cl_%04d.txt"%(i+1))
    clEE_nopu.append(ccee); clEB_nopu.append(cceb); clBB_nopu.append(ccbb);
    ll,ccee,cceb,ccbe,ccbb=read_cls(prefix_noco+"_cl_%04d.txt"%(i+1))
    clEE_noco.append(ccee); clEB_noco.append(cceb); clBB_noco.append(ccbb);
    ll,ccee,cceb,ccbe,ccbb=read_cls(prefix_nodb+"_cl_%04d.txt"%(i+1))
    clEE_nodb.append(ccee); clEB_nodb.append(cceb); clBB_nodb.append(ccbb);
clEE_pure=np.array(clEE_pure); clEB_pure=np.array(clEB_pure); clBB_pure=np.array(clBB_pure); 
clEE_nopu=np.array(clEE_nopu); clEB_nopu=np.array(clEB_nopu); clBB_nopu=np.array(clBB_nopu);
clEE_noco=np.array(clEE_noco); clEB_noco=np.array(clEB_noco); clBB_noco=np.array(clBB_noco);
clEE_nodb=np.array(clEE_nodb); clEB_nodb=np.array(clEB_nodb); clBB_nodb=np.array(clBB_nodb);

print("Computing statistics")
def compute_stats(y,y_th) :
    mean=np.mean(y,axis=0)
    cov=np.mean(y[:,:,None]*y[:,None,:],axis=0)-mean[:,None]*mean[None,:]
    icov=np.linalg.inv(cov)
    chi2_red=np.dot(mean-y_th,np.dot(icov,mean-y_th))*nsims
    chi2_all=np.sum((y-y_th)*np.sum(icov[None,:,:]*(y-y_th)[:,None,:],axis=2),axis=1)

    return mean,cov,icov,chi2_red,chi2_all

clEE_pure_mean,clEE_pure_cov,clEE_pure_icov,clEE_pure_chi2r,clEE_pure_chi2all=compute_stats(clEE_pure,clEE_th)
clEB_pure_mean,clEB_pure_cov,clEB_pure_icov,clEB_pure_chi2r,clEB_pure_chi2all=compute_stats(clEB_pure,clEB_th)
clBB_pure_mean,clBB_pure_cov,clBB_pure_icov,clBB_pure_chi2r,clBB_pure_chi2all=compute_stats(clBB_pure,clBB_th)
clEE_noco_mean,clEE_noco_cov,clEE_noco_icov,clEE_noco_chi2r,clEE_noco_chi2all=compute_stats(clEE_noco,clEE_th)
clEB_noco_mean,clEB_noco_cov,clEB_noco_icov,clEB_noco_chi2r,clEB_noco_chi2all=compute_stats(clEB_noco,clEB_th)
clBB_noco_mean,clBB_noco_cov,clBB_noco_icov,clBB_noco_chi2r,clBB_noco_chi2all=compute_stats(clBB_noco,clBB_th)
clEE_nodb_mean,clEE_nodb_cov,clEE_nodb_icov,clEE_nodb_chi2r,clEE_nodb_chi2all=compute_stats(clEE_nodb,clEE_th)
clEB_nodb_mean,clEB_nodb_cov,clEB_nodb_icov,clEB_nodb_chi2r,clEB_nodb_chi2all=compute_stats(clEB_nodb,clEB_th)
clBB_nodb_mean,clBB_nodb_cov,clBB_nodb_icov,clBB_nodb_chi2r,clBB_nodb_chi2all=compute_stats(clBB_nodb,clBB_th)
m_pure,cov_pure,icov_pure,chi2r_pure,chi2all_pure=compute_stats(np.vstack((clEE_pure.T,clEB_pure.T,clBB_pure.T)).T,
                                                           np.vstack((clEE_th,clEB_th,clBB_th)).flatten())
m_noco,cov_noco,icov_noco,chi2r_noco,chi2all_noco=compute_stats(np.vstack((clEE_noco.T,clEB_noco.T,clBB_noco.T)).T,
                                                                np.vstack((clEE_th,clEB_th,clBB_th)).flatten())
print(chi2r_pure,len(m_pure),1-st.chi2.cdf(chi2r_pure,len(m_pure)))

#Plot errorbars
plt.figure()
ax=plt.gca()
ax.plot([-1,-1],[-1,-1],'k-',lw=2,label='$BB$')
ax.plot([-1,-1],[-1,-1],'k--',lw=2,label='$EB$')
ax.plot([-1,-1],[-1,-1],'k-.',lw=2,label='$EE$')
ax.plot(ll,np.std(clBB_nopu,axis=0),'b-',lw=2,label='${\\rm Standard\\,\\,PCL}$');
ax.plot(ll,np.std(clBB_pure,axis=0),'r-',lw=2,label='${\\rm Pure}\\,B$');
ax.plot(ll,np.std(clEB_nopu,axis=0),'b--',lw=2)
ax.plot(ll,np.std(clEB_pure,axis=0),'r--',lw=2)
ax.plot(ll,np.std(clEE_nopu,axis=0),'b-.',lw=2)
ax.plot(ll,np.std(clEE_pure,axis=0),'r-.',lw=2)
ax.set_xlabel('$\\ell$',fontsize=15)
ax.set_ylabel('$\\sigma(C_\\ell)\\,[\\mu K^2\\,{\\rm srad}]$',fontsize=15)
tickfs(ax)
ax.set_xlim([4,515])
ax.set_ylim([7E-8,3E-2])
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(loc='upper right',frameon=False,fontsize=14,ncol=2)
plt.savefig("plots_paper/val_sigmas_cmb_sph.pdf",bbox_inches='tight')

#Plot covariance
plt.figure();
ax=plt.gca();
im=ax.imshow(cov_pure/np.sqrt(np.diag(cov_pure)[None,:]*np.diag(cov_pure)[:,None]),
             interpolation='nearest',cmap=plt.cm.Greys);
for i in np.arange(2)+1 :
    ax.plot([i*ndof,i*ndof],[0,3*ndof],'k--',lw=1)
    ax.plot([0,3*ndof],[i*ndof,i*ndof],'k--',lw=1)
ax.set_xlim([0,3*ndof])
ax.set_ylim([3*ndof,0])
ax.set_xticks(ndof*(np.arange(3)+0.5))
ax.set_yticks(ndof*(np.arange(3)+0.5))
ax.set_xticklabels(['$EE$','$EB$','$BB$'])
ax.set_yticklabels(['$EE$','$EB$','$BB$'])
plt.colorbar(im)
axins=zoomed_inset_axes(ax,2.5,loc=6)
axins.imshow(cov_pure/np.sqrt(np.diag(cov_pure)[None,:]*np.diag(cov_pure)[:,None]),
             interpolation='nearest',cmap=plt.cm.Greys)
axins.get_xaxis().set_visible(False)
axins.get_yaxis().set_visible(False)
axins.set_xlim(0.,0.2*ndof)
axins.set_ylim(0.2*ndof,0.)
mark_inset(ax, axins, loc1=2, loc2=1, fc="none")#, ec="0.5")
tickfs(ax)
plt.savefig("plots_paper/val_covar_cmb_sph.pdf",bbox_inches='tight')

plt.figure();
ax=plt.gca();
im=ax.imshow(cov_noco/np.sqrt(np.diag(cov_noco)[None,:]*np.diag(cov_noco)[:,None]),
             interpolation='nearest',cmap=plt.cm.Greys);
for i in np.arange(2)+1 :
    ax.plot([i*ndof,i*ndof],[0,3*ndof],'k--',lw=1)
    ax.plot([0,3*ndof],[i*ndof,i*ndof],'k--',lw=1)
ax.set_xlim([0,3*ndof])
ax.set_ylim([3*ndof,0])
ax.set_xticks(ndof*(np.arange(3)+0.5))
ax.set_yticks(ndof*(np.arange(3)+0.5))
ax.set_xticklabels(['$EE$','$EB$','$BB$'])
ax.set_yticklabels(['$EE$','$EB$','$BB$'])
plt.colorbar(im)
axins=zoomed_inset_axes(ax,2.5,loc=6)
axins.imshow(cov_noco/np.sqrt(np.diag(cov_noco)[None,:]*np.diag(cov_noco)[:,None]),
             interpolation='nearest',cmap=plt.cm.Greys)
axins.get_xaxis().set_visible(False)
axins.get_yaxis().set_visible(False)
axins.set_xlim(-0.1,0.2*ndof)
axins.set_ylim(0.2*ndof,-0.1)
mark_inset(ax, axins, loc1=2, loc2=1, fc="none")#, ec="0.5")
tickfs(ax)
plt.savefig("plots_paper/val_covar_cmb_sph_nocont.pdf",bbox_inches='tight')

#Plot residuals
cols=plt.cm.rainbow(np.linspace(0,1,3))
fig=plt.figure()
ax=fig.add_axes((0.12,0.3,0.78,0.6))
ax.plot([-1,-1],[-1,-1],'k-' ,label='${\\rm Sims}$')
ax.plot([-1,-1],[-1,-1],'k--',label='${\\rm Input}$')
ic=0
ax.plot(l_th[l_th>18],clEE_pure_mean[l_th>18],label='$EE$',c=cols[ic],alpha=0.5) #Plotting above ell=18 to avoid quirky lines due to negative values
ax.plot(l_th[l_th>18],clEE_th[l_th>18],'--',c=cols[ic]);
ic+=1
ax.plot(l_th,clEB_pure_mean,label='$EB$',c=cols[ic],alpha=0.5);
ic+=1
ax.plot(l_th[l_th>10],clBB_pure_mean[l_th>10],label='$BB$',c=cols[ic],alpha=0.5);
ax.plot(l_th,np.fabs(clBB_nodb_mean),'-.',
        label='$BB,\\,\\,{\\rm no\\,\\,debias}$',c=cols[ic]);
ax.plot(l_th,clBB_th,'--',c=cols[ic]);
ic+=1
ax.set_ylim([2E-8,1.3E-2])
ax.legend(loc='upper left',frameon=False,fontsize=14,ncol=3,labelspacing=0.1)
ax.set_xlim([0,515])
ax.set_yscale('log');
tickfs(ax)
ax.set_xticks([])
ax.set_ylabel('$C_\\ell\\,[\\mu K^2\\,{\\rm srad}]$',fontsize=15)
ax=fig.add_axes((0.12,0.1,0.78,0.2))
ic=0
ax.errorbar(l_th  ,(clEE_pure_mean-clEE_th)*np.sqrt(nsims+0.)/np.sqrt(np.diag(clEE_pure_cov)),
            yerr=np.ones(ndof),label='$EE$',fmt='.',c=cols[ic]); ic+=1
ax.errorbar(l_th+2,(clEB_pure_mean-clEB_th)*np.sqrt(nsims+0.)/np.sqrt(np.diag(clEB_pure_cov)),
            yerr=np.ones(ndof),label='$EB$',fmt='.',c=cols[ic]); ic+=1
ax.errorbar(l_th+4,(clBB_pure_mean-clBB_th)*np.sqrt(nsims+0.)/np.sqrt(np.diag(clBB_pure_cov)),
            yerr=np.ones(ndof),label='$BB$',fmt='.',c=cols[ic]); ic+=1
ax.set_xlabel('$\\ell$',fontsize=15)
ax.set_ylabel('$\\Delta C_\\ell/\\sigma_\\ell$',fontsize=15)
ax.set_ylim([-6,6])
ax.set_xlim([0,515])
ax.set_yticks([-4,0,4])
tickfs(ax)
plt.savefig("plots_paper/val_cl_cmb_sph.pdf",bbox_inches='tight')

#Plot chi2 dist
xr=[ndof-5*np.sqrt(2.*ndof),ndof+5*np.sqrt(2*ndof)]
x=np.linspace(xr[0],xr[1],256)
pdf=st.chi2.pdf(x,ndof)

plt.figure(figsize=(10,4))
ax=[plt.subplot(1,3,i+1) for i in range(3)]
plt.subplots_adjust(wspace=0, hspace=0)

h,b,p=ax[0].hist(clEE_pure_chi2all,bins=40,density=True,range=xr)
ax[0].text(0.8,0.9,'$EE$',transform=ax[0].transAxes,fontsize=14)
ax[0].plot([clEE_pure_chi2r,clEE_pure_chi2r],[0,1.4*np.amax(pdf)],'k-.')
ax[0].set_xlabel('$\\chi^2$',fontsize=15)
ax[0].set_ylabel('$P(\\chi^2)$',fontsize=15)
print('EE : %.3lE'%(1-st.chi2.cdf(clEE_pure_chi2r,ndof)))

h,b,p=ax[1].hist(clEB_pure_chi2all,bins=40,density=True,range=xr)
ax[1].text(0.8,0.9,'$EB$',transform=ax[1].transAxes,fontsize=14)
ax[1].plot([clEB_pure_chi2r,clEB_pure_chi2r],[0,1.4*np.amax(pdf)],'k-.')
print('EB : %.3lE'%(1-st.chi2.cdf(clEB_pure_chi2r,ndof)))

h,b,p=ax[2].hist(clBB_pure_chi2all,bins=40,density=True,range=xr)
ax[2].text(0.8,0.9,'$BB$',transform=ax[2].transAxes,fontsize=14)
ax[2].plot([clBB_pure_chi2r,clBB_pure_chi2r],[0,1.4*np.amax(pdf)],'k-.')
print('BB : %.3lE'%(1-st.chi2.cdf(clBB_pure_chi2r,ndof)))

for a in ax :
    a.set_xlabel('$\\chi^2$',fontsize=15)
ax[1].set_yticklabels([])
ax[2].set_yticklabels([])
for a in ax :
    tickfs(a)
    a.set_xlim([ndof-5*np.sqrt(2.*ndof),ndof+5*np.sqrt(2.*ndof)])
    a.set_ylim([0,1.4*np.amax(pdf)])
    a.plot(x,pdf,'k-',label='$P(\\chi^2)$')
    a.plot([ndof,ndof],[0,1.4*np.amax(pdf)],'k--',label='$N_{\\rm dof}$')
    a.plot([-1,-1],[-1,-1],'k-.',label='$\\chi^2_{\\rm mean}$')
ax[0].legend(loc='upper left',fontsize=12,frameon=False)
plt.savefig("plots_paper/val_chi2_cmb_sph.pdf",bbox_inches='tight')

print("Computing bandpower weights")
ls=np.arange(3*nside,dtype=int)
bpws=np.zeros(3*nside,dtype=int)-1
weights=np.ones(3*nside)
bpw_edges=[2,9,17]
while bpw_edges[-1]<3*nside :
    bpw_edges.append(min(bpw_edges[-1]+12,3*nside))
bpw_edges=np.array(bpw_edges)
for ib,b0 in enumerate(bpw_edges[:-1]) :
    bpws[b0:bpw_edges[ib+1]]=ib
    weights[b0:bpw_edges[ib+1]]=1./(bpw_edges[ib+1]-b0+0.)
b=nmt.NmtBin(nside,ells=ls,bpws=bpws,weights=weights)
w=nmt.NmtWorkspace();
w.read_from(prefix_pure+"_w22.dat")
nbpw=b.get_n_bands()
wmat=np.zeros([nbpw,3*nside])
iden=np.diag(np.ones(4*3*nside))
for l in range(3*nside) :
    if l%100==0 :
        print(l,3*nside)
    wmat[:,l]=w.decouple_cell(w.couple_cell(iden[3*3*nside+l].reshape([4,3*nside])))[3]
plt.figure();
ax=plt.gca()
for ib in np.arange(nbpw) :
    ax.plot(np.arange(3*nside),wmat[ib],'k-',lw=1)
    wbin=np.zeros(3*nside); wbin[b.get_ell_list(ib)]=b.get_weight_list(ib)
    ax.plot(np.arange(3*nside),wbin,'r-',lw=1)
ax.plot([-1,-1],[-1,-1],'k-',lw=1,label='${\\rm Exact}$')
ax.plot([-1,-1],[-1,-1],'r-',lw=1,label='${\\rm Binning\\,\\,approximation}$')
ax.set_xlim([100,200])
ax.set_ylim([-0.03,0.11])
ax.set_xlabel('$\\ell$',fontsize=15)
ax.set_ylabel('${\\rm Bandpower\\,\\,windows}$',fontsize=15)
ax.legend(loc='lower left',ncol=2,frameon=False,fontsize=14)
tickfs(ax)
plt.savefig("plots_paper/val_cmb_bpw.pdf",bbox_inches='tight')
plt.show()

