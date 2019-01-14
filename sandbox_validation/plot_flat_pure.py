import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from matplotlib import rc
import matplotlib
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

nsims=1000
prefix_clean="tests_flatb/run_pure01_cont0"

def tickfs(ax,x=True,y=True) :
    if x :
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(12)
    if y :
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(12)

def read_cls(fname) :
    l,cee,ceb,cbe,cbb=np.loadtxt(fname,unpack=True);
    return l,cee,ceb,cbe,cbb
l_th,clEE_th,clEB_th,clBE_th,clBB_th=read_cls(prefix_clean+"_cl_th.txt")
ndof=len(l_th)

print("Reading")
clEE_clean=[]; clEB_clean=[]; clBB_clean=[];
for i in np.arange(nsims) :
    ll,ccee,cceb,ccbe,ccbb=read_cls(prefix_clean+"_cl_%04d.txt"%(i+1))
    clEE_clean.append(ccee); clEB_clean.append(cceb); clBB_clean.append(ccbb);
clEE_clean=np.array(clEE_clean); clEB_clean=np.array(clEB_clean); clBB_clean=np.array(clBB_clean); 

print("Computing statistics")
hartfac=(nsims-ndof-2.)/(nsims-1.)
def compute_stats(y,y_th) :
    mean=np.mean(y,axis=0)
    cov=np.mean(y[:,:,None]*y[:,None,:],axis=0)-mean[:,None]*mean[None,:]
    icov=hartfac*np.linalg.inv(cov)
    chi2_red=np.sum((mean-y_th)**2/np.diag(cov))*nsims
    chi2_all=np.sum(((y-y_th)**2)/np.diag(cov)[None,:],axis=1)

    return mean,cov,icov,chi2_red,chi2_all

clEE_clean_mean,clEE_clean_cov,clEE_clean_icov,clEE_clean_chi2r,clEE_clean_chi2all=compute_stats(clEE_clean,clEE_th)
clEB_clean_mean,clEB_clean_cov,clEB_clean_icov,clEB_clean_chi2r,clEB_clean_chi2all=compute_stats(clEB_clean,clEB_th)
clBB_clean_mean,clBB_clean_cov,clBB_clean_icov,clBB_clean_chi2r,clBB_clean_chi2all=compute_stats(clBB_clean,clBB_th)
m,cov,icov,chi2r,chi2all=compute_stats(np.vstack((clEE_clean.T,clEB_clean.T,clBB_clean.T)).T,
                                       np.vstack((clEE_th,clEB_th,clBB_th)).flatten())
print(chi2r,len(m),1-st.chi2.cdf(chi2r,len(m)))

#Plot covariance
plt.figure();
ax=plt.gca();
im=ax.imshow(cov/np.sqrt(np.diag(cov)[None,:]*np.diag(cov)[:,None]),
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
tickfs(ax)
plt.colorbar(im)
plt.savefig("plots_paper/val_covar_cmb_flat.pdf",bbox_inches='tight')

#Plot residuals
cols=plt.cm.rainbow(np.linspace(0,1,3))
fig=plt.figure()
ax=fig.add_axes((0.12,0.3,0.78,0.6))
ic=0
ax.plot(l_th,clEE_clean_mean,label='$EE$',c=cols[ic],alpha=0.5)
ax.plot(l_th,clEE_th,'--',c=cols[ic]);
ic+=1
ax.plot(l_th,clEB_clean_mean,label='$EB$',c=cols[ic],alpha=0.5)
ic+=1
ax.plot(l_th,clBB_clean_mean,label='$BB$',c=cols[ic],alpha=0.5)
ax.plot(l_th,clBB_th,'--',c=cols[ic]);
ic+=1
ax.plot([-1,-1],[-1,-1],'k-' ,label='${\\rm Sims}$')
ax.plot([-1,-1],[-1,-1],'k--',label='${\\rm Input}$')
ax.set_ylim([5E-10,2E-3])
ax.legend(loc='upper right',frameon=False,fontsize=14,ncol=2,labelspacing=0.1)
ax.set_xlim([0,5400])
ax.set_yscale('log');
tickfs(ax)
ax.set_xticks([])
ax.set_yticks([1E-9,1E-7,1E-5,1E-3])
ax.set_ylabel('$C_\\ell\\,[\\mu K^2\\,{\\rm srad}]$',fontsize=15)
ax=fig.add_axes((0.12,0.1,0.78,0.2))
ic=0
ax.errorbar(l_th  ,(clEE_clean_mean-clEE_th)*np.sqrt(nsims+0.)/np.sqrt(np.diag(clEE_clean_cov)),
            yerr=np.ones(ndof),label='$EE$',fmt='.',c=cols[ic]); ic+=1
ax.errorbar(l_th+4,(clEB_clean_mean-clEB_th)*np.sqrt(nsims+0.)/np.sqrt(np.diag(clEB_clean_cov)),
            yerr=np.ones(ndof),label='$EB$',fmt='.',c=cols[ic]); ic+=1
ax.errorbar(l_th+8,(clBB_clean_mean-clBB_th)*np.sqrt(nsims+0.)/np.sqrt(np.diag(clBB_clean_cov)),
            yerr=np.ones(ndof),label='$BB$',fmt='.',c=cols[ic]); ic+=1
ax.set_xlabel('$\\ell$',fontsize=15)
ax.set_ylabel('$\\Delta C_\\ell/\\sigma_\\ell$',fontsize=15)
ax.set_ylim([-6,6])
ax.set_xlim([0,5400])
ax.set_yticks([-4,0,4])
tickfs(ax)
plt.savefig("plots_paper/val_cl_cmb_flat.pdf",bbox_inches='tight')

#Plot chi2 dist
xr=[ndof-5*np.sqrt(2.*ndof),ndof+5*np.sqrt(2*ndof)]
x=np.linspace(xr[0],xr[1],256)
pdf=st.chi2.pdf(x,ndof)

plt.figure(figsize=(10,4))
ax=[plt.subplot(1,3,i+1) for i in range(3)]
plt.subplots_adjust(wspace=0, hspace=0)

h,b,p=ax[0].hist(clEE_clean_chi2all,bins=40,density=True,range=xr)
ax[0].text(0.8,0.9,'$EE$',transform=ax[0].transAxes,fontsize=14)
ax[0].plot([clEE_clean_chi2r,clEE_clean_chi2r],[0,1.4*np.amax(pdf)],'k-.')
ax[0].set_xlabel('$\\chi^2$',fontsize=15)
ax[0].set_ylabel('$P(\\chi^2)$',fontsize=15)
print('EE : %.3lE'%(1-st.chi2.cdf(clEE_clean_chi2r,ndof)))

h,b,p=ax[1].hist(clEB_clean_chi2all,bins=40,density=True,range=xr)
ax[1].text(0.8,0.9,'$EB$',transform=ax[1].transAxes,fontsize=14)
ax[1].plot([clEB_clean_chi2r,clEB_clean_chi2r],[0,1.4*np.amax(pdf)],'k-.')
print('EB : %.3lE'%(1-st.chi2.cdf(clEB_clean_chi2r,ndof)))

h,b,p=ax[2].hist(clBB_clean_chi2all,bins=40,density=True,range=xr)
ax[2].text(0.8,0.9,'$BB$',transform=ax[2].transAxes,fontsize=14)
ax[2].plot([clBB_clean_chi2r,clBB_clean_chi2r],[0,1.4*np.amax(pdf)],'k-.')
print('BB : %.3lE'%(1-st.chi2.cdf(clBB_clean_chi2r,ndof)))

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
plt.savefig("plots_paper/val_chi2_cmb_flat.pdf",bbox_inches='tight')
plt.show()
