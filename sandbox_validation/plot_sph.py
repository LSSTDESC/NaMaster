import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from matplotlib import rc
import matplotlib
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

nsims=1000
rebin_step=-1
prefix_clean="/global/cscratch1/sd/jsanch87/tests_sph_july_v2/run_ns1024_mask1_cont1_nvar1_apo0.00"
#prefix_dirty="/global/cscratch1/sd/jsanch87/tests_sph_v2_binary/run_ns1024_mask1_cont1_nvar1_apo0.00"
prefix_dirty="/global/cscratch1/sd/jsanch87/tests_sph_v2_no_dep_no_binary/run_ns1024_mask1_cont1_nvar1_apo0.00"
def subsample(a,n):
   return np.array([a[i:n+i].mean() for i in range(0,len(a),n)])

def tickfs(ax,x=True,y=True) :
    if x :
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(12)
    if y :
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(12)

def read_cls(fname) :
    l,ctt,cte,ctb,cee,ceb,cbe,cbb=np.loadtxt(fname,unpack=True);
    mask = l <= 2048
    return l[mask],ctt[mask],cte[mask],ctb[mask],cee[mask],ceb[mask],cbe[mask],cbb[mask]

l_th,clTT_th,clTE_th,clTB_th,clEE_th,clEB_th,clBE_th,clBB_th=read_cls(prefix_clean+"_cl_th.txt")
if rebin_step>0:
    l_th = subsample(l_th,rebin_step)
    clTT_th = subsample(clTT_th,rebin_step); clTE_th = subsample(clTE_th,rebin_step); clTB_th = subsample(clTB_th,rebin_step);
    clEE_th = subsample(clEE_th,rebin_step); clEB_th = subsample(clEB_th,rebin_step); clBB_th = subsample(clBB_th,rebin_step);

ndof=len(l_th)-1

clTT_clean=[]; clTE_clean=[]; clTB_clean=[]; clEE_clean=[]; clEB_clean=[]; clBB_clean=[];
clTT_dirty=[]; clTE_dirty=[]; clTB_dirty=[]; clEE_dirty=[]; clEB_dirty=[]; clBB_dirty=[];
for i in np.arange(nsims) :
    ll,cctt,ccte,cctb,ccee,cceb,ccbe,ccbb=read_cls(prefix_clean+"_cl_%04d.txt"%(i+1))
    if rebin_step>0:
        ll = subsample(ll,rebin_step)
        cctt = subsample(cctt,rebin_step); ccte = subsample(ccte,rebin_step); cctb = subsample(cctb,rebin_step);
        ccee = subsample(ccee,rebin_step); cceb = subsample(cceb,rebin_step); ccbb = subsample(ccbb,rebin_step);
    clTT_clean.append(cctt); clTE_clean.append(ccte); clTB_clean.append(cctb);
    clEE_clean.append(ccee); clEB_clean.append(cceb); clBB_clean.append(ccbb);
    ll,cctt,ccte,cctb,ccee,cceb,ccbe,ccbb=read_cls(prefix_dirty+"_cl_%04d.txt"%(i+1))
    if rebin_step>0:
        cctt = subsample(cctt,rebin_step); ccte = subsample(ccte,rebin_step); cctb = subsample(cctb,rebin_step);
        ccee = subsample(ccee,rebin_step); cceb = subsample(cceb,rebin_step); ccbb = subsample(ccbb,rebin_step);
    clTT_dirty.append(cctt); clTE_dirty.append(ccte); clTB_dirty.append(cctb);
    clEE_dirty.append(ccee); clEB_dirty.append(cceb); clBB_dirty.append(ccbb);
clTT_clean=np.array(clTT_clean); clTE_clean=np.array(clTE_clean); clTB_clean=np.array(clTB_clean); 
clEE_clean=np.array(clEE_clean); clEB_clean=np.array(clEB_clean); clBB_clean=np.array(clBB_clean); 
clTT_dirty=np.array(clTT_dirty); clTE_dirty=np.array(clTE_dirty); clTB_dirty=np.array(clTB_dirty); 
clEE_dirty=np.array(clEE_dirty); clEB_dirty=np.array(clEB_dirty); clBB_dirty=np.array(clBB_dirty);

def compute_stats(y,y_th) :
    mean=np.mean(y,axis=0)
    cov=np.mean(y[:,:,None]*y[:,None,:],axis=0)-mean[:,None]*mean[None,:]
    icov=np.linalg.inv(cov)
    #chi2_red=np.dot(mean-y_th,np.dot(icov,mean-y_th))*nsims
    chi2_red=np.sum((mean-y_th)**2/np.diag(cov))*nsims
    #chi2_all=np.sum((y-y_th)*np.sum(icov[None,:,:]*(y-y_th)[:,None,:],axis=2),axis=1)
    chi2_all=np.sum(((y-y_th)**2)/np.diag(cov)[None,:],axis=1)

    return mean,cov,icov,chi2_red,chi2_all

clTT_clean_mean,clTT_clean_cov,clTT_clean_icov,clTT_clean_chi2r,clTT_clean_chi2all=compute_stats(clTT_clean,clTT_th)
clTE_clean_mean,clTE_clean_cov,clTE_clean_icov,clTE_clean_chi2r,clTE_clean_chi2all=compute_stats(clTE_clean,clTE_th)
clTB_clean_mean,clTB_clean_cov,clTB_clean_icov,clTB_clean_chi2r,clTB_clean_chi2all=compute_stats(clTB_clean,clTB_th)
clEE_clean_mean,clEE_clean_cov,clEE_clean_icov,clEE_clean_chi2r,clEE_clean_chi2all=compute_stats(clEE_clean,clEE_th)
clEB_clean_mean,clEB_clean_cov,clEB_clean_icov,clEB_clean_chi2r,clEB_clean_chi2all=compute_stats(clEB_clean,clEB_th)
clBB_clean_mean,clBB_clean_cov,clBB_clean_icov,clBB_clean_chi2r,clBB_clean_chi2all=compute_stats(clBB_clean,clBB_th)
clTT_dirty_mean,clTT_dirty_cov,clTT_dirty_icov,clTT_dirty_chi2r,clTT_dirty_chi2all=compute_stats(clTT_dirty,clTT_th)
clTE_dirty_mean,clTE_dirty_cov,clTE_dirty_icov,clTE_dirty_chi2r,clTE_dirty_chi2all=compute_stats(clTE_dirty,clTE_th)
clTB_dirty_mean,clTB_dirty_cov,clTB_dirty_icov,clTB_dirty_chi2r,clTB_dirty_chi2all=compute_stats(clTB_dirty,clTB_th)
clEE_dirty_mean,clEE_dirty_cov,clEE_dirty_icov,clEE_dirty_chi2r,clEE_dirty_chi2all=compute_stats(clEE_dirty,clEE_th)
clEB_dirty_mean,clEB_dirty_cov,clEB_dirty_icov,clEB_dirty_chi2r,clEB_dirty_chi2all=compute_stats(clEB_dirty,clEB_th)
clBB_dirty_mean,clBB_dirty_cov,clBB_dirty_icov,clBB_dirty_chi2r,clBB_dirty_chi2all=compute_stats(clBB_dirty,clBB_th)
m,cov,icov,chi2r,chi2all=compute_stats(np.vstack((clTT_clean.T,clTE_clean.T,clTB_clean.T,
                                                  clEE_clean.T,clEB_clean.T,clBB_clean.T)).T,
                                       np.vstack((clTT_th,clTE_th,clTB_th,
                                                  clEE_th,clEB_th,clBB_th)).flatten())



#Plot covariance
plt.figure();
ax=plt.gca();
im=ax.imshow(cov/np.sqrt(np.diag(cov)[None,:]*np.diag(cov)[:,None]),
             interpolation='nearest',cmap=plt.cm.Greys);
for i in np.arange(5)+1 :
    ax.plot([i*ndof,i*ndof],[0,6*ndof],'k--',lw=1)
    ax.plot([0,6*ndof],[i*ndof,i*ndof],'k--',lw=1)
ax.set_xlim([0,6*ndof])
ax.set_ylim([6*ndof,0])
ax.set_xticks(ndof*(np.arange(6)+0.5))
ax.set_yticks(ndof*(np.arange(6)+0.5))
ax.set_xticklabels(['$\\delta\\delta$','$\\delta\\gamma_E$','$\\delta\\gamma_B$',
                    '$\\gamma_E\\gamma_E$','$\\gamma_E\\gamma_B$','$\\gamma_B\\gamma_B$'])
ax.set_yticklabels(['$\\delta\\delta$','$\\delta\\gamma_E$','$\\delta\\gamma_B$',
                    '$\\gamma_E\\gamma_E$','$\\gamma_E\\gamma_B$','$\\gamma_B\\gamma_B$'])
tickfs(ax)
plt.colorbar(im)
plt.savefig("plots_paper/val_covar_lss_sph.pdf",bbox_inches='tight')

#Plot residuals
cols=plt.cm.rainbow(np.linspace(0,1,6))
plot_dirty=True
fig=plt.figure()
ax=fig.add_axes((0.12,0.4,0.78,0.55))
ic=0

ax.plot(l_th,clTT_clean_mean,label='$\\delta\\times\\delta$',c=cols[ic],alpha=0.45);
#if plot_dirty : 
#   ax.plot(l_th,clTT_dirty_mean,'-.',c=cols[ic],alpha=0.3);
ax.plot(l_th,clTT_th,'--',c=cols[ic]); 
ic+=1
ax.plot(l_th,clTE_clean_mean,label='$\\delta\\times\\gamma_E$',c=cols[ic],alpha=0.45);
#if plot_dirty :
#    ax.plot(l_th,clTE_dirty_mean,'-.',c=cols[ic],alpha=0.3); 
ax.plot(l_th,clTE_th,'--',c=cols[ic]);
ic+=1
ax.plot(l_th,clTB_clean_mean,label='$\\delta\\times\\gamma_B$',c=cols[ic],alpha=0.45); 
if plot_dirty :
    pass
#    ax.plot(l_th,clTB_dirty_mean,'-.',c=cols[ic],alpha=0.3);
else :
    ax.plot([-1,-1],[-1,-1],'k-' ,label='${\\rm Sims}$')
ic+=1
ax.plot(l_th,clEE_clean_mean,label='$\\gamma_E\\times\\gamma_E$',c=cols[ic],alpha=0.45);
#if plot_dirty :
#    ax.plot(l_th,clEE_dirty_mean,'-.',c=cols[ic],alpha=0.3);
ax.plot(l_th,clEE_th,'--',c=cols[ic]);
ic+=1
ax.plot(l_th,clEB_clean_mean,label='$\\gamma_E\\times\\gamma_B$',c=cols[ic],alpha=0.45);
#if plot_dirty :
#    ax.plot(l_th,clEB_dirty_mean,'-.',c=cols[ic],alpha=0.3);
ic+=1
ax.plot(l_th,clBB_clean_mean,label='$\\gamma_B\\times\\gamma_B$',c=cols[ic],alpha=0.45);
#if plot_dirty :
#    ax.plot(l_th,clBB_dirty_mean,'-.',c=cols[ic],alpha=0.3);
ic+=1
if plot_dirty : 
    ax.plot([-1,-1],[-1,-1],'k-' ,label='${\\rm Sims}$')
    #ax.plot([-1,-1],[-1,-1],'k-.' ,label='${\\rm No\\,\\,deproj.}$')
ax.plot([-1,-1],[-1,-1],'k--',label='${\\rm Input}$')
if plot_dirty : 
    ax.set_ylim([1E-13,5E-4])
    ax.legend(loc='upper right',frameon=False,fontsize=14,ncol=3,labelspacing=0.1)
else :
    ax.set_ylim([1E-13,5E-4])
    ax.legend(loc='upper right',frameon=False,fontsize=14,ncol=2,labelspacing=0.1)
ax.set_xlim([5,2049])
#ax.set_xscale('log')
ax.set_yscale('log');
tickfs(ax)
ax.set_xticks([])
ax.set_yticks([1E-12,1E-10,1E-8,1E-6,1E-4])
ax.set_ylabel('$C_\\ell$',fontsize=15)
ax=fig.add_axes((0.12,0.25,0.78,0.15))
ic=0
cols=plt.cm.rainbow(np.linspace(0,1,6))
errTT = np.sqrt(np.diag(clTT_clean_cov))
errTE = np.sqrt(np.diag(clTE_clean_cov))
errTB = np.sqrt(np.diag(clTB_clean_cov))
errEE = np.sqrt(np.diag(clEE_clean_cov))
errEB = np.sqrt(np.diag(clEB_clean_cov))
errBB = np.sqrt(np.diag(clBB_clean_cov))

errTTd = np.sqrt(np.diag(clTT_dirty_cov))
errTEd = np.sqrt(np.diag(clTE_dirty_cov))
errTBd = np.sqrt(np.diag(clTB_dirty_cov))
errEEd = np.sqrt(np.diag(clEE_dirty_cov))
errEBd = np.sqrt(np.diag(clEB_dirty_cov))
errBBd = np.sqrt(np.diag(clBB_dirty_cov))

ax.errorbar(l_th   ,(clTT_clean_mean-clTT_th)*np.sqrt(nsims+0.)/errTT,
            yerr=np.ones(len(l_th)),label='$\\delta\\times\\delta$',fmt='.',c=cols[ic]);
#ax.errorbar(l_th   ,(clTT_dirty_mean-clTT_th)*np.sqrt(nsims+0.)/errTTd,
#            yerr=np.ones(len(l_th)),label='_nolegend_',fmt='^',c=cols[ic],fillstyle='none');
ic+=1

ax.errorbar(l_th+2 ,(clTE_clean_mean-clTE_th)*np.sqrt(nsims+0.)/errTE,
            yerr=np.ones(len(l_th)),label='$\\delta\\times\\gamma_E$',fmt='.',c=cols[ic]);
#ax.errorbar(l_th+2 ,(clTE_dirty_mean-clTE_th)*np.sqrt(nsims+0.)/errTEd,
#            yerr=np.ones(len(l_th)),label='_nolegend_',fmt='^',c=cols[ic],fillstyle='none');
ic+=1
ax.errorbar(l_th+4 ,(clTB_clean_mean-clTB_th)*np.sqrt(nsims+0.)/errTB,
            yerr=np.ones(len(l_th)),label='$\\delta\\times\\gamma_B$',fmt='.',c=cols[ic]);
#ax.errorbar(l_th+4 ,(clTB_dirty_mean-clTB_th)*np.sqrt(nsims+0.)/errTBd,
#            yerr=np.ones(len(l_th)),label='_nolegend_',fmt='^',c=cols[ic],fillstyle='none');
ic+=1
ax.errorbar(l_th+6 ,(clEE_clean_mean-clEE_th)*np.sqrt(nsims+0.)/errEE,
            yerr=np.ones(len(l_th)),label='$\\gamma_E\\times\\gamma_E$',fmt='.',c=cols[ic]);
#ax.errorbar(l_th+6 ,(clEE_dirty_mean-clEE_th)*np.sqrt(nsims+0.)/errEEd,
#            yerr=np.ones(len(l_th)),label='_nolegend_',fmt='^',c=cols[ic],fillstyle='none');
ic+=1
ax.errorbar(l_th+8 ,(clEB_clean_mean-clEB_th)*np.sqrt(nsims+0.)/errEB,
            yerr=np.ones(len(l_th)),label='$\\gamma_E\\times\\gamma_B$',fmt='.',c=cols[ic]);
#ax.errorbar(l_th+8 ,(clEB_dirty_mean-clEB_th)*np.sqrt(nsims+0.)/errEBd,
#            yerr=np.ones(len(l_th)),label='_nolegend_',fmt='^',c=cols[ic],fillstyle='none');
ic+=1
ax.errorbar(l_th+10,(clBB_clean_mean-clBB_th)*np.sqrt(nsims+0.)/errBB,
            yerr=np.ones(len(l_th)),label='$\\gamma_B\\times\\gamma_B$',fmt='.',c=cols[ic]);
#ax.errorbar(l_th+10,(clBB_dirty_mean-clBB_th)*np.sqrt(nsims+0.)/errBBd,
#            yerr=np.ones(len(l_th)),label='_nolegend_',fmt='^',c=cols[ic],fillstyle='none');
ic+=1
#ax.plot(np.linspace(5,2050,3),-4*np.ones(3),'k--')
#ax.plot(np.linspace(5,2050,3),4*np.ones(3),'k--')

ax.set_xlabel('$\\ell$',fontsize=15)
ax.set_ylabel('$\\Delta C_\\ell/\\sigma_\\ell$',fontsize=15)
#ax.set_ylim([-17,17])
ax.set_ylim([-6,6])
#ax.set_xscale('log')
ax.set_xlim([5,2049])
#ax.set_yticks([-15,-10,-5,0,5,10,15])
ax.set_yticks([-4,4])
tickfs(ax)

ax=fig.add_axes((0.12,0.1,0.78,0.15))
ic=0
cols=plt.cm.rainbow(np.linspace(0,1,6))
ax.errorbar(l_th   ,(clTT_dirty_mean-clTT_th)*np.sqrt(nsims+0.)/errTTd,
            yerr=np.ones(len(l_th)),label='_nolegend_',fmt='.',c=cols[ic]);
ic+=1
ax.errorbar(l_th+2 ,(clTE_dirty_mean-clTE_th)*np.sqrt(nsims+0.)/errTEd,
            yerr=np.ones(len(l_th)),label='_nolegend_',fmt='.',c=cols[ic]);
ic+=1
ax.errorbar(l_th+6 ,(clEE_dirty_mean-clEE_th)*np.sqrt(nsims+0.)/errEEd,
            yerr=np.ones(len(l_th)),label='_nolegend_',fmt='.',c=cols[ic]);
ic+=1
ax.errorbar(l_th+8 ,(clEB_dirty_mean-clEB_th)*np.sqrt(nsims+0.)/errEBd,
            yerr=np.ones(len(l_th)),label='_nolegend_',fmt='.',c=cols[ic]);
ic+=1
ax.errorbar(l_th+10,(clBB_dirty_mean-clBB_th)*np.sqrt(nsims+0.)/errBBd,
            yerr=np.ones(len(l_th)),label='_nolegend_',fmt='.',c=cols[ic]);
ic+=1
ax.set_xlabel('$\\ell$',fontsize=15)
ax.set_ylabel('$\\Delta C^{\\prime}_{\\ell}/\\sigma^{\\prime}_{\\ell}$',fontsize=15)
ax.set_ylim([-37,37])
ax.set_xlim([5,2049])
ax.set_yticks([-20,0,20])
tickfs(ax)
plt.savefig("plots_paper/val_cl_lss_sph.pdf",bbox_inches='tight')

#Plot error comparison
cols=plt.cm.rainbow(np.linspace(0,1,6))
plot_dirty=True
fig=plt.figure()
ax=plt.gca()
ic=0

ax.plot(l_th,errTT/errTTd,label='$\\delta\\times\\delta$',c=cols[ic])
#if plot_dirty :
#   ax.plot(l_th,errTTd,'-.',c=cols[ic]);
ic+=1
ax.plot(l_th,errTE/errTEd,label='$\\delta\\times\\gamma_E$',c=cols[ic])
#if plot_dirty :
#    ax.plot(l_th,errTEd,'-.',c=cols[ic]);
ic+=1
ax.plot(l_th,errTB/errTBd,label='$\\delta\\times\\gamma_B$',c=cols[ic]);
#if plot_dirty :
#    ax.plot(l_th,errTBd,'-.',c=cols[ic]);
#else :
#    ax.plot([-1,-1],[-1,-1],'k-' ,label='${\\rm Sims}$')
ic+=1
ax.plot(l_th,errEE/errEEd,label='$\\gamma_E\\times\\gamma_E$',c=cols[ic])
#if plot_dirty :
#    ax.plot(l_th,errEEd,'-.',c=cols[ic]);
ic+=1
ax.plot(l_th,errEB/errEBd,label='$\\gamma_E\\times\\gamma_B$',c=cols[ic]);
#if plot_dirty :
#    ax.plot(l_th,errEBd,'-.',c=cols[ic]);
ic+=1
ax.plot(l_th,errBB/errBBd,label='$\\gamma_B\\times\\gamma_B$',c=cols[ic]);
#if plot_dirty :
#    ax.plot(l_th,errBBd,'-.',c=cols[ic]);
ic+=1
#if plot_dirty :
#    ax.plot([-1,-1],[-1,-1],'k-' ,label='${\\rm Double\\,\\,mask}$')
#    ax.plot([-1,-1],[-1,-1],'k-.' ,label='${\\rm Binary\\,\\,mask}$')

if plot_dirty :
#    ax.set_ylim([1E-13,1E-4])
    ax.legend(loc='upper right',frameon=False,fontsize=14,ncol=3,labelspacing=0.1)
else :
#    ax.set_ylim([1E-13,1E-4])
    ax.legend(loc='upper right',frameon=False,fontsize=14,ncol=2,labelspacing=0.1)
ax.set_xlim([5,2049])
ax.set_xscale('log')
#ax.set_yscale('log');
ax.set_xlabel('$\\ell$',fontsize=15)
ax.set_ylabel('$\\sigma_{\\ell}/\\sigma_{\\ell,{\\rm binary}}$',fontsize=15)
tickfs(ax)
plt.savefig('plots_paper/err_comp_sph.pdf')

#Plot chi2 dist
xr=[ndof-5*np.sqrt(2.*ndof),ndof+5*np.sqrt(2*ndof)]
x=np.linspace(xr[0],xr[1],256)
pdf=st.chi2.pdf(x,ndof)
print('Number of degrees of freedom', ndof)
plt.figure(figsize=(10,7))
ax=[plt.subplot(2,3,i+1) for i in range(6)]
plt.subplots_adjust(wspace=0, hspace=0)

h,b,p=ax[0].hist(clTT_clean_chi2all,bins=40,normed=True,range=xr)
ax[0].text(0.7,0.9,'$\\delta\\times\\delta$'    ,transform=ax[0].transAxes,fontsize=14)
ax[0].set_ylabel('$P(\\chi^2)$',fontsize=15)
ax[0].plot([clTT_clean_chi2r,clTT_clean_chi2r],[0,1.4*np.amax(pdf)],'k-.')
print('TT : %.3lE %.3lE'%(1-st.chi2.cdf(clTT_clean_chi2r,ndof),1-st.chi2.cdf(clTT_dirty_chi2r,ndof)))

h,b,p=ax[1].hist(clTE_clean_chi2all,bins=40,normed=True,range=xr)
ax[1].text(0.7,0.9,'$\\delta\\times\\gamma_E$'  ,transform=ax[1].transAxes,fontsize=14)
ax[1].plot([clTE_clean_chi2r,clTE_clean_chi2r],[0,1.4*np.amax(pdf)],'k-.')
print('TE : %.3lE %.3lE'%(1-st.chi2.cdf(clTE_clean_chi2r,ndof),1-st.chi2.cdf(clTE_dirty_chi2r,ndof)))

h,b,p=ax[2].hist(clTB_clean_chi2all,bins=40,normed=True,range=xr)
ax[2].text(0.7,0.9,'$\\delta\\times\\gamma_B$'  ,transform=ax[2].transAxes,fontsize=14)
ax[2].plot([clTB_clean_chi2r,clTB_clean_chi2r],[0,1.4*np.amax(pdf)],'k-.')
print('TB : %.3lE %.3lE'%(1-st.chi2.cdf(clTB_clean_chi2r,ndof),1-st.chi2.cdf(clTB_dirty_chi2r,ndof)))

h,b,p=ax[3].hist(clEE_clean_chi2all,bins=40,normed=True,range=xr)
ax[3].text(0.7,0.9,'$\\gamma_E\\times\\gamma_E$',transform=ax[3].transAxes,fontsize=14)
ax[3].plot([clEE_clean_chi2r,clEE_clean_chi2r],[0,1.4*np.amax(pdf)],'k-.')
ax[3].set_xlabel('$\\chi^2$',fontsize=15)
ax[3].set_ylabel('$P(\\chi^2)$',fontsize=15)
print('EE : %.3lE %.3lE'%(1-st.chi2.cdf(clEE_clean_chi2r,ndof),1-st.chi2.cdf(clEE_dirty_chi2r,ndof)))

h,b,p=ax[4].hist(clEB_clean_chi2all,bins=40,normed=True,range=xr)
ax[4].text(0.7,0.9,'$\\gamma_E\\times\\gamma_B$',transform=ax[4].transAxes,fontsize=14)
ax[4].plot([clEB_clean_chi2r,clEB_clean_chi2r],[0,1.4*np.amax(pdf)],'k-.')
print('EB : %.3lE %.3lE'%(1-st.chi2.cdf(clEB_clean_chi2r,ndof),1-st.chi2.cdf(clEB_dirty_chi2r,ndof)))

h,b,p=ax[5].hist(clBB_clean_chi2all,bins=40,normed=True,range=xr)
ax[5].text(0.7,0.9,'$\\gamma_B\\times\\gamma_B$',transform=ax[5].transAxes,fontsize=14)
ax[5].plot([clBB_clean_chi2r,clBB_clean_chi2r],[0,1.4*np.amax(pdf)],'k-.')
print('BB : %.3lE %.3lE'%(1-st.chi2.cdf(clBB_clean_chi2r,ndof),1-st.chi2.cdf(clBB_dirty_chi2r,ndof)))

for a in ax[:3] :
    a.set_xticklabels([])
for a in ax[3:] :
    a.set_xlabel('$\\chi^2$',fontsize=15)
ax[1].set_yticklabels([])
ax[2].set_yticklabels([])
ax[4].set_yticklabels([])
ax[5].set_yticklabels([])
for a in ax :
    tickfs(a)
    a.set_xlim([ndof-5*np.sqrt(2.*ndof),ndof+5*np.sqrt(2.*ndof)])
    a.set_ylim([0,1.4*np.amax(pdf)])
    a.plot(x,pdf,'k-',label='$P(\\chi^2)$')
    a.plot([ndof,ndof],[0,1.4*np.amax(pdf)],'k--',label='$N_{\\rm dof}$')
    a.plot([-1,-1],[-1,-1],'k-.',label='$\\chi^2_{\\rm mean}$')
ax[0].legend(loc='upper left',fontsize=12,frameon=False)
plt.savefig("plots_paper/val_chi2_lss_sph.pdf",bbox_inches='tight')
plt.show()
