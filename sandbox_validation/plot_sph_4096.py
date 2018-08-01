import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import os
import astropy.table

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
path = '/global/cscratch1/sd/jsanch87/tests_sph_july2018/'

def tickfs(ax,x=True,y=True) :
    if x :
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(12)
    if y :
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(12)

def plot_results(nside,apo,mask,cont,nvar,nrange=1000):
    files = [os.path.join(path,'run_ns%d_mask%d_cont%d_nvar%d_apo%.2f_cl_%04d.txt' % (nside,mask,cont,nvar,apo,run)) for run in range(0,nrange)]
    file_th = os.path.join(path,'run_ns%d_mask%d_cont%d_nvar%d_apo%.2f_cl_th.txt' % (nside,mask,cont,nvar,apo))
    cl00_all = []
    cl02_all = []
    cl22_all = []
    for fname in files:
        tab = astropy.table.Table.read(fname,format='ascii')
        cl00_all.append(tab['col2'])
        cl02_all.append([tab['col3'],tab['col4']])
        cl22_all.append([tab['col5'],tab['col5'],tab['col6'],tab['col7'],tab['col8']])
    cl00_all = np.array(cl00_all)
    cl02_all = np.array(cl02_all)
    cl22_all = np.array(cl22_all)
    cols=plt.cm.rainbow(np.linspace(0,1,6))
    tab_th = astropy.table.Table.read(file_th,format='ascii')
    cl00_th = tab_th['col2']
    cl02_th = [tab_th['col3'],tab_th['col4']]
    cl22_th = [tab_th['col5'],tab_th['col6'],tab_th['col7'],tab_th['col8']]
    #f1, ax = plt.subplots(1,1,figsize=(11,3))
    f1 = plt.figure()
    ax = plt.gca()
    plt.plot(tab['col1'],cl00_th,'--',c=cols[0],alpha=0.9,label='_nolegend_')
    plt.plot(tab['col1'],np.mean(cl00_all,axis=0),'-',c=cols[0],alpha=0.3,label=r'$\delta - \delta$')
    plt.plot(tab['col1'],cl02_th[0],'--',c=cols[1],alpha=0.9,label='_nolegend_')
    plt.plot(tab['col1'],np.mean(cl02_all,axis=0)[0],'-',c=cols[1],alpha=0.3,label=r'$\delta - \gamma_{E}$')
    plt.plot(tab['col1'],cl02_th[1],'--',c=cols[2],alpha=0.9,label='_nolegend_')
    plt.plot(tab['col1'],np.mean(cl02_all,axis=0)[1],'-',c=cols[2],alpha=0.3,label=r'$\delta - \gamma_{B}$')
    plt.plot(tab['col1'],np.mean(cl22_all,axis=0)[0],'-',c=cols[3],alpha=0.3,label=r'$\gamma_{E} - \gamma_{E}$' )
    plt.plot(tab['col1'],cl22_th[0],'--',c=cols[3],alpha=0.9,label='_nolegend_')
    plt.plot(tab['col1'],np.mean(cl22_all,axis=0)[1],'-',c=cols[4],alpha=0.3,label=r'$\gamma_{E} - \gamma_{B}$')
    plt.plot(tab['col1'],cl22_th[1],'--',c=cols[4],alpha=0.9,label='_nolegend_')
    plt.plot(tab['col1'],np.mean(cl22_all,axis=0)[3],'-',c=cols[5],alpha=0.3,label=r'$\gamma_{B} - \gamma_{B}$')
    plt.plot(tab['col1'],cl22_th[3],'--',c=cols[5],alpha=0.9,label='_nolegend_')
    plt.plot([-1,-1],[-1,-1],'k-' ,label='${\\rm Sims}$')
    plt.plot([-1,-1],[-1,-1],'k--',label='${\\rm Input}$')
    plt.xlim(0,2*4096)
    plt.ylim((1e-12,1e-5))
    plt.yscale('log')
    plt.legend(loc='best',frameon=False,fontsize=16,ncol=2)
    max_ind = np.where(tab['col1']<2*nside)[0][-1]
    plt.xlabel('$\\ell$',fontsize=16)
    plt.ylabel('$C_\\ell$',fontsize=16)
    tickfs(ax)
    plt.tight_layout()
    chi2_00 = 0
    chi2_02 = 0
    chi2_22 = 0
    ndof = len(tab['col1'])
    return f1, chi2_00, chi2_02, chi2_22, ndof

f2, chi2_00_4096, chi2_02_4096, chi2_22_4096, ndof_4096 = plot_results(4096,0,1,1,1,nrange=100)
f2.tight_layout()
f2.savefig('/global/homes/j/jsanch87/NaMaster/sandbox_validation/plots_paper/deprojected_4096.pdf')
plt.show()
