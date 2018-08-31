from matplotlib import rc
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

NSIDES=np.array([64,256,1024])
NCORES=np.array([16])

def read_data(prefix) :
    n_ns=len(NSIDES)
    n_nc=len(NCORES)
    
    data=np.zeros([n_nc,n_ns,3])

    for i_c,nc in enumerate(NCORES) :
        for i_n,nn in enumerate(NSIDES) :
            d=np.genfromtxt('output/'+prefix+"_ns%d_nc%d.txt"%(nn,nc))[3:]
            data[i_c,i_n,:]=d
    return data

t_data={}
for k in ['field','mcm','deproj','pure','pure_deproj'] :
    t_data[k]=read_data(k)

absc=np.array(['Field, $s=0$','Field, $s=2$',
               'Deproj., $s=0$','Deproj., $s=2$',
               'Purification','Deproj. + purif.',
               'MCM, $0$-$0$','MCM, $0$-$2$','MCM, $2$-$2$'])
data_1024_plot=np.array([t_data['field'][0,2,0], #Field s0
                         t_data['field'][0,2,1], #Field s2
                         t_data['deproj'][0,2,0], #Deproj s0
                         t_data['deproj'][0,2,1], #Deproj s2
                         t_data['pure'][0,2,0], #Pure s2
                         t_data['pure_deproj'][0,2,0], #Pure-deproj s2
                         t_data['mcm'][0,2,0], #MCM-00
                         t_data['mcm'][0,2,1], #MCM-02
                         t_data['mcm'][0,2,2] #MCM-22
                         ])
data_256_plot=np.array([t_data['field'][0,1,0], #Field s0
                        t_data['field'][0,1,1], #Field s2
                        t_data['deproj'][0,1,0], #Deproj s0
                        t_data['deproj'][0,1,1], #Deproj s2
                        t_data['pure'][0,1,0], #Pure s2
                        t_data['pure_deproj'][0,1,0], #Pure-deproj s2
                        t_data['mcm'][0,1,0], #MCM-00
                        t_data['mcm'][0,1,1], #MCM-02
                        t_data['mcm'][0,1,2] #MCM-22
                        ])

def tickfs(ax,x=True,y=True,fs=12) :
    if x :
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fs)
    if y :
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fs)

plt.figure(figsize=(8,4))
ax=plt.gca()
ax.bar(np.arange(len(absc)),data_1024_plot/1000.,0.75)
ax.text(0.03,0.85,'$N_{\\rm side}=1024$\n 5 contaminant templates.',transform=ax.transAxes,fontsize=15)
ax.set_xlabel('Task',fontsize=15)
ax.set_ylabel('Time (s)',fontsize=15)
ax.set_xticks(np.arange(len(absc)))
ax.set_xticklabels(absc,rotation=20)
tickfs(ax)
plt.savefig("../plots_paper/timing.pdf",bbox_inches='tight')
plt.show()
