import numpy as np
import matplotlib.pyplot as plt

NSIDES=np.array([64,256,1024])
NCORES=np.array([16]) #[1,2,4,8,16])

def read_data(prefix) :
    n_ns=len(NSIDES)
    n_nc=len(NCORES)
    
    data=np.zeros([n_nc,n_ns,3])

    for i_c,nc in enumerate(NCORES) :
        for i_n,nn in enumerate(NSIDES) :
            print prefix+"_ns%d_nc%d.txt"%(nn,nc)
            d=np.genfromtxt(prefix+"_ns%d_nc%d.txt"%(nn,nc))[3:]
            print d
            data[i_c,i_n,:]=d
    return data

t_data={}
for k in ['field','mcm','deproj','pure','pure_deproj'] :
    t_data[k]=read_data(k)
    plt.plot(NSIDES,t_data[k][0,:,0],label=k)
    plt.plot(NSIDES,t_data[k][0,1,0]*((NSIDES+0.)/NSIDES[1])**3,'k--')
plt.loglog()
plt.legend(loc='upper left')
plt.show()

absc=np.array(['Field, $s=0$','Field, $s=2$',
               'Deproj., $s=0$','Deproj., $s=2$',
               'Purification','Deproj. $+$ purif.',
               'MCM, 0-0','MCM, 0-2','MCM, 0-3'])
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

plt.figure()
ax=plt.gca()
ax.bar(np.arange(len(absc)),data_1024_plot,0.5)
ax.bar(np.arange(len(absc))+0.5,data_256_plot,0.5)
ax.set_xlabel('Task')
ax.set_ylabel('Time (ms)')
ax.set_xticks(np.arange(len(absc))+0.25)
ax.set_xticklabels(absc)
plt.show()
