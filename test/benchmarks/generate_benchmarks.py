from __future__ import print_function
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt

def getmaskapoana(ns,aps,fsk=0.1,dec0=-50,ra0=0.) :
    #This generates a correctly-apodized mask
    v0=np.array([np.sin(DTOR*(90-dec0))*np.cos(DTOR*ra0),
                 np.sin(DTOR*(90-dec0))*np.sin(DTOR*ra0),
                 np.cos(DTOR*(90-dec0))])
    vv=np.array(hp.pix2vec(ns,np.arange(hp.nside2npix(ns))))
    cth=np.sum(v0[:,None]*vv,axis=0); th=np.arccos(cth); th0=np.arccos(1-2*fsk); th_apo=aps*DTOR
    id0=np.where(th>=th0)[0]
    id1=np.where(th<=th0-th_apo)[0]
    idb=np.where((th>th0-th_apo) & (th<th0))[0]
    x=np.sqrt((1-np.cos(th[idb]-th0))/(1-np.cos(th_apo)))
    mask_apo=np.zeros(hp.nside2npix(ns))
    mask_apo[id0]=0.
    mask_apo[id1]=1.
    mask_apo[idb]=x-np.sin(2*np.pi*x)/(2*np.pi)
    return mask_apo


nside_out=64
DTOR=np.pi/180
mask=getmaskapoana(nside_out,20.,0.4,dec0=90.)
l,cltt,clee,clbb,clte,nltt,nlee,nlbb,nlte=np.loadtxt("cls_lss.txt",unpack=True)
dl,dw_q,dw_u=hp.synfast([(cltt+nltt)[:3*nside_out],
                         (clee+nlee)[:3*nside_out],
                         (clbb+nlbb)[:3*nside_out],
                         (clte+nlte)[:3*nside_out]],
                        nside_out,new=True,verbose=False,pol=True)
sl=hp.ud_grade(hp.read_map("../../sandbox_validation/data/cont_lss_dust_ns64.fits",
                           field=0,verbose=False),nside_out=nside_out);
sw_q=hp.ud_grade(hp.read_map("../../sandbox_validation/data/cont_wl_psf_ns64.fits",
                             field=0,verbose=False),nside_out=nside_out);
sw_u=hp.ud_grade(hp.read_map("../../sandbox_validation/data/cont_wl_psf_ns64.fits",
                             field=1,verbose=False),nside_out=nside_out);
hp.write_map("msk.fits",mask,overwrite=True)
hp.write_map("mps.fits",[dl,dw_q,dw_u],overwrite=True)
hp.write_map("tmp.fits",[sl,sw_q,sw_u],overwrite=True)

b=nmt.NmtBin(nside_out,nlb=16)
leff=b.get_effective_ells()
lfull=np.arange(3*nside_out)

#No contaminants
prefix='bm_nc_np'
f0=nmt.NmtField(mask,[dl])
f2=nmt.NmtField(mask,[dw_q,dw_u])
w00=nmt.NmtWorkspace(); w00.compute_coupling_matrix(f0,f0,b);
clb00=np.array([nltt[:3*nside_out]])
c00=w00.decouple_cell(nmt.compute_coupled_cell(f0,f0),cl_bias=clb00)
w00.write_to(prefix+'_w00.dat')
np.savetxt(prefix+"_c00.txt",np.transpose([leff,c00[0]]))
w02=nmt.NmtWorkspace(); w02.compute_coupling_matrix(f0,f2,b);
clb02=np.array([nlte[:3*nside_out],0*nlte[:3*nside_out]])
c02=w02.decouple_cell(nmt.compute_coupled_cell(f0,f2),cl_bias=clb02)
w02.write_to(prefix+'_w02.dat')
np.savetxt(prefix+"_c02.txt",np.transpose([leff,c02[0],c02[1]]))
w22=nmt.NmtWorkspace(); w22.compute_coupling_matrix(f2,f2,b);
clb22=np.array([nlee[:3*nside_out],0*nlee[:3*nside_out],
                0*nlbb[:3*nside_out],nlbb[:3*nside_out]])
c22=w22.decouple_cell(nmt.compute_coupled_cell(f2,f2),cl_bias=clb22)
w22.write_to(prefix+'_w22.dat')
np.savetxt(prefix+"_c22.txt",np.transpose([leff,c22[0],c22[1],c22[2],c22[3]]))

#With contaminants
prefix='bm_yc_np'
f0=nmt.NmtField(mask,[dl],templates=[[sl]])
f2=nmt.NmtField(mask,[dw_q,dw_u],templates=[[sw_q,sw_u]])
w00=nmt.NmtWorkspace(); w00.compute_coupling_matrix(f0,f0,b);
clb00=nmt.deprojection_bias(f0,f0,[(cltt+nltt)[:3*nside_out]])
np.savetxt(prefix+'_cb00.txt',np.transpose([lfull,clb00[0]]))
clb00+=np.array([nltt[:3*nside_out]])
c00=w00.decouple_cell(nmt.compute_coupled_cell(f0,f0),cl_bias=clb00)
w00.write_to(prefix+'_w00.dat')
np.savetxt(prefix+"_c00.txt",np.transpose([leff,c00[0]]))
w02=nmt.NmtWorkspace(); w02.compute_coupling_matrix(f0,f2,b);
clb02=nmt.deprojection_bias(f0,f2,[(clte+nlte)[:3*nside_out],0*clte[:3*nside_out]])
np.savetxt(prefix+'_cb02.txt',np.transpose([lfull,clb02[0],clb02[1]]))
clb02+=np.array([nlte[:3*nside_out],0*nlte[:3*nside_out]])
c02=w02.decouple_cell(nmt.compute_coupled_cell(f0,f2),cl_bias=clb02)
w02.write_to(prefix+'_w02.dat')
np.savetxt(prefix+"_c02.txt",np.transpose([leff,c02[0],c02[1]]))
w22=nmt.NmtWorkspace(); w22.compute_coupling_matrix(f2,f2,b);
clb22=nmt.deprojection_bias(f2,f2,[(clee+nlee)[:3*nside_out],0*clee[:3*nside_out],
                                   0*clbb[:3*nside_out],(clbb+nlbb)[:3*nside_out]])
np.savetxt(prefix+'_cb22.txt',np.transpose([lfull,clb22[0],clb22[1],clb22[2],clb22[3]]))
clb22+=np.array([nlee[:3*nside_out],0*nlee[:3*nside_out],
                 0*nlbb[:3*nside_out],nlbb[:3*nside_out]])
c22=w22.decouple_cell(nmt.compute_coupled_cell(f2,f2),cl_bias=clb22)
w22.write_to(prefix+'_w22.dat')
np.savetxt(prefix+"_c22.txt",np.transpose([leff,c22[0],c22[1],c22[2],c22[3]]))

#No contaminants, purified
prefix='bm_nc_yp'
f0=nmt.NmtField(mask,[dl])
f2=nmt.NmtField(mask,[dw_q,dw_u],purify_b=True)
w00=nmt.NmtWorkspace(); w00.compute_coupling_matrix(f0,f0,b);
clb00=np.array([nltt[:3*nside_out]])
c00=w00.decouple_cell(nmt.compute_coupled_cell(f0,f0),cl_bias=clb00)
w00.write_to(prefix+'_w00.dat')
np.savetxt(prefix+"_c00.txt",np.transpose([leff,c00[0]]))
w02=nmt.NmtWorkspace(); w02.compute_coupling_matrix(f0,f2,b);
clb02=np.array([nlte[:3*nside_out],0*nlte[:3*nside_out]])
c02=w02.decouple_cell(nmt.compute_coupled_cell(f0,f2),cl_bias=clb02)
w02.write_to(prefix+'_w02.dat')
np.savetxt(prefix+"_c02.txt",np.transpose([leff,c02[0],c02[1]]))
w22=nmt.NmtWorkspace(); w22.compute_coupling_matrix(f2,f2,b);
clb22=np.array([nlee[:3*nside_out],0*nlee[:3*nside_out],
                0*nlbb[:3*nside_out],nlbb[:3*nside_out]])
c22=w22.decouple_cell(nmt.compute_coupled_cell(f2,f2),cl_bias=clb22)
w22.write_to(prefix+'_w22.dat')
np.savetxt(prefix+"_c22.txt",np.transpose([leff,c22[0],c22[1],c22[2],c22[3]]))

#With contaminants
prefix='bm_yc_yp'
f0=nmt.NmtField(mask,[dl],templates=[[sl]])
f2=nmt.NmtField(mask,[dw_q,dw_u],templates=[[sw_q,sw_u]],purify_b=True)
w00=nmt.NmtWorkspace(); w00.compute_coupling_matrix(f0,f0,b);
clb00=nmt.deprojection_bias(f0,f0,[(cltt+nltt)[:3*nside_out]])
np.savetxt(prefix+'_cb00.txt',np.transpose([lfull,clb00[0]]))
clb00+=np.array([nltt[:3*nside_out]])
c00=w00.decouple_cell(nmt.compute_coupled_cell(f0,f0),cl_bias=clb00)
w00.write_to(prefix+'_w00.dat')
np.savetxt(prefix+"_c00.txt",np.transpose([leff,c00[0]]))
w02=nmt.NmtWorkspace(); w02.compute_coupling_matrix(f0,f2,b);
clb02=nmt.deprojection_bias(f0,f2,[(clte+nlte)[:3*nside_out],0*clte[:3*nside_out]])
np.savetxt(prefix+'_cb02.txt',np.transpose([lfull,clb02[0],clb02[1]]))
clb02+=np.array([nlte[:3*nside_out],0*nlte[:3*nside_out]])
c02=w02.decouple_cell(nmt.compute_coupled_cell(f0,f2),cl_bias=clb02)
w02.write_to(prefix+'_w02.dat')
np.savetxt(prefix+"_c02.txt",np.transpose([leff,c02[0],c02[1]]))
w22=nmt.NmtWorkspace(); w22.compute_coupling_matrix(f2,f2,b);
clb22=nmt.deprojection_bias(f2,f2,[(clee+nlee)[:3*nside_out],0*clee[:3*nside_out],
                                   0*clbb[:3*nside_out],(clbb+nlbb)[:3*nside_out]])
np.savetxt(prefix+'_cb22.txt',np.transpose([lfull,clb22[0],clb22[1],clb22[2],clb22[3]]))
clb22+=np.array([nlee[:3*nside_out],0*nlee[:3*nside_out],
                 0*nlbb[:3*nside_out],nlbb[:3*nside_out]])
c22=w22.decouple_cell(nmt.compute_coupled_cell(f2,f2),cl_bias=clb22)
w22.write_to(prefix+'_w22.dat')
np.savetxt(prefix+"_c22.txt",np.transpose([leff,c22[0],c22[1],c22[2],c22[3]]))
