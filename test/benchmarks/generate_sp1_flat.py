import numpy as np
import matplotlib.pyplot as plt
import pymaster as nmt
from astropy.wcs import WCS
from astropy.io import fits


l,cltt,clee,clbb,clte,nltt,nlee,nlbb,nlte=np.loadtxt("cls_lss.txt",unpack=True)
cltt[:2] = 0
DTOR = np.pi/180


def write_flat_map(filename,maps,wcs,descript=None) :
    if maps.ndim<2 :
        raise ValueError("Must supply at least one map")
    if maps.ndim==2 :
        maps=np.array([maps])
    if descript is not None :
        if len(maps)==1 :
            descript=[descript]
        if len(maps)!=len(descript) :
            raise ValueError("Need one description per map")

    header=wcs.to_header()
    hdus=[]
    for im,m in enumerate(maps) :
        head=header.copy()
        if m.shape!=wcs.array_shape:
            raise ValueError("Map shape differs from WCS!")
        if descript is not None :
            head['DESCR']=(descript[im],'Description')
        if im==0 :
            hdu=fits.PrimaryHDU(data=m,header=head)
        else :
            hdu=fits.ImageHDU(data=m,header=head)
        hdus.append(hdu)
    hdulist=fits.HDUList(hdus)
    hdulist.writeto(filename,overwrite=True)

def read_flat_map(filename,i_map=0) :
    """
    Reads a flat-sky map and the details of its pixelization scheme.
    The latter are returned as a FlatMapInfo object.
    i_map : map to read. If -1, all maps will be read.
    """
    hdul=fits.open(filename)
    w=WCS(hdul[0].header)

    maps=hdul[i_map].data
    ny,nx=maps.shape

    return w,maps

wcs,msk=read_flat_map("msk_flat.fits")
(ny,nx)=msk.shape
lx=np.fabs(nx*wcs.wcs.cdelt[0])*np.pi/180
ly=np.fabs(ny*wcs.wcs.cdelt[1])*np.pi/180
d_ell=20; lmax=500.; ledges=np.arange(int(lmax/d_ell)+1)*d_ell+2
b=nmt.NmtBinFlat(ledges[:-1],ledges[1:])
leff=b.get_effective_ells();
dt,dq,du=nmt.synfast_flat(int(nx),int(ny),lx,ly,
                          [cltt,cltt,0*cltt,cltt,0*cltt,0*cltt],[0,1])
write_flat_map("mps_sp1_flat.fits", np.array([dt,dq,du]),wcs,["T", "Q", "U"])
prefix = 'bm_f_sp1'
f0=nmt.NmtFieldFlat(lx,ly,msk,[dt])
f1=nmt.NmtFieldFlat(lx,ly,msk,[dq,du], spin=1)

w00=nmt.NmtWorkspaceFlat(); w00.compute_coupling_matrix(f0,f0,b);
w00.write_to(prefix+'_w00.fits')
c00=w00.decouple_cell(nmt.compute_coupled_cell_flat(f0,f0,b))
np.savetxt(prefix+'_c00.txt', np.transpose([leff, c00[0]]))

w01=nmt.NmtWorkspaceFlat(); w01.compute_coupling_matrix(f0,f1,b);
w01.write_to(prefix+'_w01.fits')
c01=w01.decouple_cell(nmt.compute_coupled_cell_flat(f0,f1,b))
np.savetxt(prefix+'_c01.txt', np.transpose([leff, c01[0], c01[1]]))

w11=nmt.NmtWorkspaceFlat(); w11.compute_coupling_matrix(f1,f1,b);
w11.write_to(prefix+'_w11.fits')
c11=w11.decouple_cell(nmt.compute_coupled_cell_flat(f1,f1,b))
np.savetxt(prefix+'_c11.txt', np.transpose([leff, c11[0], c11[1], c11[2], c11[3]]))

plt.figure()
plt.plot(leff, c01[0]/c00[0]-1, 'r--')
plt.plot(leff, c11[0]/c00[0]-1, 'b:')

plt.figure(); plt.imshow(dt*msk)
plt.figure(); plt.imshow(dq*msk)
plt.figure(); plt.imshow(du*msk)
plt.show()
print(dt.shape)
exit(1)
