from pymaster import nmtlib as lib
import numpy as np

def mask_apodization(mask_in,aposize,apotype="C1") :
    """
    Apodizes a mask with an given apodization scale using different methods.

    :param mask_in: input mask, provided as an array of floats corresponding to a HEALPix map in RING order.
    :param aposize: apodization scale in degrees.
    :param apotype: apodization type. Three methods implemented: "C1", "C2" and "Smooth". See the description of the C-function nmt_apodize_mask in the C API documentation for a full description of these methods.
    :return: apodized mask as a HEALPix map
    """
    return lib.apomask(mask_in.astype('float64'),len(mask_in),aposize,apotype)

def mask_apodization_flat(mask_in,lx,ly,aposize,apotype="C1") :
    """
    Apodizes a flat-sky mask with an given apodization scale using different methods.

    :param mask_in: input mask, provided as a 2D array (ny,nx) of floats.
    :param float lx: patch size in the x-axis (in radians)
    :param float ly: patch size in the y-axis (in radians)
    :param aposize: apodization scale in degrees.
    :param apotype: apodization type. Three methods implemented: "C1", "C2" and "Smooth". See the description of the C-function nmt_apodize_mask in the C API documentation for a full description of these methods.
    :return: apodized mask as a 2D array (ny,nx)
    """
    nx=len(mask_in[0])
    ny=len(mask_in)
    mask_apo_flat=lib.apomask_flat(nx,ny,lx,ly,mask_in.flatten().astype('float64'),nx*ny,aposize,apotype)
    return mask_apo_flat.reshape([ny,nx])

def synfast_spherical(nside,cls,pol=False,beam=None) :
    """
    Generates a full-sky Gaussian random field according to a given power spectrum. This function should produce outputs similar to healpy's synfast.

    :param int nside: HEALpix resolution parameter
    :param array-like cls: array containing power spectra. If pol=False, cls should be a 1D array. If pol=True it should be a 2D array with 4 (TT,EE,BB,TE) or 6 (TT,EE,BB,TE,EB,TB) power spectra.
    :param boolean pol: Set to True if you want to generate T, Q and U
    :param beam array-like: 1D array containing the instrumental beam (the output map(s) will be convolved with it)
    :return: 1 or 3 full-sky maps
    """
    seed=np.random.randint(50000000)
    if pol :
        use_pol=1
        nmaps=3
        if(len(np.shape(cls))!=2) :
            raise KeyError("You should supply more than one power spectrum if you want polarization")
        ncl=len(cls)
        if ((ncl!=4) and (ncl!=6)) :
            raise KeyError("You should provide 4 or 6 power spectra if you want polarization")
        lmax=len(cls[0])-1
        cls_use=np.zeros([6,lmax+1])
        cls_use[0,:]=cls[0] #TT
        cls_use[1,:]=cls[3] #TE
        cls_use[3,:]=cls[1] #EE
        cls_use[5,:]=cls[2] #BB
        if ncl==6 :
            cls_use[4,:]=cls[4] #EB
            cls_use[2,:]=cls[5] #TB
    else :
        use_pol=0
        nmaps=1
        if(len(np.shape(cls))!=1) :
            raise KeyError("You should supply only one power spectrum if you don't want polarization")
        lmax=len(cls)-1
        cls_use=np.array([cls])

    if beam is None :
        beam_use=np.ones(lmax+1)
    else :
        if len(beam)!=lmax+1 :
            raise KeyError("The beam should have as many multipoles as the power spectrum")
        beam_use=beam
    data=lib.synfast_new(nside,use_pol,seed,cls_use,beam_use,nmaps*12*nside*nside)

    maps=data.reshape([nmaps,12*nside*nside])

    return maps

def synfast_flat(nx,ny,lx,ly,cls,pol=False,beam=None) :
    """
    Generates a flat-sky Gaussian random field according to a given power spectrum. This function is the flat-sky equivalent of healpy's synfast.

    :param int nx: number of pixels in the x-axis
    :param int ny: number of pixels in the y-axis
    :param float lx: patch size in the x-axis (in radians)
    :param float ly: patch size in the y-axis (in radians)
    :param array-like cls: array containing power spectra. If pol=False, cls should be a 1D array. If pol=True it should be a 2D array with 4 (TT,EE,BB,TE) or 6 (TT,EE,BB,TE,EB,TB) power spectra.
    :param boolean pol: Set to True if you want to generate T, Q and U
    :param beam array-like: 1D array containing the instrumental beam (the output map(s) will be convolved with it)
    :return: 1 or 3 2D arrays of size (ny,nx) containing the simulated maps
    """
    seed=np.random.randint(50000000)
    if pol :
        use_pol=1
        nmaps=3
        if(len(np.shape(cls))!=2) :
            raise KeyError("You should supply more than one power spectrum if you want polarization")
        ncl=len(cls)
        if ((ncl!=4) and (ncl!=6)) :
            raise KeyError("You should provide 4 or 6 power spectra if you want polarization")
        lmax=len(cls[0])-1
        cls_use=np.zeros([6,lmax+1])
        cls_use[0,:]=cls[0] #TT
        cls_use[1,:]=cls[3] #TE
        cls_use[3,:]=cls[1] #EE
        cls_use[5,:]=cls[2] #BB
        if ncl==6 :
            cls_use[2,:]=cls[4] #TB
            cls_use[4,:]=cls[5] #EB
    else :
        use_pol=0
        nmaps=1
        if(len(np.shape(cls))!=1) :
            raise KeyError("You should supply only one power spectrum if you don't want polarization")
        lmax=len(cls)-1
        cls_use=np.array([cls])

    if beam is None :
        beam_use=np.ones(lmax+1)
    else :
        if len(beam)!=lmax+1 :
            raise KeyError("The beam should have as many multipoles as the power spectrum")
        beam_use=beam
    data=lib.synfast_new_flat(nx,ny,lx,ly,use_pol,seed,cls_use,beam_use,nmaps*ny*nx)

    maps=data.reshape([nmaps,ny,nx])

    return maps
