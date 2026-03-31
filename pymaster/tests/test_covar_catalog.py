import numpy as np
import healpy as hp
import pymaster as nmt


def get_cl_in(lmax=300):
    ls = np.arange(lmax+1)
    cls = 1/(10+ls)
    return ls, cls


def get_cat(ncat=100000, seed=1234, get_field=False, lmax=300):
    np.random.seed(seed)
    phis = 2*np.pi*np.random.rand(ncat)
    thetas = np.arccos(-1 + 2*np.random.rand(ncat))
    pos = np.array([thetas, phis], dtype=np.float64)
    w = np.ones(ncat)
    alm = hp.synalm(get_cl_in(lmax=lmax)[1])
    f = nmt.utils._alm2catalog_ducc0(alm, pos, spin=0, lmax=lmax)

    if not get_field:
        return pos, w, f
    else:
        fld = nmt.NmtFieldCatalog(pos, w, f,
                                  lmax=lmax, retain_catalog=True)
        return fld


def test_allsame():
    f = get_cat(seed=1234, get_field=True)
    b = nmt.NmtBin.from_lmax_linear(lmax=f.ainfo_mask.lmax, nlb=10)
    w = nmt.NmtWorkspace.from_fields(f, f, b)
    cw = nmt.NmtCovarianceWorkspace.from_fields(f, f, f, f)
    cl_guess = np.atleast_2d(get_cl_in()[1])
    cl_inka = nmt.get_iNKA_cell(f, f, cl_guess=cl_guess)
    cw.gaussian_covariance(cl_inka, cl_inka, cl_inka, cl_inka,
                           wa=w, wb=w, coupled=True)
    cw.gaussian_covariance(cl_inka, cl_inka, cl_inka, cl_inka,
                           wa=w, wb=w)


def test_alldiff():
    f1 = get_cat(seed=1001, get_field=True)
    f2 = get_cat(seed=1002, get_field=True)
    f3 = get_cat(seed=1003, get_field=True)
    f4 = get_cat(seed=1004, get_field=True)
    b = nmt.NmtBin.from_lmax_linear(lmax=f1.ainfo_mask.lmax, nlb=10)
    wa = nmt.NmtWorkspace.from_fields(f1, f2, b)
    wb = nmt.NmtWorkspace.from_fields(f3, f4, b)
    cw = nmt.NmtCovarianceWorkspace.from_fields(f1, f2, f3, f4)
    cl_guess = np.atleast_2d(get_cl_in()[1])
    cl13_inka = nmt.get_iNKA_cell(f1, f3, cl_guess=cl_guess)
    cl14_inka = nmt.get_iNKA_cell(f1, f4, cl_guess=cl_guess)
    cl23_inka = nmt.get_iNKA_cell(f2, f3, cl_guess=cl_guess)
    cl24_inka = nmt.get_iNKA_cell(f2, f4, cl_guess=cl_guess)
    cw.gaussian_covariance(cl13_inka, cl14_inka, cl23_inka, cl24_inka,
                           wa=wa, wb=wb, coupled=True)
    cw.gaussian_covariance(cl13_inka, cl14_inka, cl23_inka, cl24_inka,
                           wa=wa, wb=wb)
