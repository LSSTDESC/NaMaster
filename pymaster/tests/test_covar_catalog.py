import numpy as np
import healpy as hp
import pytest
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


def test_io():
    # Create a covariance workspace
    f = get_cat(seed=1234, get_field=True)
    b = nmt.NmtBin.from_lmax_linear(lmax=f.ainfo_mask.lmax, nlb=10)
    w = nmt.NmtWorkspace.from_fields(f, f, b)
    cw = nmt.NmtCovarianceWorkspace.from_fields(f, f, f, f)
    cl_guess = np.atleast_2d(get_cl_in()[1])
    cl_inka = nmt.get_iNKA_cell(f, f, cl_guess=cl_guess)
    cov = cw.gaussian_covariance(cl_inka, cl_inka, cl_inka, cl_inka,
                                 wa=w, wb=w)

    # Write to file
    cw.write_to("test_cov.fits", fname_NN="test_cov_NN.fits",
                fname_SN="test_cov_SN.fits",
                fname_NS="test_cov_NS.fits")

    # Read from file and check that the covariance is the same
    cw2 = nmt.NmtCovarianceWorkspace.from_file("test_cov.fits",
                                               fname_NN="test_cov_NN.fits",
                                               fname_SN="test_cov_SN.fits",
                                               fname_NS="test_cov_NS.fits")
    cov2 = cw2.gaussian_covariance(cl_inka, cl_inka,
                                   cl_inka, cl_inka, wa=w, wb=w)
    assert np.allclose(cov, cov2)
    # Read again to check that the workspaces are deleted before
    # being read again
    cw2._read_from("test_cov.fits",
                   fname_NN="test_cov_NN.fits",
                   fname_SN="test_cov_SN.fits",
                   fname_NS="test_cov_NS.fits")
    cov3 = cw2.gaussian_covariance(cl_inka, cl_inka,
                                   cl_inka, cl_inka, wa=w, wb=w)
    assert np.allclose(cov, cov3)

    # Repeat this for a workspace with different fields
    f2 = get_cat(seed=1002, get_field=True)
    f3 = get_cat(seed=1003, get_field=True)
    f4 = get_cat(seed=1004, get_field=True)
    b = nmt.NmtBin.from_lmax_linear(lmax=f.ainfo_mask.lmax, nlb=10)
    cw = nmt.NmtCovarianceWorkspace.from_fields(f, f2, f3, f4)
    with pytest.raises(ValueError):  # No maps and no spin
        cw.write_to("test_cov.fits", fname_NN="dummy.fits")
    with pytest.raises(ValueError):  # No maps and no spin
        cw.write_to("test_cov.fits", fname_NS="dummy.fits")
    with pytest.raises(ValueError):  # No maps and no spin
        cw.write_to("test_cov.fits", fname_SN="dummy.fits")


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
    cov = cw.gaussian_covariance(cl13_inka, cl14_inka, cl23_inka, cl24_inka,
                                 wa=wa, wb=wb, coupled=True)

    # Hand-made covariance and compare
    lmax = f1.ainfo_mask.lmax
    ls = np.arange(lmax+1)
    wlms = [hp.almxfl(f.get_mask_alms(),
                      np.exp(-0.5*ls*(ls+1)*f.get_theta_ipd()**2))
            for f in [f1, f2, f3, f4]]
    wmaps = [hp.alm2map(wlm, nside=128) for wlm in wlms]
    cl_w13_w24 = hp.alm2cl(hp.map2alm(wmaps[0]*wmaps[2], lmax=lmax, mmax=lmax),
                           hp.map2alm(wmaps[1]*wmaps[3], lmax=lmax, mmax=lmax))
    cl_w14_w23 = hp.alm2cl(hp.map2alm(wmaps[0]*wmaps[3], lmax=lmax, mmax=lmax),
                           hp.map2alm(wmaps[1]*wmaps[2], lmax=lmax, mmax=lmax))
    xi_13_24 = nmt.get_general_coupling_matrix(cl_w13_w24,
                                               0, 0, 0, 0)/(2*ls+1)[None, :]
    xi_14_23 = nmt.get_general_coupling_matrix(cl_w14_w23,
                                               0, 0, 0, 0)/(2*ls+1)[None, :]
    cl_13_24 = 0.5*(cl13_inka[:, :, None]*cl24_inka[:, None, :] +
                    cl24_inka[:, :, None]*cl13_inka[:, None, :]).squeeze()
    cl_14_23 = 0.5*(cl14_inka[:, :, None]*cl23_inka[:, None, :] +
                    cl23_inka[:, :, None]*cl14_inka[:, None, :]).squeeze()
    cov_mine = (cl_13_24 * xi_13_24 + cl_14_23 * xi_14_23)
    assert np.allclose(cov, cov_mine)
