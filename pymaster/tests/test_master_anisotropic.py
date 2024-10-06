import numpy as np
import healpy as hp
import pymaster as nmt
import pytest


def test_anisotropic_weighting_smoke():
    nside = 64
    npix = hp.nside2npix(nside)
    m = np.ones(npix)

    f = nmt.NmtField(m, [m, m],
                     mask_12=0.5*m,
                     mask_22=2.0*m)
    msk_a = f.get_anisotropic_mask()

    assert np.allclose(msk_a[1], 0.5*m)
    assert np.allclose(msk_a[0], -0.5*m)


def test_anisotropic_weighting_errors():
    nside = 64
    npix = hp.nside2npix(nside)
    m = np.ones(npix)

    # Field-level errors
    # Must pass full weight matrix
    with pytest.raises(ValueError):
        nmt.NmtField(m, [m, m], mask_12=m)

    # Non-positive-definite weight matrix
    with pytest.raises(ValueError):
        nmt.NmtField(m, [m, m], mask_22=0*m, mask_12=m)

    # No anisotropic matrix for scalar fields
    with pytest.raises(ValueError):
        nmt.NmtField(m, None, spin=0, mask_22=m, mask_12=m*0)

    # No anisotropic matrix for scalar fields (second check)
    with pytest.raises(ValueError):
        nmt.NmtField(m, [m], mask_22=m, mask_12=m*0)

    # No purification with anisotropic weights
    with pytest.raises(NotImplementedError):
        nmt.NmtField(m, [m, m], mask_12=0*m, mask_22=m,
                     purify_b=True)

    # No contaminant deprojection with anisotropic weights
    with pytest.raises(NotImplementedError):
        nmt.NmtField(m, [m, m], mask_12=0*m, mask_22=m,
                     templates=[[m, m]])

    # Workspace-level errors
    f = nmt.NmtField(m, [m, m],
                     mask_12=0.5*m,
                     mask_22=2.0*m)
    b = nmt.NmtBin.from_nside_linear(nside, nlb=4)
    with pytest.raises(NotImplementedError):
        nmt.NmtWorkspace.from_fields(f, f, b, l_toeplitz=3)

    # Covariance-level errors
    with pytest.raises(NotImplementedError):
        nmt.NmtCovarianceWorkspace.from_fields(f, f, f, f)


def test_anisotropic_weighting():
    # Test parameters
    nside = 64
    spin = 2
    nlb = 4
    nsims = 100

    # Create anisotropic weights
    mask = hp.read_map("test/benchmarks/msk.fits")
    mask = hp.ud_grade(mask, nside_out=nside)
    mask = hp.smoothing(mask, sigma=np.radians(1.0))
    mask[mask > 1] = 1
    mask[mask < 0.001] = 0

    delta_m = 0.9
    r_m = 0.5
    w11 = (1+delta_m)*mask
    w22 = (1-delta_m)*mask
    w12 = r_m*np.sqrt(w11*w22)

    # Input power spectra
    ls = np.arange(3*nside)
    cl_temp = 1/(ls+10)
    cl_tt = 1.5*cl_temp
    cl_te = 0.6*cl_temp
    cl_tb = 0.3*cl_temp
    cl_ee = 1.0*cl_temp
    cl_eb = 0.2*cl_temp
    cl_bb = 0.4*cl_temp

    # Workspaces
    b = nmt.NmtBin.from_nside_linear(nside, nlb=nlb)
    leff = b.get_effective_ells()
    f0 = nmt.NmtField(mask, None, spin=0, n_iter=0)
    fs = nmt.NmtField(mask, None, spin=spin, n_iter=0)
    fsa = nmt.NmtField(w11, None, spin=spin, n_iter=0,
                       mask_12=w12, mask_22=w22)
    w0sa = nmt.NmtWorkspace.from_fields(f0, fsa, b)
    wsa0 = nmt.NmtWorkspace.from_fields(fsa, f0, b)
    wsasa = nmt.NmtWorkspace.from_fields(fsa, fsa, b)
    wssa = nmt.NmtWorkspace.from_fields(fs, fsa, b)
    wsas = nmt.NmtWorkspace.from_fields(fsa, fs, b)

    # Run simulations
    cl0sa_s = []
    clsa0_s = []
    clsasa_s = []
    clssa_s = []
    clsas_s = []
    for i in range(nsims):
        almt, alme, almb = hp.synalm([cl_tt, cl_ee, cl_bb,
                                      cl_te, cl_eb, cl_tb], new=True)
        map_t = hp.alm2map(almt, nside, lmax=3*nside-1)
        map_q, map_u = hp.alm2map_spin([alme, almb], nside, spin,
                                       lmax=3*nside-1, mmax=3*nside-1)

        f0 = nmt.NmtField(mask, [map_t], n_iter=0)
        fs = nmt.NmtField(mask, [map_q, map_u], spin=spin, n_iter=0)
        fsa = nmt.NmtField(w11, [map_q, map_u], spin=spin, n_iter=0,
                           mask_12=w12, mask_22=w22)

        cl0sa_s.append(w0sa.decouple_cell(nmt.compute_coupled_cell(f0, fsa)))
        clsa0_s.append(wsa0.decouple_cell(nmt.compute_coupled_cell(fsa, f0)))
        clsasa_s.append(wsasa.decouple_cell(nmt.compute_coupled_cell(fsa,
                                                                     fsa)))
        clssa_s.append(wssa.decouple_cell(nmt.compute_coupled_cell(fs, fsa)))
        clsas_s.append(wsas.decouple_cell(nmt.compute_coupled_cell(fsa, fs)))
    cl0sa_s = np.array(cl0sa_s)
    clsa0_s = np.array(clsa0_s)
    clsasa_s = np.array(clsasa_s)
    clssa_s = np.array(clssa_s)
    clsas_s = np.array(clsas_s)
    # Mean
    cl0sa_m = np.mean(cl0sa_s, axis=0)
    clsa0_m = np.mean(clsa0_s, axis=0)
    clsasa_m = np.mean(clsasa_s, axis=0)
    clssa_m = np.mean(clssa_s, axis=0)
    clsas_m = np.mean(clsas_s, axis=0)
    # STD
    cl0sa_e = np.std(cl0sa_s, axis=0)
    clsa0_e = np.std(clsa0_s, axis=0)
    clsasa_e = np.std(clsasa_s, axis=0)
    clssa_e = np.std(clssa_s, axis=0)
    clsas_e = np.std(clsas_s, axis=0)
    # Truth
    cl0sa_t = w0sa.decouple_cell(w0sa.couple_cell([cl_te, cl_tb]))
    clsa0_t = wsa0.decouple_cell(wsa0.couple_cell([cl_te, cl_tb]))
    clsasa_t = wsasa.decouple_cell(wsasa.couple_cell([cl_ee, cl_eb,
                                                      cl_eb, cl_bb]))
    clssa_t = wssa.decouple_cell(wssa.couple_cell([cl_ee, cl_eb,
                                                   cl_eb, cl_bb]))
    clsas_t = wsas.decouple_cell(wsas.couple_cell([cl_ee, cl_eb,
                                                   cl_eb, cl_bb]))

    # Compare all power spectra and check for > 6sigma deviations
    def comp_cl(clm, cle, clt, nsigma=6):
        lgood = leff < 2*nside
        for m, e, t in zip(clm, cle, clt):
            r = ((m-t)*np.sqrt(nsims)/e)[lgood]
            if np.any(np.fabs(r) > nsigma):
                return False
        return True

    test_0sa = comp_cl(cl0sa_m, cl0sa_e, cl0sa_t)
    test_sa0 = comp_cl(clsa0_m, clsa0_e, clsa0_t)
    test_sasa = comp_cl(clsasa_m, clsasa_e, clsasa_t)
    test_ssa = comp_cl(clssa_m, clssa_e, clssa_t)
    test_sas = comp_cl(clsas_m, clsas_e, clsas_t)

    assert np.all([test_0sa, test_sa0, test_sasa, test_ssa, test_sas])
