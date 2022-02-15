import numpy as np
import pymaster as nmt
import healpy as hp
import warnings
import pytest
import sys
import os


class WorkspaceTester(object):
    def __init__(self):
        # This is to avoid showing an ugly warning that
        # has nothing to do with pymaster
        if (sys.version_info > (3, 1)):
            warnings.simplefilter("ignore", ResourceWarning)

        self.nside = 64
        self.nlb = 16
        self.lmax = 3*self.nside-1
        self.npix = int(hp.nside2npix(self.nside))
        self.msk = hp.read_map("test/benchmarks/msk.fits",
                               verbose=False, dtype=float)
        self.mps = np.array(hp.read_map("test/benchmarks/mps.fits",
                                        verbose=False,
                                        field=[0, 1, 2], dtype=float))
        self.mps_s1 = np.array(hp.read_map("test/benchmarks/mps_sp1.fits",
                                           verbose=False,
                                           field=[0, 1, 2], dtype=float))
        self.tmp = np.array(hp.read_map("test/benchmarks/tmp.fits",
                                        verbose=False,
                                        field=[0, 1, 2], dtype=float))
        self.b = nmt.NmtBin.from_nside_linear(self.nside, self.nlb)
        self.f0 = nmt.NmtField(self.msk,
                               [self.mps[0]])  # Original nside
        self.f2 = nmt.NmtField(self.msk,
                               [self.mps[1], self.mps[2]])
        # Half nside
        self.f0_half = nmt.NmtField(self.msk[:self.npix//4],
                                    [self.mps[0, :self.npix//4]])
        # Small-nside bandpowers
        self.b_half = nmt.NmtBin.from_nside_linear(self.nside//2,
                                                   self.nlb)
        # Large-nside bandposers
        self.b_doub = nmt.NmtBin.from_nside_linear(2*self.nside,
                                                   self.nlb)
        self.n_good = np.zeros([1, 3*self.nside])
        self.n_bad = np.zeros([2, 3*self.nside])
        self.n_half = np.zeros([1, 3*(self.nside//2)])

        dd = np.loadtxt("test/benchmarks/cls_lss.txt", unpack=True)
        l, cltt, clee, clbb, clte, nltt, nlee, nlbb, nlte = dd
        self.ll = l[:3*self.nside]
        self.cltt = cltt[:3*self.nside]
        self.clee = clee[:3*self.nside]
        self.clbb = clbb[:3*self.nside]
        self.clte = clte[:3*self.nside]
        self.nltt = nltt[:3*self.nside]
        self.nlee = nlee[:3*self.nside]
        self.nlbb = nlbb[:3*self.nside]
        self.nlte = nlte[:3*self.nside]


WT = WorkspaceTester()


def test_toeplitz_raises():
    fp = nmt.NmtField(WT.msk, [WT.mps[1], WT.mps[2]],
                      purify_b=True)
    # No Toeplitz with purification
    with pytest.raises(ValueError):
        w = nmt.NmtWorkspace()
        w.compute_coupling_matrix(fp, fp, WT.b,
                                  l_toeplitz=WT.nside,
                                  l_exact=WT.nside//2,
                                  dl_band=10)
    # l_exact is zero
    with pytest.raises(ValueError):
        w = nmt.NmtWorkspace()
        w.compute_coupling_matrix(WT.f2, WT.f2, WT.b,
                                  l_toeplitz=WT.nside,
                                  l_exact=0,
                                  dl_band=10)
    # dl_band is negative
    with pytest.raises(ValueError):
        w = nmt.NmtWorkspace()
        w.compute_coupling_matrix(WT.f2, WT.f2, WT.b,
                                  l_toeplitz=WT.nside,
                                  l_exact=WT.nside//2,
                                  dl_band=-1)
    # l_exact > l_toeplitz
    with pytest.raises(ValueError):
        w = nmt.NmtWorkspace()
        w.compute_coupling_matrix(WT.f2, WT.f2, WT.b,
                                  l_toeplitz=WT.nside,
                                  l_exact=WT.nside+1,
                                  dl_band=10)
    # l_toeplitz > lmax
    with pytest.raises(ValueError):
        w = nmt.NmtWorkspace()
        w.compute_coupling_matrix(WT.f2, WT.f2, WT.b,
                                  l_toeplitz=3*WT.nside,
                                  l_exact=WT.nside//2,
                                  dl_band=10)
    # dl_band > lmax
    with pytest.raises(ValueError):
        w = nmt.NmtWorkspace()
        w.compute_coupling_matrix(WT.f2, WT.f2, WT.b,
                                  l_toeplitz=WT.nside,
                                  l_exact=WT.nside//2,
                                  dl_band=3*WT.nside)


def compare_toeplitz(ce, ct, l_toeplitz, l_exact, dl_band):
    from scipy.linalg import toeplitz

    lmaxp1 = 3*WT.nside
    ls = np.arange(lmaxp1)
    ce = ce / (2*ls[None, :]+1.)
    ct = ct / (2*ls[None, :]+1.)
    de = np.diag(ce)

    # Construct Toeplitz form exact
    # First, pure Toeplitz
    xi = np.divide(ce,
                   np.sqrt(de[:, None]*de[None, :]),
                   out=np.zeros([lmaxp1, lmaxp1]),
                   where=de[:, None]*de[None, :] > 0)
    ind = (np.arange(lmaxp1)+l_toeplitz) % lmaxp1
    rb = toeplitz(xi[ind, l_toeplitz])
    c_toe = rb * np.sqrt(de[:, None]*de[None, :])
    cb = c_toe.copy()
    # Add exact band
    cb[:, :l_exact+1] = ce[:, :l_exact+1]
    # Add diagonals
    for i in range(dl_band+1):
        ind = np.diag_indices(lmaxp1-i)
        cb[i:, :lmaxp1-i][ind] = np.diag(ce, i)
    # Annoying triangle
    xi_col = np.divide(ce[:, l_exact],
                       np.sqrt(de*ce[l_exact, l_exact]),
                       out=np.zeros(lmaxp1), where=de > 0)
    for i in range(l_toeplitz-l_exact):
        ii = i + l_exact
        x = xi_col[lmaxp1-1-l_toeplitz+l_exact:lmaxp1-i]
        d = de[lmaxp1-1-l_toeplitz+ii:]
        di = de[ii]
        cb[lmaxp1-1-l_toeplitz+ii:, ii] = x * np.sqrt(di*d)
    # Correct high-ell box back to toeplitz
    cb[l_toeplitz:, l_toeplitz:] = c_toe[l_toeplitz:, l_toeplitz:]
    # Symmetrize
    ind = np.triu_indices(lmaxp1, k=1)
    cb[ind] = 0
    cb = cb + cb.T - np.diag(np.diag(cb))

    assert (np.sum(np.fabs(cb-ct)) < 1E-10)


def test_toeplitz_00():
    l_toeplitz = WT.nside
    l_exact = WT.nside // 2
    dl_band = WT.nside // 4

    we = nmt.NmtWorkspace()
    we.compute_coupling_matrix(WT.f0, WT.f0, WT.b)
    ce = we.get_coupling_matrix()

    wt = nmt.NmtWorkspace()
    wt.compute_coupling_matrix(WT.f0, WT.f0, WT.b,
                               l_toeplitz=l_toeplitz,
                               l_exact=l_exact,
                               dl_band=dl_band)
    ct = wt.get_coupling_matrix()

    # Check that the approximate matrix is constructed
    # as expected.
    compare_toeplitz(ce, ct,
                     l_toeplitz, l_exact, dl_band)
    # Check that it's not a bad approximation (0.5% for these ells)
    cle = we.decouple_cell(nmt.compute_coupled_cell(WT.f0, WT.f0))
    clt = wt.decouple_cell(nmt.compute_coupled_cell(WT.f0, WT.f0))
    assert np.all(np.fabs(clt[0]/cle[0]-1) < 5E-3)


def test_toeplitz_02():
    l_toeplitz = WT.nside
    l_exact = WT.nside // 2
    dl_band = WT.nside // 4

    we = nmt.NmtWorkspace()
    we.compute_coupling_matrix(WT.f0, WT.f2, WT.b)
    ce = we.get_coupling_matrix().reshape([3*WT.nside, 2,
                                           3*WT.nside, 2])
    ce = ce[:, 0, :, 0]

    wt = nmt.NmtWorkspace()
    wt.compute_coupling_matrix(WT.f0, WT.f2, WT.b,
                               l_toeplitz=l_toeplitz,
                               l_exact=l_exact,
                               dl_band=dl_band)
    ct = wt.get_coupling_matrix().reshape([3*WT.nside, 2,
                                           3*WT.nside, 2])
    ct = ct[:, 0, :, 0]

    # Check that the approximate matrix is constructed
    # as expected.
    compare_toeplitz(ce, ct,
                     l_toeplitz, l_exact, dl_band)
    # Check that it's not a bad approximation (0.5% for these ells)
    cle = we.decouple_cell(nmt.compute_coupled_cell(WT.f0, WT.f2))
    clt = wt.decouple_cell(nmt.compute_coupled_cell(WT.f0, WT.f2))
    assert np.all(np.fabs(clt[0]/cle[0]-1) < 5E-3)


def test_toeplitz_22():
    l_toeplitz = WT.nside
    l_exact = WT.nside // 2
    dl_band = WT.nside // 4

    we = nmt.NmtWorkspace()
    we.compute_coupling_matrix(WT.f2, WT.f2, WT.b)
    ce = we.get_coupling_matrix().reshape([3*WT.nside, 4,
                                           3*WT.nside, 4])
    ce_pp = ce[:, 0, :, 0]
    ce_mm = ce[:, 0, :, 3]

    wt = nmt.NmtWorkspace()
    wt.compute_coupling_matrix(WT.f2, WT.f2, WT.b,
                               l_toeplitz=l_toeplitz,
                               l_exact=l_exact,
                               dl_band=dl_band)
    ct = wt.get_coupling_matrix().reshape([3*WT.nside, 4,
                                           3*WT.nside, 4])
    ct_pp = ct[:, 0, :, 0]
    ct_mm = ct[:, 0, :, 3]

    # Check that the approximate matrix is constructed
    # as expected.
    compare_toeplitz(ce_pp, ct_pp,
                     l_toeplitz, l_exact, dl_band)
    compare_toeplitz(ce_mm, ct_mm,
                     l_toeplitz, l_exact, dl_band)
    # Check that it's not a bad approximation (0.5% for these ells)
    cle = we.decouple_cell(nmt.compute_coupled_cell(WT.f2, WT.f2))
    clt = wt.decouple_cell(nmt.compute_coupled_cell(WT.f2, WT.f2))
    assert np.all(np.fabs(clt[0]/cle[0]-1) < 5E-3)


def test_lite_pure():
    f0 = nmt.NmtField(WT.msk, [WT.mps[0]])
    f2l = nmt.NmtField(WT.msk, [WT.mps[1], WT.mps[2]],
                       purify_b=True, lite=True)
    f2e = nmt.NmtField(WT.msk, None, purify_b=True,
                       lite=True, spin=2)
    nlth = np.array([WT.nlte, 0*WT.nlte])
    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(f0, f2e, WT.b)
    clb = nlth
    cl = w.decouple_cell(nmt.compute_coupled_cell(f0, f2l),
                         cl_bias=clb)
    tl = np.loadtxt("test/benchmarks/bm_nc_yp_c02.txt",
                    unpack=True)[1:, :]
    assert (np.fabs(cl-tl) <= np.fmin(np.fabs(cl),
                                      np.fabs(tl))*1E-5).all()


def test_lite_cont():
    f0 = nmt.NmtField(WT.msk, [WT.mps[0]], templates=[[WT.tmp[0]]])
    f2 = nmt.NmtField(WT.msk, [WT.mps[1], WT.mps[2]],
                      templates=[[WT.tmp[1], WT.tmp[2]]])
    f2l = nmt.NmtField(WT.msk, [WT.mps[1], WT.mps[2]],
                       templates=[[WT.tmp[1], WT.tmp[2]]])
    f2e = nmt.NmtField(WT.msk, None, lite=True, spin=2)
    clth = np.array([WT.clte, 0*WT.clte])
    nlth = np.array([WT.nlte, 0*WT.nlte])
    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(f0, f2e, WT.b)
    clb = nlth
    dlb = nmt.deprojection_bias(f0, f2, clth+nlth)
    clb += dlb
    cl = w.decouple_cell(nmt.compute_coupled_cell(f0, f2l),
                         cl_bias=clb)
    tl = np.loadtxt("test/benchmarks/bm_yc_np_c02.txt",
                    unpack=True)[1:, :]
    tlb = np.loadtxt("test/benchmarks/bm_yc_np_cb02.txt",
                     unpack=True)[1:, :]
    assert (np.fabs(dlb-tlb) <= np.fmin(np.fabs(dlb),
                                        np.fabs(tlb))*1E-5).all()
    assert (np.fabs(cl-tl) <= np.fmin(np.fabs(cl),
                                      np.fabs(tl))*1E-5).all()


def test_spin1():
    prefix = "test/benchmarks/bm_sp1"
    f0 = nmt.NmtField(WT.msk, [WT.mps_s1[0]])
    f1 = nmt.NmtField(WT.msk,
                      [WT.mps_s1[1], WT.mps_s1[2]],
                      spin=1)
    f = [f0, f1]

    for ip1 in range(2):
        for ip2 in range(ip1, 2):
            w = nmt.NmtWorkspace()
            w.compute_coupling_matrix(f[ip1], f[ip2], WT.b)
            cl = w.decouple_cell(nmt.compute_coupled_cell(f[ip1],
                                                          f[ip2]))[0]
            tl = np.loadtxt(prefix+'_c%d%d.txt' % (ip1, ip2),
                            unpack=True)[1]
            assert (np.fabs(cl-tl) <= np.fmin(np.fabs(cl),
                                              np.fabs(tl))*1E-5).all()


def mastest(wtemp, wpure, do_teb=False):
    prefix = "test/benchmarks/bm"
    if wtemp:
        prefix += "_yc"
        f0 = nmt.NmtField(WT.msk, [WT.mps[0]],
                          templates=[[WT.tmp[0]]])
        f2 = nmt.NmtField(WT.msk, [WT.mps[1], WT.mps[2]],
                          templates=[[WT.tmp[1], WT.tmp[2]]],
                          purify_b=wpure)
    else:
        prefix += "_nc"
        f0 = nmt.NmtField(WT.msk, [WT.mps[0]])
        f2 = nmt.NmtField(WT.msk,
                          [WT.mps[1], WT.mps[2]],
                          purify_b=wpure)
    f = [f0, f2]

    if wpure:
        prefix += "_yp"
    else:
        prefix += "_np"

    for ip1 in range(2):
        for ip2 in range(ip1, 2):
            if ip1 == ip2 == 0:
                clth = np.array([WT.cltt])
                nlth = np.array([WT.nltt])
            elif ip1 == ip2 == 1:
                clth = np.array([WT.clee, 0*WT.clee,
                                 0*WT.clbb, WT.clbb])
                nlth = np.array([WT.nlee, 0*WT.nlee,
                                 0*WT.nlbb, WT.nlbb])
            else:
                clth = np.array([WT.clte, 0*WT.clte])
                nlth = np.array([WT.nlte, 0*WT.nlte])
            w = nmt.NmtWorkspace()
            w.compute_coupling_matrix(f[ip1], f[ip2], WT.b)
            clb = nlth
            if wtemp:
                dlb = nmt.deprojection_bias(f[ip1], f[ip2],
                                            clth+nlth)
                tlb = np.loadtxt(prefix+'_cb%d%d.txt' % (2*ip1, 2*ip2),
                                 unpack=True)[1:, :]
                assert ((np.fabs(dlb-tlb) <=
                         np.fmin(np.fabs(dlb),
                                 np.fabs(tlb))*1E-5).all())
                clb += dlb
            cl = w.decouple_cell(nmt.compute_coupled_cell(f[ip1],
                                                          f[ip2]),
                                 cl_bias=clb)
            tl = np.loadtxt(prefix+'_c%d%d.txt' % (2*ip1, 2*ip2),
                            unpack=True)[1:, :]
            assert ((np.fabs(cl-tl) <=
                     np.fmin(np.fabs(cl),
                             np.fabs(tl))*1E-5).all())

    # TEB
    if do_teb:
        clth = np.array([WT.cltt, WT.clte, 0*WT.clte,
                         WT.clee, 0*WT.clee, 0*WT.clbb,
                         WT.clbb])
        nlth = np.array([WT.nltt, WT.nlte, 0*WT.nlte,
                         WT.nlee, 0*WT.nlee, 0*WT.nlbb,
                         WT.nlbb])
        w = nmt.NmtWorkspace()
        w.compute_coupling_matrix(f[0], f[1], WT.b, is_teb=True)
        c00 = nmt.compute_coupled_cell(f[0], f[0])
        c02 = nmt.compute_coupled_cell(f[0], f[1])
        c22 = nmt.compute_coupled_cell(f[1], f[1])
        cl = np.array([c00[0], c02[0], c02[1], c22[0],
                       c22[1], c22[2], c22[3]])
        t00 = np.loadtxt(prefix+'_c00.txt', unpack=True)[1:, :]
        t02 = np.loadtxt(prefix+'_c02.txt', unpack=True)[1:, :]
        t22 = np.loadtxt(prefix+'_c22.txt', unpack=True)[1:, :]
        tl = np.array([t00[0], t02[0], t02[1], t22[0],
                       t22[1], t22[2], t22[3]])
        cl = w.decouple_cell(cl, cl_bias=nlth)
        assert ((np.fabs(cl-tl) <=
                 np.fmin(np.fabs(cl),
                         np.fabs(tl))*1E-5).all())


@pytest.mark.parametrize("wtemp,wpure,do_teb",
                         [(False, False, True),
                          (False, True, True),
                          (False, False, False),
                          (True, False, False),
                          (False, True, False),
                          (True, True, False)])
def test_workspace_master_teb_np(wtemp, wpure, do_teb):
    mastest(wtemp, wpure, do_teb=do_teb)


def test_workspace_shorten():
    w = nmt.NmtWorkspace()
    w.read_from("test/benchmarks/bm_yc_yp_w02.fits")  # OK read
    lmax = w.wsp.lmax
    larr = np.arange(lmax + 1)
    larr_long = np.arange(2 * lmax + 1)
    cls = 100. / (larr + 10.)
    cls_long = 100. / (larr_long + 10.)

    cls_c = w.couple_cell([cls, cls])
    cls_c_long = w.couple_cell([cls_long, cls_long])
    assert np.all(cls_c == cls_c_long)


def test_workspace_rebeam():
    w = nmt.NmtWorkspace()
    w.read_from("test/benchmarks/bm_yc_yp_w02.fits")  # OK read
    lmax = w.wsp.lmax_fields
    b = np.ones(lmax+1)*2.
    w.update_beams(b, b)  # All good
    b2 = np.ones(lmax//2+1)*2.  # Too short
    with pytest.raises(ValueError):
        w.update_beams(b, b2)
    b2 = 1.  # Not array
    with pytest.raises(ValueError):
        w.update_beams(b, b2)


def test_workspace_rebin():
    b4 = nmt.NmtBin.from_nside_linear(WT.nside, 4)
    w = nmt.NmtWorkspace()
    w.read_from("test/benchmarks/bm_yc_yp_w02.fits")  # OK read
    w.update_bins(b4)
    assert (w.wsp.bin.n_bands == b4.bin.n_bands)
    b4 = nmt.NmtBin.from_nside_linear(WT.nside//2, 4)
    with pytest.raises(RuntimeError):  # Wrong lmax
        w.update_bins(b4)


def test_workspace_io():
    w = nmt.NmtWorkspace()
    with pytest.raises(RuntimeError):  # Invalid writing
        w.write_to("test/wspc.fits")
    w.read_from("test/benchmarks/bm_yc_yp_w02.fits")  # OK read
    assert w.wsp.cs.n_eq == 64
    w.get_coupling_matrix()  # Read mode coupling matrix
    # Updating mode-coupling matrix
    mcm_new = np.identity(3*w.wsp.cs.n_eq*2)
    w.update_coupling_matrix(mcm_new)
    # Retireve MCM and check it's correct
    mcm_back = w.get_coupling_matrix()
    assert (np.fabs(np.sum(np.diagonal(mcm_back)) -
                    3*w.wsp.cs.n_eq*2) <= 1E-16)
    with pytest.raises(RuntimeError):  # Can't write on that file
        w.write_to("tests/wspc.fits")
    with pytest.raises(RuntimeError):  # File doesn't exist
        w.read_from("none")


def test_workspace_bandpower_windows():
    # This tests the bandpower window functions returned by NaMaster
    # Compute MCMs
    w00 = nmt.NmtWorkspace()
    w00.compute_coupling_matrix(WT.f0, WT.f0, WT.b)
    w02 = nmt.NmtWorkspace()
    w02.compute_coupling_matrix(WT.f0, WT.f2, WT.b)
    w22 = nmt.NmtWorkspace()
    w22.compute_coupling_matrix(WT.f2, WT.f2, WT.b)

    # Create some random theory power spectra
    larr = np.arange(3*WT.nside)
    cltt = (larr+1.)**-0.8
    clee = cltt.copy()
    clbb = 0.1*clee
    clte = np.sqrt(cltt)*0.01
    cltb = 0.1*clte
    cleb = 0.01*clbb

    # For each spin combination,  test that decouple-couple is the
    # same as bandpass-convolutions.
    def compare_bpw_convolution(cl_th, w):
        cl_dec_a = w.decouple_cell(w.couple_cell(cl_th))
        bpws = w.get_bandpower_windows()
        cl_dec_b = np.einsum('ijkl, kl', bpws, cl_th)
        assert (np.amax(np.fabs(cl_dec_a-cl_dec_b)) <= 1E-10)

    # 00
    compare_bpw_convolution(np.array([cltt]),  w00)
    compare_bpw_convolution(np.array([clte, cltb]),  w02)
    compare_bpw_convolution(np.array([clee, cleb, cleb, clbb]),  w22)


def test_lite_errors():
    f0 = nmt.NmtField(WT.msk, [WT.mps[0]], n_iter=0)
    fl = nmt.NmtField(WT.msk, [WT.mps[0]],
                      templates=[[WT.tmp[0]]], n_iter=0,
                      lite=True)
    with pytest.raises(ValueError):  # Needs spin
        fe = nmt.NmtField(WT.msk, None)
    fe = nmt.NmtField(WT.msk, None, spin=0)

    for f in [fl, fe]:
        with pytest.raises(RuntimeError):  # No deprojection bias
            nmt.deprojection_bias(f0, f, np.zeros([1, 3*WT.nside]))
        with pytest.raises(RuntimeError):  # No deprojection bias
            nmt.uncorr_noise_deprojection_bias(fl, WT.mps[0])
    with pytest.raises(RuntimeError):  # No C_l without maps
        nmt.compute_coupled_cell(f0, fe)


def test_workspace_methods():
    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(WT.f0, WT.f0, WT.b)  # OK init
    assert w.wsp.cs.n_eq == 64
    with pytest.raises(RuntimeError):  # Incompatible bandpowers
        w.compute_coupling_matrix(WT.f0, WT.f0, WT.b_doub)
    with pytest.raises(RuntimeError):  # Incompatible resolutions
        w.compute_coupling_matrix(WT.f0, WT.f0_half, WT.b)
    with pytest.raises(RuntimeError):  # Wrong fields for TEB
        w.compute_coupling_matrix(WT.f0, WT.f0, WT.b, is_teb=True)

    w.compute_coupling_matrix(WT.f0, WT.f0, WT.b)

    # Test couple_cell
    c = w.couple_cell(WT.n_good)
    assert c.shape == (1, w.wsp.lmax+1)
    with pytest.raises(ValueError):
        w.couple_cell(WT.n_bad)
    with pytest.raises(ValueError):
        w.couple_cell(WT.n_half)

    # Test decouple_cell
    c = w.decouple_cell(WT.n_good)
    assert c.shape == (1, WT.b.bin.n_bands)
    with pytest.raises(ValueError):
        w.decouple_cell(WT.n_bad)
    with pytest.raises(ValueError):
        w.decouple_cell(WT.n_half)
    c = w.decouple_cell(WT.n_good, cl_bias=WT.n_good,
                        cl_noise=WT.n_good)
    assert c.shape == (1, WT.b.bin.n_bands)
    with pytest.raises(ValueError):
        w.decouple_cell(WT.n_good, cl_bias=WT.n_good,
                        cl_noise=WT.n_bad)
    with pytest.raises(ValueError):
        w.decouple_cell(WT.n_good, cl_bias=WT.n_good,
                        cl_noise=WT.n_half)
    with pytest.raises(ValueError):
        w.decouple_cell(WT.n_good, cl_bias=WT.n_bad,
                        cl_noise=WT.n_good)
    with pytest.raises(ValueError):
        w.decouple_cell(WT.n_good, cl_bias=WT.n_half,
                        cl_noise=WT.n_good)


def test_workspace_full_master():
    # Test compute_full_master
    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(WT.f0, WT.f0, WT.b)  # OK init
    w_half = nmt.NmtWorkspace()
    w_half.compute_coupling_matrix(WT.f0_half, WT.f0_half, WT.b_half)

    c = nmt.compute_full_master(WT.f0, WT.f0, WT.b)
    assert c.shape == (1, WT.b.bin.n_bands)
    with pytest.raises(RuntimeError):  # Incompatible bandpowers
        nmt.compute_full_master(WT.f0, WT.f0, WT.b_doub)
    with pytest.raises(ValueError):  # Incompatible resolutions
        nmt.compute_full_master(WT.f0, WT.f0_half, WT.b)
    # Passing correct input workspace
    w.compute_coupling_matrix(WT.f0, WT.f0, WT.b)
    # Computing from correct workspace
    c = nmt.compute_full_master(WT.f0, WT.f0, WT.b,
                                workspace=w)
    assert c.shape == (1, WT.b.bin.n_bands)
    with pytest.raises(RuntimeError):  # Inconsistent workspace
        nmt.compute_full_master(WT.f0_half, WT.f0_half,
                                WT.b_half, workspace=w)
    # Incorrect input spectra
    with pytest.raises(ValueError):
        nmt.compute_full_master(WT.f0, WT.f0, WT.b,
                                cl_noise=WT.n_bad)
    with pytest.raises(ValueError):
        nmt.compute_full_master(WT.f0, WT.f0, WT.b,
                                cl_guess=WT.n_bad)
    with pytest.raises(RuntimeError):
        nmt.compute_full_master(WT.f0, WT.f0, WT.b,
                                cl_noise=WT.n_half)
    with pytest.raises(RuntimeError):
        nmt.compute_full_master(WT.f0, WT.f0, WT.b,
                                cl_guess=WT.n_half)


def test_workspace_deprojection_bias():
    # Test deprojection_bias
    c = nmt.deprojection_bias(WT.f0, WT.f0, WT.n_good)
    assert c.shape == (1, WT.f0.fl.lmax+1)
    with pytest.raises(ValueError):
        nmt.deprojection_bias(WT.f0, WT.f0, WT.n_bad)
    with pytest.raises(ValueError):
        nmt.deprojection_bias(WT.f0, WT.f0, WT.n_half)
    with pytest.raises(RuntimeError):
        nmt.deprojection_bias(WT.f0, WT.f0_half, WT.n_good)


def test_workspace_uncorr_noise_deprojection_bias():
    # Test uncorr_noise_deprojection_bias
    c = nmt.uncorr_noise_deprojection_bias(WT.f0, np.zeros(WT.npix))
    assert c.shape == (1, WT.f0.fl.lmax+1)
    with pytest.raises(ValueError):
        nmt.uncorr_noise_deprojection_bias(WT.f0, WT.n_good)


def test_workspace_compute_coupled_cell():
    # Test compute_coupled_cell
    c = nmt.compute_coupled_cell(WT.f0, WT.f0)
    assert c.shape == (1, WT.f0.fl.lmax+1)
    with pytest.raises(ValueError):  # Different resolutions
        nmt.compute_coupled_cell(WT.f0, WT.f0_half)


def test_unbinned_mcm_io():
    f0 = nmt.NmtField(WT.msk, [WT.mps[0]])
    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(f0, f0, WT.b)
    w.write_to("test/wspc.fits")
    assert w.has_unbinned

    w1 = nmt.NmtWorkspace()
    w1.read_from("test/wspc.fits")
    assert w1.has_unbinned

    w2 = nmt.NmtWorkspace()
    w2.read_from("test/wspc.fits", read_unbinned_MCM=False)
    assert w2.has_unbinned is False

    with pytest.raises(ValueError):
        w2.check_unbinned()

    with pytest.raises(ValueError):
        w2.write_to("dum")

    with pytest.raises(ValueError):
        w2.get_coupling_matrix()

    with pytest.raises(ValueError):
        w2.update_coupling_matrix(None)

    with pytest.raises(ValueError):
        w2.couple_cell(np.ones([1, 3*WT.nside]))

    with pytest.raises(ValueError):
        w2.get_bandpower_windows()

    os.system("rm test/wspc.fits")
