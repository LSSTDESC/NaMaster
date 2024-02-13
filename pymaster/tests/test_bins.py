import pytest
import numpy as np
import pymaster as nmt
from .utils import normdiff


class BinTester(object):
    def __init__(self):
        self.nside = 1024
        self.lmax = 2000
        self.nlb = 4
        self.bc = nmt.NmtBin.from_lmax_linear(lmax=self.lmax, nlb=4)
        ells = np.arange(self.lmax - 4, dtype=int)+2
        bpws = (ells - 2)//4
        weights = 0.25*np.ones(self.lmax - 4)
        fell = ells*(ells+1.)/(2*np.pi)
        self.bv = nmt.NmtBin(lmax=self.lmax, bpws=bpws, ells=ells,
                             weights=weights)
        self.bcf = nmt.NmtBin.from_lmax_linear(lmax=self.lmax, nlb=4,
                                               is_Dell=True)
        self.bvf = nmt.NmtBin(lmax=self.lmax, bpws=bpws, ells=ells,
                              weights=weights, f_ell=fell)
        self.l_edges = np.arange(2, self.lmax+2, 4, dtype=int)
        self.be = nmt.NmtBin.from_edges(self.l_edges[:-1],
                                        self.l_edges[1:], is_Dell=True)


BT = BinTester()


def test_bins_nside():
    b1 = nmt.NmtBin.from_nside_linear(nside=256, nlb=4, is_Dell=True)
    b2 = nmt.NmtBin.from_lmax_linear(lmax=3*256-1, nlb=4, is_Dell=True)
    ls1 = b1.get_effective_ells()
    ls2 = b2.get_effective_ells()
    assert np.allclose(ls1, ls2, atol=0, rtol=1E-10)


def test_bins_defaults():
    ells = np.arange(BT.lmax+1, dtype=int)
    bpws = ells // 4
    b = nmt.NmtBin(ells=ells, bpws=bpws)
    # Default ell_max
    assert b.lmax == BT.lmax

    # Default weights
    for i in np.unique(bpws):
        w = b.get_weight_list(i)
        # Equal weights
        assert len(np.unique(w)) == 1
        # Normalised
        assert np.fabs(w[0] - 1/len(w)) < 1E-5


def test_bins_errors():
    # Tests raised exceptions
    ells = np.arange(BT.lmax - 4, dtype=int)+2
    bpws = (ells - 2)//4
    weights = 0.25*np.ones(BT.lmax - 4)
    weights[16:20] = 0
    with pytest.raises(RuntimeError):
        nmt.NmtBin(lmax=BT.lmax,
                   bpws=bpws,
                   ells=ells,
                   weights=weights)
    with pytest.raises(ValueError):
        BT.bv.bin_cell(np.random.randn(3, 3, 3))
    with pytest.raises(ValueError):
        BT.bv.unbin_cell(np.random.randn(3, 3, 3))
    with pytest.raises(TypeError):
        nmt.NmtBin()


def test_bins_nell_list():
    nlst = BT.be.get_nell_list()
    assert len(nlst) == BT.be.get_n_bands()
    assert (nlst == 4).all()


def test_bins_edges():
    # Tests bandpowers generated from edges
    assert BT.bc.get_n_bands() == BT.be.get_n_bands()
    assert np.sum(np.fabs(BT.bc.get_effective_ells() -
                          BT.be.get_effective_ells())) < 1E-10


def test_min_max():
    n = BT.be.get_n_bands()
    assert (BT.be.get_ell_min(0) == BT.l_edges[0])
    assert (BT.be.get_ell_max(0) == BT.l_edges[1] - 1)
    assert (BT.be.get_ell_min(1) == BT.l_edges[1])
    assert (BT.be.get_ell_max(1) == BT.l_edges[2] - 1)
    assert (BT.be.get_ell_min(n - 1) == BT.l_edges[-2])
    assert (BT.be.get_ell_max(n - 1) == BT.l_edges[-1] - 1)


def test_bins_constant():
    # Tests constant bandpower initialization
    assert (BT.bc.get_n_bands() == (BT.lmax - 2)//BT.nlb)
    assert (BT.bc.get_ell_list(5)[2] == 2+BT.nlb*5+2)
    b = nmt.NmtBin.from_lmax_linear(lmax=2000, nlb=4)
    assert (b.bin.ell_max == 2000)


def test_bins_variable():
    # Tests variable bandpower initialization
    assert (BT.bv.get_n_bands() == (BT.lmax - 2)//BT.nlb)
    assert (BT.bv.get_n_bands() == BT.bc.get_n_bands())
    for i in range(BT.bv.get_n_bands()):
        ll1 = BT.bv.get_ell_list(i)
        ll2 = BT.bc.get_ell_list(i)
        wl1 = BT.bv.get_weight_list(i)
        assert (ll1 == ll2).all()
        assert np.fabs(np.sum(wl1) - 1.) < 1E-5
    nbarr = np.arange(BT.bv.get_n_bands())
    assert (normdiff(BT.bv.get_effective_ells(),
                     (2 + BT.nlb * nbarr +
                      0.5 * (BT.nlb - 1))) < 1E-5)


def test_unbin_from_edges():
    bpw_edges = [0, 6, 12, 18, 24, 30, 36, 42, 48,
                 54, 60, 66, 72, 78, 84, 90, 96]
    b = nmt.NmtBin.from_edges(bpw_edges[:-1], bpw_edges[1:])
    cl_b = np.arange(len(bpw_edges)-1)
    cl_u = b.unbin_cell(cl_b)
    for i, (b1, b2) in enumerate(zip(bpw_edges[:-1],
                                     bpw_edges[1:])):
        assert np.all(cl_u[b1:b2] == i)


def test_bins_binning():
    # Tests C_l binning and unbinning
    cls = np.arange(BT.lmax+1, dtype=float)
    cl_b = BT.bv.bin_cell(cls)
    cl_u = BT.bv.unbin_cell(cl_b)
    iend = 2+BT.nlb*((BT.lmax - 2)//BT.nlb)
    cl_b_p = np.mean(cls[2:iend].reshape([-1, BT.nlb]), axis=1)
    assert normdiff(cl_b_p, cl_b) < 1E-5
    cl_u_p = (cl_b[:, None] * np.ones([len(cl_b), BT.nlb])).flatten()
    assert normdiff(cl_u_p, cl_u[2:iend]) < 1E-5


def test_bins_binning_f_ell():
    # Tests C_l binning and unbinning with ell-dependent prefactors
    cls = np.arange(BT.lmax+1, dtype=float)
    fell = cls * (cls + 1.) / 2 / np.pi
    cl_b = BT.bcf.bin_cell(cls)
    assert normdiff(cl_b, BT.bvf.bin_cell(cls)) < 1E-5
    cl_u = BT.bcf.unbin_cell(cl_b)
    assert normdiff(cl_u, BT.bvf.unbin_cell(cl_b)) < 1E-5
    iend = 2+BT.nlb*((BT.lmax - 2)//BT.nlb)
    cl_b_p = np.mean((fell*cls)[2:iend].reshape([-1, BT.nlb]), axis=1)
    assert normdiff(cl_b_p, cl_b) < 1E-5
    cl_u_p = (cl_b[:, None] * np.ones([len(cl_b), BT.nlb])).flatten()
    cl_u_p /= fell[2:2+BT.nlb*((BT.lmax - 2)//BT.nlb)]
    assert normdiff(cl_u_p, cl_u[2:iend]) < 1E-5
