import pytest
import numpy as np
import pymaster as nmt
import healpy as hp
import warnings
import sys
from .utils import normdiff


class FieldTester(object):
    def __init__(self):
        # This is to avoid showing an ugly warning that
        # has nothing to do with pymaster
        if (sys.version_info > (3, 1)):
            warnings.simplefilter("ignore", ResourceWarning)

        self.nside = 64
        self.lmax = 3*self.nside-1
        self.ntemp = 5
        self.npix = int(hp.nside2npix(self.nside))
        self.msk = np.ones(self.npix)
        self.mps = np.zeros([3, self.npix])
        self.tmp = np.zeros([self.ntemp, 3, self.npix])
        self.beam = np.ones(self.lmax+1)

        th, ph = hp.pix2ang(self.nside, np.arange(self.npix))
        sth = np.sin(th)
        cth = np.cos(th)
        # Re(Y_22)
        self.mps[0] = np.sqrt(15./2./np.pi)*sth**2*np.cos(2*ph)
        # _2Y^E_20 + _2Y^B_30
        self.mps[1] = -np.sqrt(15./2./np.pi)*sth**2/4.
        self.mps[2] = -np.sqrt(105./2./np.pi)*cth*sth**2/2.
        for i in range(self.ntemp):
            # Re(Y_22)
            self.tmp[i][0] = np.sqrt(15./2./np.pi)*sth**2*np.cos(2*ph)
            # _2Y^E_20 + _2Y^B_3 0
            self.tmp[i][1] = -np.sqrt(15./2./np.pi)*sth**2/4.
            self.tmp[i][2] = -np.sqrt(105./2./np.pi)*cth*sth**2/2.


FT = FieldTester()


def test_field_compatibility():
    map_ran = np.random.rand(FT.npix)*0.1+1
    f0 = nmt.NmtField(FT.msk, [FT.mps[0]])
    f1 = nmt.NmtField(FT.msk*map_ran, [map_ran])
    assert f0.is_compatible(f1)

    # Diff. lmax
    f1 = nmt.NmtField(FT.msk*map_ran, [map_ran],
                      lmax=127)
    assert not f0.is_compatible(f1)

    # Diff. lmax_mask
    f1 = nmt.NmtField(FT.msk*map_ran, [map_ran],
                      lmax_mask=127)
    assert not f0.is_compatible(f1)

    # Diff. nside
    npix = hp.nside2npix(128)
    msk = np.ones(npix)
    mp = np.random.randn(npix)
    f1 = nmt.NmtField(msk, [mp])
    assert not f0.is_compatible(f1)

    # Strictness
    # Diff. nside but same lmax
    f0 = nmt.NmtField(FT.msk, [FT.mps[0]], lmax=100, lmax_mask=100)
    f1 = nmt.NmtField(msk, [mp], lmax=100, lmax_mask=100)
    assert f0.is_compatible(f1, strict=False)
    assert not f0.is_compatible(f1)


def test_field_get_mask():
    nside = 32
    npix = hp.nside2npix(nside)
    mp = np.random.randn(1, npix)
    msk = np.random.rand(npix)
    f = nmt.NmtField(msk, mp, n_iter=0)
    mskb = f.get_mask()
    assert np.amax(np.fabs(mskb-msk)/np.std(msk)) < 1E-5
    # Do the same with a big-endian mask
    f = nmt.NmtField(msk.astype('>f8'), mp, n_iter=0)
    mskb = f.get_mask()
    assert np.amax(np.fabs(mskb-msk)/np.std(msk)) < 1E-5


def test_field_get_alms():
    nside = 32
    npix = hp.nside2npix(nside)
    mp = np.random.randn(3, npix)
    msk = np.ones(npix)

    # Spin 0
    f = nmt.NmtField(msk, [mp[0]], n_iter=0)
    alm = f.get_alms()[0]
    cl_tt_nmt = hp.alm2cl(alm)

    # Spin 2
    f = nmt.NmtField(msk, mp[1:], n_iter=0)
    alm = f.get_alms()
    cl_ee_nmt = hp.alm2cl(alm[0])
    cl_bb_nmt = hp.alm2cl(alm[1])

    cl_tt, cl_ee, cl_bb, cl_te, cl_eb, cl_tb = hp.anafast(mp, iter=0,
                                                          pol=True)
    assert (np.all(np.fabs(cl_tt_nmt/cl_tt-1) < 1E-10))
    assert (np.all(np.fabs(cl_ee_nmt[2:]/cl_ee[2:]-1) < 1E-10))
    assert (np.all(np.fabs(cl_bb_nmt[2:]/cl_bb[2:]-1) < 1E-10))


def test_field_map_alm():
    # Compare map-based alms with analytical input and make sure they
    # are equal up to numerical accuracy.
    nside = 32
    npix = int(hp.nside2npix(nside))
    lmax = 20

    msk = np.ones(npix)
    mps = np.zeros([3, npix])
    th, ph = hp.pix2ang(nside, np.arange(npix))
    sth = np.sin(th)
    cth = np.cos(th)
    # Re(Y_22)
    mps[0] = np.sqrt(15./2./np.pi)*sth**2*np.cos(2*ph)
    # _2Y^E_20 + _2Y^B_30
    mps[1] = -np.sqrt(15./2./np.pi)*sth**2/4.
    mps[2] = -np.sqrt(105./2./np.pi)*cth*sth**2/2.

    # spin 0
    f0_map = nmt.NmtField(msk, [mps[0]], lmax=lmax)

    # spin 2
    f2_map = nmt.NmtField(msk, [mps[1], mps[2]], lmax=lmax)

    alms_map = np.array([f0_map.get_alms()[0],
                         f2_map.get_alms()[0],
                         f2_map.get_alms()[1]])

    alms_in = np.zeros_like(alms_map)
    alms_in[0, hp.Alm.getidx(lmax, 2, 2)] = 2.
    alms_in[1, hp.Alm.getidx(lmax, 2, 0)] = 1.
    alms_in[2, hp.Alm.getidx(lmax, 3, 0)] = 2.

    for f_idx in range(3):
        assert np.all(np.absolute(alms_map - alms_in)[f_idx] < 1.e-12)


def test_field_masked():
    nside = 64
    b = nmt.NmtBin.from_nside_linear(nside, 16)
    msk = hp.read_map("test/benchmarks/msk.fits",
                      dtype=float)
    mskb = msk > 0
    mps = np.array(hp.read_map("test/benchmarks/mps.fits",
                               field=[0, 1, 2],
                               dtype=float))*mskb[None, :]
    mps_msk = np.array([m * msk for m in mps])
    f0 = nmt.NmtField(msk, [mps[0]])
    f0_msk = nmt.NmtField(msk, [mps_msk[0]],
                          masked_on_input=True)
    f2 = nmt.NmtField(msk, mps[1:])
    f2_msk = nmt.NmtField(msk, mps_msk[1:],
                          masked_on_input=True)
    w00 = nmt.NmtWorkspace.from_fields(f0, f0, b)
    w02 = nmt.NmtWorkspace.from_fields(f0, f2, b)
    w22 = nmt.NmtWorkspace.from_fields(f2, f2, b)

    def mkcl(w, f, g):
        return w.decouple_cell(nmt.compute_coupled_cell(f, g))

    c00 = mkcl(w00, f0, f0).flatten()
    c02 = mkcl(w02, f0, f2).flatten()
    c22 = mkcl(w22, f2, f2).flatten()
    c00_msk = mkcl(w00, f0_msk, f0_msk).flatten()
    c02_msk = mkcl(w02, f0_msk, f2_msk).flatten()
    c22_msk = mkcl(w22, f2_msk, f2_msk).flatten()
    assert (np.all(np.fabs(c00-c00_msk) /
                   np.mean(c00) < 1E-10))
    assert (np.all(np.fabs(c02-c02_msk) /
                   np.mean(c02) < 1E-10))
    assert (np.all(np.fabs(c22-c22_msk) /
                   np.mean(c22) < 1E-10))


def test_field_masked_pure():
    nside = 64
    b = nmt.NmtBin.from_nside_linear(nside, 16)
    msk = hp.read_map("test/benchmarks/msk.fits",
                      dtype=float)
    mskb = msk > 0
    mps = np.array(hp.read_map("test/benchmarks/mps.fits",
                               field=[1, 2],
                               dtype=float))*mskb[None, :]
    mps_msk = np.array([m * msk for m in mps])
    f2 = nmt.NmtField(msk, mps,
                      templates=[[FT.tmp[0][1]*mskb,
                                  FT.tmp[0][2]*mskb]],
                      purify_b=True)
    f2_msk = nmt.NmtField(msk, mps_msk,
                          templates=[[FT.tmp[0][1]*msk*mskb,
                                      FT.tmp[0][2]*msk*mskb]],
                          masked_on_input=True,
                          purify_b=True)
    w22 = nmt.NmtWorkspace.from_fields(f2, f2, b)

    def mkcl(w, f, g):
        return w.decouple_cell(nmt.compute_coupled_cell(f, g))

    c22 = mkcl(w22, f2, f2).flatten()
    c22_msk = mkcl(w22, f2_msk, f2_msk).flatten()
    assert (np.all(np.fabs(c22-c22_msk) /
                   np.mean(c22) < 1E-10))


def test_field_alloc():
    # No templates
    f0 = nmt.NmtField(FT.msk, [FT.mps[0]],
                      beam=FT.beam)
    f2 = nmt.NmtField(FT.msk, [FT.mps[1], FT.mps[2]],
                      beam=FT.beam)
    f2p = nmt.NmtField(FT.msk, [FT.mps[1], FT.mps[2]],
                       beam=FT.beam,
                       purify_e=True, purify_b=True,
                       n_iter_mask=10)
    assert (normdiff(f0.get_maps()[0],
                     FT.mps[0]*FT.msk) < 1E-10)
    assert (normdiff(f2.get_maps()[0],
                     FT.mps[1]*FT.msk) < 1E-10)
    assert (normdiff(f2.get_maps()[1],
                     FT.mps[2]*FT.msk) < 1E-10)
    assert (1E-5*np.mean(np.fabs(f2p.get_maps()[0])) >
            np.mean(np.fabs(f2p.get_maps()[0] -
                            FT.mps[1]*FT.msk)))
    assert (1E-5*np.mean(np.fabs(f2p.get_maps()[1])) >
            np.mean(np.fabs(f2p.get_maps()[1] -
                            FT.mps[2]*FT.msk)))
    for f in [f0, f2, f2p]:
        with pytest.raises(ValueError):  # No templates
            f.get_templates()

    # With templates
    f0 = nmt.NmtField(FT.msk, [FT.mps[0]],
                      templates=np.array([[t[0]] for t in FT.tmp]),
                      beam=FT.beam)
    f2 = nmt.NmtField(FT.msk, [FT.mps[1], FT.mps[2]],
                      templates=np.array([[t[1], t[2]] for t in FT.tmp]),
                      beam=FT.beam)
    # Map should be zero, since template =  map
    assert (normdiff(f0.get_maps()[0], 0*FT.msk) < 1E-10)
    assert (normdiff(f2.get_maps()[0], 0*FT.msk) < 1E-10)
    assert (normdiff(f2.get_maps()[1], 0*FT.msk) < 1E-10)
    assert (len(f0.get_templates()) == 5)
    assert (len(f2.get_templates()) == 5)


def test_field_lite():
    # Lite field
    fl = nmt.NmtField(FT.msk, [FT.mps[0]],
                      beam=FT.beam, lite=True)
    # Empty field
    with pytest.raises(ValueError):  # No maps and no spin
        fe = nmt.NmtField(FT.msk, None, beam=FT.beam)
    fe = nmt.NmtField(FT.msk, None, beam=FT.beam, spin=1)

    # Error checks
    for f in [fl, fe]:
        with pytest.raises(ValueError):  # Query maps
            f.get_maps()
        with pytest.raises(ValueError):  # Query templates
            f.get_templates()


def test_field_error():
    with pytest.raises(ValueError):  # Incorrect mask size
        nmt.NmtField(FT.msk[:15], FT.mps)
    with pytest.raises(ValueError):  # Incorrect map size
        nmt.NmtField(FT.msk, [FT.mps[0, :15]])
    with pytest.raises(ValueError):  # Incorrect template size
        nmt.NmtField(FT.msk, [FT.mps[0]],
                     templates=[[FT.tmp[0, 0, :15]]])
    with pytest.raises(ValueError):  # Passing 3 maps!
        nmt.NmtField(FT.msk, FT.mps)
    with pytest.raises(ValueError):  # Passing 3 template maps!
        nmt.NmtField(FT.msk, [FT.mps[1], FT.mps[2]],
                     templates=FT.tmp)
    with pytest.raises(ValueError):  # Passing crap as templates
        nmt.NmtField(FT.msk, [FT.mps[1], FT.mps[2]],
                     templates=1)
    with pytest.raises(ValueError):  # Passing wrong beam
        nmt.NmtField(FT.msk, [FT.mps[0]], beam=FT.beam[:30])
    with pytest.raises(ValueError):  # Passing crap as beam
        nmt.NmtField(FT.msk, [FT.mps[0]], beam=1)

    # Automatically assign spin = 0 for a single map
    f = nmt.NmtField(FT.msk, [FT.mps[0]], n_iter=0)
    assert (f.spin == 0)
    # Automatically assign spin = 2 for 2 maps
    f = nmt.NmtField(FT.msk, [FT.mps[1], FT.mps[2]], n_iter=0)
    assert (f.spin == 2)
    with pytest.raises(ValueError):  # Spin=0 but 2 maps
        f = nmt.NmtField(FT.msk, [FT.mps[1], FT.mps[2]],
                         spin=0, n_iter=0)
    with pytest.raises(ValueError):  # Spin=1 but 1 maps
        f = nmt.NmtField(FT.msk, [FT.mps[0]], spin=1, n_iter=0)
    with pytest.raises(ValueError):
        f = nmt.NmtField(FT.msk, [FT.mps[1], FT.mps[2]], spin=1,
                         purify_b=True, n_iter=0)

    # lmax must be zero
    with pytest.raises(ValueError):
        nmt.NmtField(FT.msk, [FT.mps[0]], lmax=0)
    with pytest.raises(ValueError):
        nmt.NmtField(FT.msk, [FT.mps[0]], lmax_mask=0)

    # No anisotropic masks for standard fields
    with pytest.raises(ValueError):
        f = nmt.NmtField(FT.msk, [FT.mps[0]])
        f.get_anisotropic_mask()

    with pytest.raises(ValueError):
        f = nmt.NmtField(FT.msk, [FT.mps[0]])
        f.get_anisotropic_mask_alms()
