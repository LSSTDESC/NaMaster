import numpy as np
import healpy as hp
import pymaster as nmt
import warnings
import sys


class FieldTesterCatalog(object):
    def __init__(self):
        # This is to avoid showing an ugly warning that
        # has nothing to do with pymaster
        if (sys.version_info > (3, 1)):
            warnings.simplefilter("ignore", ResourceWarning)
        nside = 64
        self.ncat = hp.nside2npix(nside)
        self.f = np.zeros([3, self.ncat])
        self.w = np.ones(self.ncat)
        self.lmax = 2*nside - 1

        th, ph = hp.pix2ang(nside, np.arange(self.ncat))
        self.p = np.array([th, ph])
        sth = np.sin(th)
        cth = np.cos(th)
        # Re(Y_22)
        self.f[0] = np.sqrt(15./2./np.pi)*sth**2*np.cos(2*ph)
        # _2Y^E_20 + _2Y^B_30
        self.f[1] = -np.sqrt(15./2./np.pi)*sth**2/4.
        self.f[2] = -np.sqrt(105./2./np.pi)*cth*sth**2/2.


FT = FieldTesterCatalog()


def test_field_catalog_compatibility():
    # Different field values
    f_rand = np.random.rand(FT.ncat)*0.1 + 1
    f0 = nmt.NmtFieldCatalog(FT.p, FT.w, FT.f[0], FT.lmax)
    f1 = nmt.NmtFieldCatalog(FT.p, FT.w, f_rand, FT.lmax)
    assert f0.is_compatible(f1)

    # Different positions
    p1 = np.array([np.random.permutation(FT.p[0]), FT.p[1]])
    f1 = nmt.NmtFieldCatalog(p1, FT.w, f_rand, FT.lmax)
    assert f0.is_compatible(f1)

    # Different weights
    w_rand = np.random.rand(FT.ncat)*0.1 + 1
    f1 = nmt.NmtFieldCatalog(FT.p, w_rand, f_rand, FT.lmax)
    assert f0.is_compatible(f1)

    # Different lmax
    print(f0.ainfo_mask.lmax)
    f1 = nmt.NmtFieldCatalog(FT.p, w_rand, f_rand, lmax=111)
    assert not f0.is_compatible(f1)

    # Different lmax_mask
    f1 = nmt.NmtFieldCatalog(FT.p, w_rand, f_rand, FT.lmax, lmax_mask=300)
    assert not f0.is_compatible(f1)


def test_field_catalog_init():
    # Checks correct initialization of positions (lon/lat and theta/phi),
    # weights, fields, different spins.
    Ncat = 100
    lmax = 10
    np.random.seed(5675)
    w0 = np.zeros(Ncat)
    val_s0 = np.random.rand(Ncat) - 0.5
    val_s2 = np.random.rand(2, Ncat) - 0.5
    col_rad = np.pi*np.random.rand(Ncat)
    lon_rad = 2*np.pi*np.random.rand(Ncat)
    lon_deg = 180./np.pi*lon_rad
    lat_deg = 90. - 180./np.pi*col_rad

    for ndim, vals in zip([1, 2], [val_s0, val_s2]):
        f = nmt.NmtFieldCatalog([col_rad, lon_rad], w0, vals,
                                lmax, field_is_weighted=True)
        assert np.array_equal(f.field, vals)
        assert ndim == f.field.ndim
        f = nmt.NmtFieldCatalog([col_rad, lon_rad], w0, vals,
                                lmax, field_is_weighted=False)
        assert not np.any(f.field)
        assert ndim == f.field.ndim

    for vals in [val_s0, val_s2]:
        f1 = nmt.NmtFieldCatalog([col_rad, lon_rad], w0, vals,
                                 lmax, lonlat=False)
        f2 = nmt.NmtFieldCatalog([lon_deg, lat_deg], w0, vals,
                                 lmax, lonlat=True)
        assert np.array_equal(f1.field, f2.field)


def test_field_catalog_Nw():
    # Check if Nw matches mask pcl in noise-dominated regime.
    Ncat = 100
    lmax = 100
    lmin = 50
    Nsims = 10
    w1 = np.ones(Ncat)

    Nw = []
    Nw_true = Ncat/(4.*np.pi)

    pcl = []
    for i in range(Nsims):
        np.random.seed(5675 + i)
        val_s0 = np.random.rand(Ncat) - 0.5
        col_rad = np.pi*np.random.rand(Ncat)
        lon_rad = 2*np.pi*np.random.rand(Ncat)
        f = nmt.NmtFieldCatalog([col_rad, lon_rad], w1, val_s0, lmax)
        pcl_mask = hp.alm2cl(f.get_mask_alms())
        ell = np.arange(len(pcl_mask))
        msk = ell > lmin
        pcl.append(pcl_mask[msk])
        assert f.Nw == Nw_true
        Nw.append(f.Nw)

    y = np.mean(np.array(pcl) - np.array(Nw)[:, None], axis=0)
    yerr = np.std(np.array(pcl) - np.array(Nw)[:, None], axis=0)/np.sqrt(Nsims)

    assert all(np.logical_and(v < 5, v > -5) for v in y/yerr)


def test_field_catalog_Nf():
    # Check if Nf is mathematically correct and only subtracted
    # for equal fields.
    Ncat = 100
    lmax = 100

    np.random.seed(5675)
    val_s0 = np.random.rand(Ncat) - 0.5
    val_s2 = np.random.rand(2, Ncat) - 0.5
    col_rad = np.pi*np.random.rand(Ncat)
    lon_rad = 2*np.pi*np.random.rand(Ncat)

    for val in [val_s0, val_s2]:
        nd = val.ndim
        idx = -1 if nd == 1 else (0, -1)

        # Perturb val to extract Nf correction
        val2 = np.copy(val)
        val2[idx] *= (1. + 1.e-6)
        f1 = nmt.NmtFieldCatalog([col_rad, lon_rad], np.ones(Ncat), val, lmax)
        f2 = nmt.NmtFieldCatalog([col_rad, lon_rad], np.ones(Ncat), val2, lmax)

        cl1 = nmt.compute_coupled_cell(f1, f1)
        cl2 = nmt.compute_coupled_cell(f1, f2)
        nl = np.shape(cl1)[-1]
        diff_true = np.sum(f1.field**2)/(4*np.pi*nd)*np.eye(nd).reshape(nd**2)
        diff_true = diff_true[:, None]*np.ones((nd**2, nl))

        assert not np.any(np.fabs(cl2 - cl1 - diff_true) > 1.e-6)


def test_field_catalog_alm():
    # Compare catalog-based alms with analytical input and make sure they
    # are equal up to numerical accuracy.
    nside = 32
    npix = int(hp.nside2npix(nside))
    pixel_area = np.pi*4./npix
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
    f0_cat = nmt.NmtFieldCatalog([th, ph], np.ones(npix), mps[0], lmax)
    f0_map = nmt.NmtField(msk, [mps[0]], lmax=lmax)

    # spin 2
    f2_cat = nmt.NmtFieldCatalog([th, ph], np.ones(npix),
                                 np.array([mps[1], mps[2]]), lmax)
    f2_map = nmt.NmtField(msk, [mps[1], mps[2]], lmax=lmax)

    alms_map = np.array([f0_map.get_alms()[0],
                         f2_map.get_alms()[0],
                         f2_map.get_alms()[1]])
    alms_cat = np.array([pixel_area*f0_cat.get_alms()[0],
                         pixel_area*f2_cat.get_alms()[0],
                         pixel_area*f2_cat.get_alms()[1]])

    alms_in = np.zeros_like(alms_map)
    alms_in[0, hp.Alm.getidx(lmax, 2, 2)] = 2.
    alms_in[1, hp.Alm.getidx(lmax, 2, 0)] = 1.
    alms_in[2, hp.Alm.getidx(lmax, 3, 0)] = 2.

    for f_idx in range(3):
        assert np.all(np.fabs(np.real(alms_cat - alms_in)[f_idx]) < 1.e-3)
        assert np.all(np.fabs(np.imag(alms_cat - alms_in)[f_idx]) < 1.e-3)


def test_field_catalog_errors():
    import pytest

    with pytest.raises(ValueError):  # Incorrect field size
        nmt.NmtFieldCatalog(np.zeros((2, 100)), np.zeros(99), np.zeros(99), 10)
    with pytest.raises(ValueError):  # Passing 3 fields
        nmt.NmtFieldCatalog([[0., 0.], [1., 1.]], [1., 1.],
                            [[1., 1.], [1., 1.], [1., 1.]], 10)
    with pytest.raises(ValueError):  # Passing single angle
        nmt.NmtFieldCatalog([[0., 0.]], [1., 1.], [1., 1.], 10)
    with pytest.raises(ValueError):  # Trash angles (th, phi)
        nmt.NmtFieldCatalog([[-1., 0.], [1., 1.]], [1., 1.], [1., 1.], 10)
    with pytest.raises(ValueError):  # Trash angles (lon, lat)
        nmt.NmtFieldCatalog([[-45., 0.], [30., 120.]], [1., 1.], [1., 1.], 10,
                            lonlat=True)
    with pytest.raises(ValueError):  # Passing crap beam
        nmt.NmtFieldCatalog([[0., 0.], [1., 1.]], [1., 1.], [1., 1.], 10,
                            beam=1)
    # Automatically assign spin = 0 for a single field
    f = nmt.NmtFieldCatalog([[0., 0.], [1., 1.]], [1., 1.], [1., 1.], 10)
    assert (f.spin == 0)
    # Automatically assign spin = 2 for 2 fields
    f = nmt.NmtFieldCatalog([[0., 0.], [1., 1.]], [1., 1.],
                            [[1., 1.], [1., 1.]], 10)
    assert (f.spin == 2)
    with pytest.raises(ValueError):  # Spin = 0 but 2 maps
        f = nmt.NmtFieldCatalog([[0., 0.], [1., 1.]], [1., 1.],
                                [[1., 1.], [1., 1.]], 10, spin=0)
    with pytest.raises(ValueError):  # Spin = 2 but single map
        f = nmt.NmtFieldCatalog([[0., 0.], [1., 1.]], [1., 1.],
                                [1., 1.], 10, spin=2)
