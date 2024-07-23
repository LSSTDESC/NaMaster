import numpy as np
import healpy as hp
import pymaster as nmt


def test_field_catalog_compatibility():
    # Different field values
    nside = 64
    ncat = hp.nside2npix(nside)
    th, ph = hp.pix2ang(nside, np.arange(ncat))
    p = np.array([th, ph])
    f = np.zeros([3, ncat])
    w = np.ones(ncat)
    lmax = 2*nside - 1
    f_rand = np.random.rand(ncat)*0.1 + 1
    f0 = nmt.NmtFieldCatalog(p, w, f[0], lmax)
    f1 = nmt.NmtFieldCatalog(p, w, f_rand, lmax)
    assert f0.is_compatible(f1)

    # Different positions
    p1 = np.array([np.random.permutation(p[0]), p[1]])
    f1 = nmt.NmtFieldCatalog(p1, w, f_rand, lmax)
    assert f0.is_compatible(f1)

    # Different weights
    w_rand = np.random.rand(ncat)*0.1 + 1
    f1 = nmt.NmtFieldCatalog(p, w_rand, f_rand, lmax)
    assert f0.is_compatible(f1)

    # Different lmax
    f1 = nmt.NmtFieldCatalog(p, w_rand, f_rand, lmax=111)
    assert not f0.is_compatible(f1)

    # Different lmax_mask
    f1 = nmt.NmtFieldCatalog(p, w_rand, f_rand, lmax, lmax_mask=300)
    assert not f0.is_compatible(f1)


def test_field_catalog_init():
    # Checks correct initialization of positions (lon/lat and theta/phi),
    # weights, fields, different spins.
    Ncat = 100
    lmax = 70
    np.random.seed(5675)
    w1 = np.random.rand(Ncat)
    val_s0 = np.random.rand(Ncat) - 0.5
    val_s2 = np.random.rand(2, Ncat) - 0.5
    col_rad = np.arccos(-1+2*np.random.rand(Ncat))
    lon_rad = 2*np.pi*np.random.rand(Ncat)
    lon_deg = np.rad2deg(lon_rad)
    lat_deg = np.rad2deg(np.pi/2. - col_rad)

    for ndim, vals in zip([1, 2], [val_s0, val_s2]):
        f = nmt.NmtFieldCatalog([col_rad, lon_rad], w1, vals,
                                lmax, field_is_weighted=True)
        assert np.array_equal(f.field.squeeze(), vals)
        assert ndim == len(f.field)
        f = nmt.NmtFieldCatalog([col_rad, lon_rad], w1, vals,
                                lmax, field_is_weighted=False)
        assert np.array_equal(f.field.squeeze(), vals*w1)
        assert ndim == len(f.field)

    for vals in [val_s0, val_s2]:
        f1 = nmt.NmtFieldCatalog([col_rad, lon_rad], w1, vals,
                                 lmax, lonlat=False)
        f2 = nmt.NmtFieldCatalog([lon_deg, lat_deg], w1, vals,
                                 lmax, lonlat=True)
        # Observation: after index 4lmax, the difference starts to become O(1).
        assert np.all(np.absolute(f1.alm_mask - f2.alm_mask)[:4*lmax] < 1E-7)


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
        col_rad = np.arccos(-1+2*np.random.rand(Ncat))
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
    col_rad = np.arccos(-1+2*np.random.rand(Ncat))
    lon_rad = 2*np.pi*np.random.rand(Ncat)

    for val in [val_s0, val_s2]:
        nd = val.ndim

        f1 = nmt.NmtFieldCatalog([col_rad, lon_rad], np.ones(Ncat), val, lmax)
        f2 = nmt.NmtFieldCatalog([col_rad, lon_rad], np.ones(Ncat), val, lmax)
        # Note that NmtFieldCatalog.Nf is called internally by
        # compute_coupled_cell(f1, f2) and subtracted if f1 and f2 are the
        # same object.
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
    pixel_area = hp.nside2pixarea(nside)
    lmax = 20

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

    # spin 2
    f2_cat = nmt.NmtFieldCatalog([th, ph], np.ones(npix), mps[1:], lmax)

    alms_cat = pixel_area * np.array(list(f0_cat.get_alms()) +
                                     list(f2_cat.get_alms()))

    alms_in = np.zeros_like(alms_cat)
    alms_in[0, hp.Alm.getidx(lmax, 2, 2)] = 2.
    alms_in[1, hp.Alm.getidx(lmax, 2, 0)] = 1.
    alms_in[2, hp.Alm.getidx(lmax, 3, 0)] = 2.

    assert np.all(np.absolute(alms_cat - alms_in) < 1.e-3)


def test_field_catalog_errors():
    import pytest

    with pytest.raises(ValueError):  # Incorrect field size
        nmt.NmtFieldCatalog(np.zeros((2, 100)), np.zeros(99), np.zeros(99), 10)
    with pytest.raises(ValueError):  # Passing 3 fields
        nmt.NmtFieldCatalog([[0., 0.], [1., 1.]], [1., 1.],
                            [[1., 1.], [1., 1.], [1., 1.]], 10)
    with pytest.raises(ValueError):  # Passing single angle
        nmt.NmtFieldCatalog([[0., 0.]], [1., 1.], [1., 1.], 10)
    with pytest.raises(ValueError):
        nmt.NmtFieldCatalogClustering([[0., 0.]], [1., 1.],
                                      [[0., 0.]], [1., 1.], 10)
    with pytest.raises(ValueError):  # Trash latitude (th, phi)
        nmt.NmtFieldCatalog([[-1., 0.], [1., 1.]], [1., 1.], [1., 1.], 10)
    with pytest.raises(ValueError):
        nmt.NmtFieldCatalogClustering([[-1., 0.], [1., 1.]], [1., 1.],
                                      [[-1., 0.], [1., 1.]], [1., 1.], 10)
    with pytest.raises(ValueError):  # Trash longitude (th, phi)
        nmt.NmtFieldCatalog([[0., 0.], [-1., 1.]], [1., 1.], [1., 1.], 10)
    with pytest.raises(ValueError):
        nmt.NmtFieldCatalogClustering([[0., 0.], [-1., 1.]], [1., 1.],
                                      [[0., 0.], [-1., 1.]], [1., 1.], 10)
    with pytest.raises(ValueError):  # Trash latitude (lonlat=True)
        nmt.NmtFieldCatalog([[0., 0.], [930., 420.]], [1., 1.], [1., 1.], 10,
                            lonlat=True)
    with pytest.raises(ValueError):
        nmt.NmtFieldCatalogClustering([[0., 0.], [930., 420.]], [1., 1.],
                                      [[0., 0.], [930., 420.]], [1., 1.],
                                      10, lonlat=True)
    with pytest.raises(ValueError):  # Trash longitude (lonlat=True)
        nmt.NmtFieldCatalog([[-10., 0.], [45., 45.]], [1., 1.], [1., 1.], 10,
                            lonlat=True)
    with pytest.raises(ValueError):
        nmt.NmtFieldCatalogClustering([[-10., 0.], [45., 45.]], [1., 1.],
                                      [[-10., 0.], [45., 45.]], [1., 1.],
                                      10, lonlat=True)

    f = nmt.NmtFieldCatalog(  # Check beam
        [[0., 0.], [1., 1.]], [1., 1.], [1., 1.], 10, beam=np.ones(20)
    )
    assert np.array_equal(f.beam, np.ones(20))
    with pytest.raises(ValueError):  # Passing crap beam
        nmt.NmtFieldCatalog([[0., 0.], [1., 1.]], [1., 1.], [1., 1.], 10,
                            beam=1)
    with pytest.raises(ValueError):  # Passing mismatching beam
        nmt.NmtFieldCatalog([[0., 0.], [1., 1.]], [1., 1.], [1., 1.], 10,
                            beam=np.ones(10))

    f = nmt.NmtFieldCatalog(  # Check spin
        [[0., 0.], [1., 1.]], [1., 1.], [[1., 1.], [1., 1.]], 10, spin=2
    )
    assert (f.spin == 2)
    f = nmt.NmtFieldCatalog(  # Spin provided if field is None
        [[0., 0.], [1., 1.]], [1., 1.], None, 10, spin=2
    )
    assert (f.spin == 2)
    with pytest.raises(ValueError):  # Spin = 2 but single map
        nmt.NmtFieldCatalog([[0., 0.], [1., 1.]], [1., 1.], [1., 1.], 10,
                            spin=2)
    # Automatically assign spin = 0 for a single field
    f = nmt.NmtFieldCatalog([[0., 0.], [1., 1.]], [1., 1.], [1., 1.], 10)
    assert (f.spin == 0)

    # Can't access mask
    with pytest.raises(ValueError):
        f.get_mask()

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
    with pytest.raises(ValueError):  # Field is none but spin is not provided
        f = nmt.NmtFieldCatalog([[0., 0.], [1., 1.]], [1., 1.], None, 10)

    nmt.utils.HAVE_DUCC = False  # Fake us not having ducc
    with pytest.raises(ValueError):
        nmt.NmtFieldCatalog([[0., 0.], [1., 1.]], [1., 1.],
                            [[1., 1.], [1., 1.]], 10)
    with pytest.raises(ValueError):
        nmt.NmtFieldCatalogClustering([[0., 0.], [1., 1.]], [1., 1.],
                                      [[0., 0.], [1., 1.]], [1., 1.], 10)
    nmt.utils.HAVE_DUCC = True


def test_field_catalog_clustering_poisson():
    # Checks that a purely Poisson catalog has a power spectrum
    # that is close to zero.

    # Create random catalog
    nside = 64
    npix = hp.nside2npix(nside)
    pos_ran = np.array([np.arccos(-1+2*np.random.rand(4*npix)),
                        2*np.pi*np.random.rand(4*npix)])
    w_ran = np.ones(4*npix)
    # We'll do a full-sky version and a half-sky version
    goodhalfran = pos_ran[0] <= np.pi/2

    # Create dummy field for workspaces
    b = nmt.NmtBin.from_nside_linear(nside, 4)
    f = nmt.NmtFieldCatalogClustering(pos_ran, w_ran, pos_ran, w_ran,
                                      lmax=3*nside-1)
    w = nmt.NmtWorkspace.from_fields(f, f, b)
    # Half-sky version
    f = nmt.NmtFieldCatalogClustering(pos_ran[:, goodhalfran],
                                      w_ran[goodhalfran],
                                      pos_ran[:, goodhalfran],
                                      w_ran[goodhalfran],
                                      lmax=3*nside-1)
    wh = nmt.NmtWorkspace.from_fields(f, f, b)

    nsims = 10
    ncat = npix//4
    cls_full = []
    cls_half = []
    w_dat = np.ones(ncat)
    for i in range(nsims):
        # Generate sim and get C_ell
        pos_dat = np.array([np.arccos(-1+2*np.random.rand(ncat)),
                            2*np.pi*np.random.rand(ncat)])
        f = nmt.NmtFieldCatalogClustering(pos_dat, w_dat, pos_ran, w_ran,
                                          lmax=3*nside-1)
        assert np.isclose(f.alpha, 1/16)
        cls_full.append(w.decouple_cell(nmt.compute_coupled_cell(f, f)))

        # Half-sky version
        goodhalfdat = pos_dat[0] <= np.pi/2
        f = nmt.NmtFieldCatalogClustering(pos_dat[:, goodhalfdat],
                                          w_dat[goodhalfdat],
                                          pos_ran[:, goodhalfran],
                                          w_ran[goodhalfran],
                                          lmax=3*nside-1)
        cls_half.append(wh.decouple_cell(nmt.compute_coupled_cell(f, f)))
    cls_full = np.array(cls_full).squeeze()
    cls_half = np.array(cls_half).squeeze()

    for c in [cls_full, cls_half]:
        x = np.mean(c, axis=0)
        s = np.std(c, axis=0)/np.sqrt(nsims)
        assert np.all(np.fabs(x) < 5*s)
