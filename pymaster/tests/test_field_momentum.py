import numpy as np
import healpy as hp
import pymaster as nmt


def _gen_random(nsrc, mask):
    # Generates random objects following a given mask
    pmap = mask / np.amax(mask)
    nsrc_hi = int(nsrc/np.mean(pmap))
    nside = hp.npix2nside(len(pmap))
    cth = -1 + 2*np.random.rand(int(nsrc_hi))
    phi = 2*np.pi*np.random.rand(int(nsrc_hi))
    sth = np.sqrt(1 - cth**2)
    unif = np.random.rand(nsrc_hi)
    vec = np.array([sth*np.cos(phi), sth*np.sin(phi), cth])
    ipix = hp.vec2pix(nside, *vec)
    good = unif <= pmap[ipix]
    positions = np.array([np.arccos(cth[good]), phi[good]])
    return positions, ipix[good]


def test_field_momentum_init_CAR():
    from astropy.io import fits
    from astropy.wcs import WCS

    hdul = fits.open("test/benchmarks/msk_car.fits")
    wcs = WCS(hdul[0].header)
    hdul.close()
    mask = fits.open("test/benchmarks/msk_car.fits")[0].data

    nsrc = 1E5
    pos, ipix = _gen_random(nsrc, np.ones(12*64**2))
    w = np.ones(len(ipix))

    # Test that we can initialise just fine
    nmt.NmtFieldCatalogMomentum(pos, w, w,
                                None, None, 3*64-1,
                                mask=mask, wcs=wcs)


def test_field_momentum_Nw_Nf():
    nside = 128
    lmax = 3*nside-1
    ls = np.arange(lmax+1)

    # Input velocity power spectrum
    cl = 1/(ls+10)**4

    # Read mask
    mask = hp.ud_grade(hp.read_map('test/sel_quaia.fits'),
                       nside_out=nside)

    # Catalog field with mask
    nsrc = 1E5
    pos, ipix = _gen_random(nsrc, mask)
    w = np.ones(len(ipix))
    alm = hp.synalm(cl, lmax=lmax)
    apos = nmt.utils._alm2catalog_ducc0(np.array([alm]), pos, 0, lmax)[0]
    fc = nmt.NmtFieldCatalogMomentum(pos, w, apos,
                                     None, None, lmax,
                                     mask=mask)
    ndens = len(w)/(4*np.pi*np.mean(mask))
    assert fc._Nw == 0
    Nf_pred = np.sum((apos/ndens)**2)/(4*np.pi)
    assert np.fabs(fc._Nf/Nf_pred-1) < 1E-6

    # Catalog field with randoms
    nrand = 1E6
    posr, ipixr = _gen_random(nrand, mask)
    wr = np.ones(len(ipixr))
    fc = nmt.NmtFieldCatalogMomentum(pos, w, apos,
                                     posr, wr, lmax)
    ndens_r = len(wr)/(4*np.pi)
    assert np.fabs(fc._Nw/ndens_r-1) < 1E-6


def test_field_momentum_unbiased():
    nside = 128
    lmax = 3*nside-1
    ls = np.arange(lmax+1)

    # Input velocity power spectrum
    cl = 1/(ls+10)**4

    # Read mask
    mask = hp.ud_grade(hp.read_map('test/sel_quaia.fits'),
                       nside_out=nside)

    # Binning
    b = nmt.NmtBin.from_nside_linear(nside, nlb=4)

    # Map-based field
    fm = nmt.NmtField(mask, None, spin=0)
    wm = nmt.NmtWorkspace.from_fields(fm, fm, b)

    # Catalog field with mask
    nsrc = 1E5
    pos, ipix = _gen_random(nsrc, mask)
    w = np.ones(len(ipix))
    fc = nmt.NmtFieldCatalogMomentum(pos, w, w,
                                     None, None, lmax,
                                     mask=mask)
    wx = nmt.NmtWorkspace.from_fields(fm, fc, b)

    # Catalog field with randoms
    nrand = 1E6
    posr, ipixr = _gen_random(nrand, mask)
    wr = np.ones(len(ipixr))
    fc = nmt.NmtFieldCatalogMomentum(pos, w, w,
                                     posr, wr, lmax)
    wxr = nmt.NmtWorkspace.from_fields(fm, fc, b)

    # Run 100 sims
    nsims = 100
    cls_x = []
    cls_xr = []
    for i in range(nsims):
        if i % 10 == 0:
            print(i)
        # Generate alm
        alm = hp.synalm(cl, lmax=lmax)

        # Map field
        amap = hp.alm2map(alm, nside)
        fm = nmt.NmtField(mask, [amap])

        pos, ipix = _gen_random(nsrc, mask)
        w = np.ones(len(ipix))
        apos = nmt.utils._alm2catalog_ducc0(np.array([alm]), pos, 0, lmax)[0]

        # Catalog field with mask
        fc = nmt.NmtFieldCatalogMomentum(pos, w, apos, None, None,
                                         lmax, mask=mask)
        clx = wx.decouple_cell(nmt.compute_coupled_cell(fm, fc)).squeeze()
        cls_x.append(clx)

        # Catalog field with randoms
        fc = nmt.NmtFieldCatalogMomentum(pos, w, apos, posr, wr, lmax)
        clx = wxr.decouple_cell(nmt.compute_coupled_cell(fm, fc)).squeeze()
        cls_xr.append(clx)
    cls_x = np.array(cls_x)
    cls_xr = np.array(cls_xr)

    cl_th = wm.decouple_cell(wm.couple_cell([cl]))[0]
    cl_thr = wxr.decouple_cell(wxr.couple_cell([cl]))[0]

    err_x = np.std(cls_x, axis=0)/np.sqrt(nsims)
    err_xr = np.std(cls_xr, axis=0)/np.sqrt(nsims)
    bias_mask = (np.mean(cls_x, axis=0) - cl_th) / err_x
    bias_rand = (np.mean(cls_xr, axis=0) - cl_thr) / err_xr

    # No deviation by more than 7 x error on mean
    assert np.all(np.fabs(bias_mask) < 7)
    assert np.all(np.fabs(bias_rand) < 7)


def test_field_momentum_errors():
    import pytest

    nside = 128
    lmax = 3*nside-1
    mask = hp.ud_grade(hp.read_map('test/sel_quaia.fits'),
                       nside_out=nside)
    nsrc = 1E5
    pos, ipix = _gen_random(nsrc, mask)
    w = np.ones(len(ipix))
    f = np.ones(len(ipix))

    # This is fine
    nmt.NmtFieldCatalogMomentum(pos, w, f,
                                None, None, lmax,
                                mask=mask)

    with pytest.raises(ValueError):  # Field too short
        nmt.NmtFieldCatalogMomentum(pos, w, f[1:],
                                    None, None, lmax, mask=mask)

    with pytest.raises(ValueError):  # Spin field not implemented
        nmt.NmtFieldCatalogMomentum(pos, w, np.array([f, f]),
                                    None, None, lmax, mask=mask)

    with pytest.raises(ValueError):  # Weights too short
        nmt.NmtFieldCatalogMomentum(pos, w[1:], f,
                                    None, None, lmax, mask=mask)
