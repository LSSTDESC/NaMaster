import numpy as np
import pymaster as nmt
import warnings
import sys
import pytest
from .utils import normdiff
from astropy.io import fits
from astropy.wcs import WCS


class FieldTesterCAR(object):
    def __init__(self):
        # This is to avoid showing an ugly warning that
        # has nothing to do with pymaster
        if (sys.version_info > (3, 1)):
            warnings.simplefilter("ignore", ResourceWarning)

        hdul = fits.open("test/benchmarks/mps_car_small.fits")
        self.wcs = WCS(hdul[0].header)
        self.ny, self.nx = hdul[0].data.shape
        hdul.close()

        self.minfo = nmt.NmtMapInfo(self.wcs, (self.ny, self.nx))
        self.lmax = self.minfo.get_lmax()
        self.ntemp = 5
        self.npix = self.ny*self.nx
        self.msk = np.ones([self.ny, self.nx])
        self.mps = np.ones([3, self.ny, self.nx])
        self.tmp = np.ones([self.ntemp, 3, self.ny, self.nx])
        self.beam = np.ones(self.lmax+1)
        ix = np.arange(self.nx)
        iy = np.arange(self.ny)
        shp1 = [2, self.ny*self.nx]
        shp2 = [2, self.ny, self.nx]
        world = np.array(np.meshgrid(ix, iy)).reshape(shp1).T
        ph, th = self.wcs.wcs_pix2world(world, 0).T.reshape(shp2)
        ph = np.radians(ph)
        th = np.radians(90-th)
        sth = np.sin(th)
        cth = np.cos(th)
        # Re(Y_22)
        self.mps[0, :, :] = np.sqrt(15./2./np.pi)*sth**2*np.cos(2*ph)
        # _2Y^E_20 + _2Y^B_30
        self.mps[1, :, :] = -np.sqrt(15./2./np.pi)*sth**2/4.
        self.mps[2, :, :] = -np.sqrt(105./2./np.pi)*cth*sth**2/2.
        for i in range(self.ntemp):
            # Re(Y_22)
            self.tmp[i][0, :, :] = np.sqrt(15./2./np.pi)*sth**2*np.cos(2*ph)
            # _2Y^E_20 + _2Y^B_30
            self.tmp[i][1, :, :] = -np.sqrt(15./2./np.pi)*sth**2/4.
            self.tmp[i][2, :, :] = -np.sqrt(105./2./np.pi)*cth*sth**2/2.


FT = FieldTesterCAR()


def test_field_lite():
    # Lite field
    fl = nmt.NmtField(FT.msk, [FT.mps[0]], wcs=FT.wcs,
                      beam=FT.beam, lite=True)
    # Empty field
    with pytest.raises(ValueError):  # No maps and no spin
        fe = nmt.NmtField(FT.msk, None, wcs=FT.wcs,
                          beam=FT.beam)
    fe = nmt.NmtField(FT.msk, None, wcs=FT.wcs,
                      beam=FT.beam, spin=1)

    # Error checks
    for f in [fl, fe]:
        with pytest.raises(ValueError):  # Query maps
            f.get_maps()
        with pytest.raises(ValueError):  # Query templates
            f.get_templates()


def test_field_alloc():
    # No templates
    f0 = nmt.NmtField(FT.msk, [FT.mps[0]],
                      beam=FT.beam, wcs=FT.wcs)
    f2 = nmt.NmtField(FT.msk, [FT.mps[1], FT.mps[2]],
                      beam=FT.beam, wcs=FT.wcs)
    f2p = nmt.NmtField(FT.msk, [FT.mps[1], FT.mps[2]],
                       beam=FT.beam, wcs=FT.wcs,
                       purify_e=True, purify_b=True, n_iter_mask=10)
    assert (normdiff(f0.get_maps()[0],
                     (FT.mps[0]*FT.msk).flatten()) < 1E-10)
    assert (normdiff(f2.get_maps()[0],
                     (FT.mps[1]*FT.msk).flatten()) < 1E-10)
    assert (normdiff(f2.get_maps()[1],
                     (FT.mps[2]*FT.msk).flatten()) < 1E-10)
    assert (1E-5*np.mean(np.fabs(f2p.get_maps()[0])) >
            np.mean(np.fabs(f2p.get_maps()[0] -
                            (FT.mps[1]*FT.msk).flatten())))
    assert (1E-5*np.mean(np.fabs(f2p.get_maps()[1])) >
            np.mean(np.fabs(f2p.get_maps()[1] -
                            (FT.mps[2]*FT.msk).flatten())))
    for f in [f0, f2, f2p]:
        with pytest.raises(ValueError):
            f.get_templates()

    # With templates
    f0 = nmt.NmtField(FT.msk, [FT.mps[0]],
                      templates=np.array([[t[0]] for t in FT.tmp]),
                      beam=FT.beam, wcs=FT.wcs)
    f2 = nmt.NmtField(FT.msk, [FT.mps[1], FT.mps[2]],
                      templates=np.array([[t[1], t[2]] for t in FT.tmp]),
                      beam=FT.beam, wcs=FT.wcs)
    # Map should be zero, since template =  map
    assert (normdiff(f0.get_maps()[0],
                     0*FT.msk.flatten()) < 1E-10)
    assert (normdiff(f2.get_maps()[0],
                     0*FT.msk.flatten()) < 1E-10)
    assert (normdiff(f2.get_maps()[1],
                     0*FT.msk.flatten()) < 1E-10)
    assert (len(f0.get_templates()) == 5)
    assert (len(f2.get_templates()) == 5)


def test_alminfo_eq():
    ainfo = nmt.NmtAlmInfo(lmax=1000)

    # Identity is equivalence
    assert ainfo == ainfo

    # Equivalence without identity
    ainfob = nmt.NmtAlmInfo(lmax=1000)
    assert ainfo == ainfob

    # Wrong type
    assert ainfo != 3

    # Wrong lmax
    ainfob = nmt.NmtAlmInfo(lmax=1001)
    assert ainfo != ainfob


def test_mapinfo_eq():
    # MapInfo for CAR with wrong input
    with pytest.raises(ValueError):
        nmt.NmtMapInfo(FT.wcs, FT.mps[0].flatten().shape)

    mi_car = nmt.NmtMapInfo(FT.wcs, FT.mps[0].shape)

    # Identity is equivalence
    assert mi_car == mi_car

    # Equivalence without identity
    mi_car2 = nmt.NmtMapInfo(FT.wcs, FT.mps[0].shape)
    assert mi_car == mi_car2

    # Wrong type
    assert mi_car != 3

    # Healpix and CAR
    mi_hpx = nmt.NmtMapInfo(None, (12*64*64,))
    assert mi_car != mi_hpx
    assert mi_hpx != mi_car


def test_field_error():
    # SHTs using healpy for CAR maps
    f0 = nmt.NmtField(FT.msk, [FT.mps[0]], wcs=FT.wcs)
    mp = f0.get_maps()
    alm = f0.get_alms()
    nmt.set_sht_calculator('healpy')
    with pytest.raises(ValueError):
        nmt.utils.map2alm(mp, 0, f0.minfo, f0.ainfo, n_iter=0)
    with pytest.raises(ValueError):
        nmt.utils.alm2map(alm, 0, f0.minfo, f0.ainfo)
    nmt.set_sht_calculator('ducc')

    with pytest.raises(ValueError):  # Not passing WCS
        nmt.NmtField(FT.msk, [FT.mps[0]], beam=FT.beam)
    with pytest.raises(ValueError):  # Passing 1D maps
        nmt.NmtField(FT.msk.flatten(),
                     [FT.mps[0]], beam=FT.beam,
                     wcs=FT.wcs)
    with pytest.raises(AttributeError):  # Passing incorrect WCS
        nmt.NmtField(FT.msk, [FT.mps[0]], beam=FT.beam, wcs=1)
    # Incorrect sky projection
    wcs = FT.wcs.deepcopy()
    wcs.wcs.ctype[0] = 'RA---TAN'
    with pytest.raises(ValueError):
        nmt.NmtField(FT.msk, [FT.mps[0]], beam=FT.beam, wcs=wcs)
    # Incorrect reference pixel coords
    wcs = FT.wcs.deepcopy()
    wcs.wcs.crval[1] = 96.
    with pytest.raises(ValueError):
        nmt.NmtField(FT.msk, [FT.mps[0]], beam=FT.beam, wcs=wcs)
    # Incorrect pixel sizes
    wcs = FT.wcs.deepcopy()
    wcs.wcs.cdelt[1] = 1.01
    with pytest.raises(ValueError):
        nmt.NmtField(FT.msk, [FT.mps[0]], beam=FT.beam, wcs=wcs)
    # Input maps are too big
    msk = np.ones([FT.ny, FT.nx+10])
    with pytest.raises(ValueError):
        nmt.NmtField(msk, [FT.mps[0]], beam=FT.beam, wcs=FT.wcs)
    with pytest.raises(ValueError):
        msk = np.ones([FT.ny+100, FT.nx])
        nmt.NmtField(msk, [msk], beam=FT.beam, wcs=FT.wcs)
    # Reference pixel has wrong pixel coordinates
    wcs = FT.wcs.deepcopy()
    wcs.wcs.crpix[0] = 1.
    wcs.wcs.cdelt[0] = -1.
    with pytest.raises(ValueError):
        nmt.NmtField(FT.msk, [FT.mps[0]],
                     beam=FT.beam, wcs=wcs)
    with pytest.raises(ValueError):  # Incorrect mask size
        nmt.NmtField(FT.msk[:90], [FT.mps[0]],
                     beam=FT.beam, wcs=FT.wcs)
    with pytest.raises(ValueError):  # Incorrect maps size
        nmt.NmtField(FT.msk, [FT.mps[0, :90]],
                     beam=FT.beam, wcs=FT.wcs)
    with pytest.raises(ValueError):  # Too many maps
        nmt.NmtField(FT.msk, FT.mps, wcs=FT.wcs)
    with pytest.raises(ValueError):  # Too many maps per template
        nmt.NmtField(FT.msk, [FT.mps[0]],
                     templates=FT.tmp, beam=FT.beam, wcs=FT.wcs)
    with pytest.raises(ValueError):
        # Number of maps per template does not match spin
        nmt.NmtField(FT.msk, [FT.mps[0]],
                     templates=[[t[0], t[1]] for t in FT.tmp],
                     beam=FT.beam, wcs=FT.wcs)
    with pytest.raises(ValueError):  # Incorrect template size
        nmt.NmtField(FT.msk, [FT.mps[0]],
                     templates=[[t[0, :90]] for t in FT.tmp],
                     beam=FT.beam, wcs=FT.wcs)
    with pytest.raises(ValueError):  # Passing crap as templates
        nmt.NmtField(FT.msk, [FT.mps[0]],
                     templates=1, beam=FT.beam, wcs=FT.wcs)
    with pytest.raises(ValueError):  # Passing wrong beam
        nmt.NmtField(FT.msk, [FT.mps[0]],
                     templates=[[t[0]] for t in FT.tmp],
                     beam=FT.beam[:90], wcs=FT.wcs)
    with pytest.raises(ValueError):  # Passing crap as beam
        nmt.NmtField(FT.msk, [FT.mps[0]],
                     templates=[[t[0]] for t in FT.tmp],
                     beam=1, wcs=FT.wcs)


def test_field_car_wrap():
    # Tests that full-sky maps that could wrap around azimuth
    # with floating point errors in `cdelt` are still parseable.

    hdul = fits.open("test/benchmarks/car_wrap.fits")
    wcs = WCS(hdul[0].header)
    m = hdul[0].data
    # This should throw no error
    nmt.NmtField(m[0], [m[0]], wcs=wcs)
