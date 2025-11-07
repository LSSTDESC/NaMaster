import pytest
import numpy as np
import pymaster as nmt


class CARTester(object):
    def __init__(self):
        from astropy.io import fits
        from astropy.wcs import WCS

        hdul = fits.open("test/benchmarks/msk_car.fits")
        self.msk = hdul[0].data
        self.wcs = WCS(hdul[0].header)
        hdul.close()

        hdul = fits.open("test/benchmarks/mps_car.fits")
        self.mp = hdul[0].data
        hdul.close()

        self.leff, self.cl_bm = np.loadtxt(
            "test/benchmarks/bm_car.txt", unpack=True)

        self.lmax = 100
        self.f0 = nmt.NmtField(self.msk, [self.mp],
                               wcs=self.wcs, lmax=self.lmax)
        self.b = nmt.NmtBin.from_lmax_linear(self.lmax, nlb=5)


WT = CARTester()


def test_cl_car():
    w = nmt.NmtWorkspace.from_fields(WT.f0, WT.f0, WT.b)
    cl = w.decouple_cell(nmt.compute_coupled_cell(WT.f0,
                                                  WT.f0))[0]

    assert np.amax(np.fabs(cl-WT.cl_bm)) <= 1E-10


def test_workspace_car_methods():
    w = nmt.NmtWorkspace.from_fields(WT.f0, WT.f0,
                                     WT.b)  # OK init
    # Incompatible bandpowers
    with pytest.raises(ValueError):
        bd = nmt.NmtBin.from_lmax_linear(2*WT.lmax, nlb=5)
        w.compute_coupling_matrix(WT.f0, WT.f0, bd)

    # Incompatible resolutions
    nx, ny = WT.wcs.pixel_shape
    fhalf = nmt.NmtField(WT.msk[:ny//2, :nx//2],
                         [WT.mp[:ny//2, :nx//2]],
                         wcs=WT.wcs, lmax=WT.lmax)
    assert not WT.f0.is_compatible(fhalf)

    # Wrong fields for TEB
    with pytest.raises(RuntimeError):
        w.compute_coupling_matrix(WT.f0, WT.f0,
                                  WT.b, is_teb=True)

    w.compute_coupling_matrix(WT.f0, WT.f0, WT.b)

    # Test couple_cell
    n_good = np.ones([1, WT.lmax+1])
    n_bad = np.ones([2, WT.lmax+1])
    n_half = np.ones([1, WT.lmax//2+1])
    c = w.couple_cell(n_good)
    assert c.shape == (1, w.wsp.lmax+1)
    with pytest.raises(ValueError):
        w.couple_cell(n_bad)
    with pytest.raises(ValueError):
        w.couple_cell(n_half)

    # Test decouple_cell
    c = w.decouple_cell(n_good)
    assert c.shape == (1, WT.b.bin.n_bands)
    with pytest.raises(ValueError):
        w.couple_cell(n_bad)
    with pytest.raises(ValueError):
        w.couple_cell(n_half)
    c = w.decouple_cell(n_good, cl_bias=n_good,
                        cl_noise=n_good)
    assert c.shape == (1, WT.b.bin.n_bands)
    with pytest.raises(ValueError):
        w.decouple_cell(n_good, cl_bias=n_good,
                        cl_noise=n_bad)
    with pytest.raises(ValueError):
        w.decouple_cell(n_good, cl_bias=n_good,
                        cl_noise=n_half)
    with pytest.raises(ValueError):
        w.decouple_cell(n_good, cl_bias=n_bad,
                        cl_noise=n_good)
    with pytest.raises(ValueError):
        w.decouple_cell(n_good, cl_bias=n_half,
                        cl_noise=n_good)
    return
