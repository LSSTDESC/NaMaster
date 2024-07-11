import pytest
import numpy as np
import pymaster as nmt


class WorkspaceTesterCAR(object):
    def __init__(self):
        from astropy.io import fits
        from astropy.wcs import WCS

        # Read mask
        hdul = fits.open("test/benchmarks/msk_car.fits")
        self.msk = hdul[0].data
        # Set up coordinates
        self.wcs = WCS(hdul[0].header)
        self.nx, self.ny = self.wcs.pixel_shape
        hdul.close()
        # Read maps
        hdul = fits.open("test/benchmarks/mps_car.fits")
        self.mps = hdul[0].data
        hdul.close()

        self.minfo = nmt.NmtMapInfo(self.wcs, (self.ny, self.nx))
        self.lmax = self.minfo.get_lmax()
        self.nlb = 50
        self.npix = self.minfo.npix
        self.b = nmt.NmtBin.from_lmax_linear(self.lmax, self.nlb)
        (self.l, self.cltt, self.clte,
         self.clee, self.clbb, self.cltb,
         self.cleb) = np.loadtxt("test/benchmarks/pspy_cls.txt", unpack=True)
        self.f0 = nmt.NmtField(self.msk, [self.mps[0]], wcs=self.wcs, n_iter=0)
        self.f0_half = nmt.NmtField(self.msk[:self.ny//2, :self.nx//2],
                                    [self.mps[0][:self.ny//2, :self.nx//2]],
                                    wcs=self.wcs, n_iter=0)
        self.b_half = nmt.NmtBin.from_lmax_linear(self.lmax//2, self.nlb)
        self.b_doub = nmt.NmtBin.from_lmax_linear(self.lmax*2, self.nlb)
        self.n_good = np.zeros([1, (self.lmax+1)])
        self.n_bad = np.zeros([2, (self.lmax+1)])
        self.n_half = np.zeros([1, (self.lmax//2+1)])


WT = WorkspaceTesterCAR()


@pytest.mark.skipif(True, reason='slow')
def test_workspace_car_master():
    f0 = nmt.NmtField(WT.msk, [WT.mps[0]],
                      wcs=WT.wcs, n_iter=0)
    f2 = nmt.NmtField(WT.msk, [WT.mps[1], WT.mps[2]],
                      wcs=WT.wcs, n_iter=0)
    f = [f0, f2]

    for ip1 in range(2):
        for ip2 in range(ip1, 2):
            if ip1 == ip2 == 0:
                cl_bm = np.array([WT.cltt])
            elif ip1 == ip2 == 1:
                cl_bm = np.array([WT.clee, WT.cleb,
                                  WT.cleb, WT.clbb])
            else:
                cl_bm = np.array([WT.clte, WT.cltb])
            w = nmt.NmtWorkspace.from_fields(f[ip1], f[ip2], WT.b)
            cl = w.decouple_cell(nmt.compute_coupled_cell(f[ip1], f[ip2]))
            assert np.amax(np.fabs(cl-cl_bm)) <= 1E-10

    # TEB
    w = nmt.NmtWorkspace.from_fields(f[0], f[1], WT.b,
                                     is_teb=True)
    c00 = nmt.compute_coupled_cell(f[0], f[0])
    c02 = nmt.compute_coupled_cell(f[0], f[1])
    c22 = nmt.compute_coupled_cell(f[1], f[1])
    cl = w.decouple_cell(np.array([c00[0], c02[0], c02[1],
                                   c22[0], c22[1], c22[2],
                                   c22[3]]))
    cl_bm = np.array([WT.cltt, WT.clte, WT.cltb,
                      WT.clee, WT.cleb, WT.cleb,
                      WT.clbb])
    assert np.amax(np.fabs(cl-cl_bm)) <= 1E-10


@pytest.mark.skipif(False, reason='slow')
def test_workspace_car_methods():
    w = nmt.NmtWorkspace.from_fields(WT.f0, WT.f0,
                                     WT.b)  # OK init
    # Incompatible bandpowers
    with pytest.raises(ValueError):
        w.compute_coupling_matrix(WT.f0, WT.f0,
                                  WT.b_doub)
    # Incompatible resolutions
    assert not WT.f0.is_compatible(WT.f0_half)

    # Wrong fields for TEB
    with pytest.raises(RuntimeError):
        w.compute_coupling_matrix(WT.f0, WT.f0,
                                  WT.b, is_teb=True)

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
