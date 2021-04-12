import pytest
import numpy as np
import pymaster as nmt
from .utils import read_flat_map


class CovarTester(object):
    def __init__(self):
        wcs, msk = read_flat_map("test/benchmarks/msk_flat.fits")
        (ny, nx) = msk.shape
        lx = np.radians(np.fabs(nx*wcs.wcs.cdelt[0]))
        ly = np.radians(np.fabs(ny*wcs.wcs.cdelt[1]))
        mps = np.array([read_flat_map("test/benchmarks/mps_flat.fits",
                                      i_map=i)[1] for i in range(3)])

        d_ell = 20
        lmax = 500.
        ledges = np.arange(int(lmax/d_ell)+1)*d_ell+2
        self.b = nmt.NmtBinFlat(ledges[:-1], ledges[1:])
        ledges_half = ledges[:len(ledges)//2]
        self.b_half = nmt.NmtBinFlat(ledges_half[:-1], ledges_half[1:])
        self.f0 = nmt.NmtFieldFlat(lx, ly, msk, [mps[0]])
        self.f2 = nmt.NmtFieldFlat(lx, ly, msk, [mps[1], mps[2]])
        self.f0_half = nmt.NmtFieldFlat(lx, ly, msk[:ny//2, :nx//2],
                                        [mps[0, :ny//2, :nx//2]])
        self.w = nmt.NmtWorkspaceFlat()
        self.w.read_from("test/benchmarks/bm_f_nc_np_w00.fits")
        self.w02 = nmt.NmtWorkspaceFlat()
        self.w02.read_from("test/benchmarks/bm_f_nc_np_w02.fits")
        self.w22 = nmt.NmtWorkspaceFlat()
        self.w22.read_from("test/benchmarks/bm_f_nc_np_w22.fits")

        cls = np.loadtxt("test/benchmarks/cls_lss.txt", unpack=True)
        l, cltt, clee, clbb, clte, nltt, nlee, nlbb, nlte = cls
        self.ll = l
        self.cltt = cltt+nltt
        self.clee = clee+nlee
        self.clbb = clbb+nlbb
        self.clte = clte


CT = CovarTester()


def test_workspace_covar_flat_benchmark():
    def compare_covars(c, cb):
        # Check first and second diagonals
        for k in [0, 1]:
            d = np.diag(c, k=k)
            db = np.diag(cb, k=k)
            assert (np.fabs(d-db) <=
                    np.fmin(np.fabs(d),
                            np.fabs(db))*1E-4).all()

    cw = nmt.NmtCovarianceWorkspaceFlat()
    cw.compute_coupling_coefficients(CT.f0, CT.f0, CT.b)

    # [0,0 ; 0,0]
    covar = nmt.gaussian_covariance_flat(cw, 0, 0, 0, 0, CT.ll,
                                         [CT.cltt], [CT.cltt],
                                         [CT.cltt], [CT.cltt],
                                         CT.w)
    covar_bench = np.loadtxt("test/benchmarks/bm_f_nc_np_cov.txt",
                             unpack=True)
    compare_covars(covar, covar_bench)
    # [0,2 ; 0,2]
    covar = nmt.gaussian_covariance_flat(cw, 0, 2, 0, 2, CT.ll,
                                         [CT.cltt],
                                         [CT.clte, 0*CT.clte],
                                         [CT.clte, 0*CT.clte],
                                         [CT.clee, 0*CT.clee,
                                          0*CT.clee, CT.clbb],
                                         CT.w02, wb=CT.w02)
    covar_bench = np.loadtxt("test/benchmarks/bm_f_nc_np_cov0202.txt")
    compare_covars(covar, covar_bench)
    # [0,0 ; 0,2]
    covar = nmt.gaussian_covariance_flat(cw, 0, 0, 0, 2, CT.ll,
                                         [CT.cltt],
                                         [CT.clte, 0*CT.clte],
                                         [CT.cltt],
                                         [CT.clte, 0*CT.clte],
                                         CT.w, wb=CT.w02)
    covar_bench = np.loadtxt("test/benchmarks/bm_f_nc_np_cov0002.txt")
    compare_covars(covar, covar_bench)
    # [0,0 ; 2,2]
    covar = nmt.gaussian_covariance_flat(cw, 0, 0, 2, 2, CT.ll,
                                         [CT.clte, 0*CT.clte],
                                         [CT.clte, 0*CT.clte],
                                         [CT.clte, 0*CT.clte],
                                         [CT.clte, 0*CT.clte],
                                         CT.w, wb=CT.w22)
    covar_bench = np.loadtxt("test/benchmarks/bm_f_nc_np_cov0022.txt")
    compare_covars(covar, covar_bench)
    # [2,2 ; 2,2]
    covar = nmt.gaussian_covariance_flat(cw, 2, 2, 2, 2, CT.ll,
                                         [CT.clee, 0*CT.clee,
                                          0*CT.clee, CT.clbb],
                                         [CT.clee, 0*CT.clee,
                                          0*CT.clee, CT.clbb],
                                         [CT.clee, 0*CT.clee,
                                          0*CT.clee, CT.clbb],
                                         [CT.clee, 0*CT.clee,
                                          0*CT.clee, CT.clbb],
                                         CT.w22, wb=CT.w22)
    covar_bench = np.loadtxt("test/benchmarks/bm_f_nc_np_cov2222.txt")
    compare_covars(covar, covar_bench)


def test_workspace_covar_flat_errors():
    cw = nmt.NmtCovarianceWorkspaceFlat()

    with pytest.raises(ValueError):  # Write uninitialized
        cw.write_to("wsp.fits")

    cw.compute_coupling_coefficients(CT.f0, CT.f0, CT.b)  # All good
    assert cw.wsp.bin.n_bands == CT.w.wsp.bin.n_bands

    with pytest.raises(RuntimeError):  # Write uninitialized
        cw.write_to("tests/wsp.fits")

    cw.read_from('test/benchmarks/bm_f_nc_np_cw00.fits')  # Correct reading
    assert cw.wsp.bin.n_bands == CT.w.wsp.bin.n_bands

    # gaussian_covariance
    with pytest.raises(ValueError):  # Wrong input power spectra
        nmt.gaussian_covariance_flat(cw, 0, 0, 0, 0, CT.ll,
                                     [CT.cltt], [CT.cltt],
                                     [CT.cltt], [CT.cltt[:15]],
                                     CT.w)
    with pytest.raises(ValueError):  # Wrong input power shapes
        nmt.gaussian_covariance_flat(cw, 0, 0, 0, 0, CT.ll,
                                     [CT.cltt, CT.cltt],
                                     [CT.cltt], [CT.cltt],
                                     [CT.cltt[:15]], CT.w)
    with pytest.raises(ValueError):  # Wrong input spins
        nmt.gaussian_covariance_flat(cw, 0, 2, 0, 0, CT.ll,
                                     [CT.cltt], [CT.cltt],
                                     [CT.cltt], [CT.cltt],
                                     CT.w)

    with pytest.raises(RuntimeError):  # Incorrect reading
        cw.read_from('none')
    with pytest.raises(ValueError):  # Incompatible resolutions
        cw.compute_coupling_coefficients(CT.f0, CT.f0_half, CT.b)
    with pytest.raises(RuntimeError):  # Incompatible bandpowers
        cw.compute_coupling_coefficients(CT.f0, CT.f0, CT.b,
                                         CT.f0, CT.f0, CT.b_half)
