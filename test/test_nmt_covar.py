import unittest
import numpy as np
import pymaster as nmt
import healpy as hp
import warnings
import sys
from .testutils import read_flat_map

# Unit tests associated with the NmtField and NmtFieldFlat classes


class TestCovarFsk(unittest.TestCase):
    def setUp(self):
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

    def test_workspace_covar_flat_benchmark(self):
        def compare_covars(c, cb):
            # Check first and second diagonals
            for k in [0, 1]:
                d = np.diag(c, k=k)
                db = np.diag(cb, k=k)
                self.assertTrue((np.fabs(d-db) <=
                                 np.fmin(np.fabs(d),
                                         np.fabs(db))*1E-4).all())

        cw = nmt.NmtCovarianceWorkspaceFlat()
        cw.compute_coupling_coefficients(self.f0, self.f0, self.b)

        # [0,0 ; 0,0]
        covar = nmt.gaussian_covariance_flat(cw, 0, 0, 0, 0, self.ll,
                                             [self.cltt], [self.cltt],
                                             [self.cltt], [self.cltt],
                                             self.w)
        covar_bench = np.loadtxt("test/benchmarks/bm_f_nc_np_cov.txt",
                                 unpack=True)
        compare_covars(covar, covar_bench)
        # [0,2 ; 0,2]
        covar = nmt.gaussian_covariance_flat(cw, 0, 2, 0, 2, self.ll,
                                             [self.cltt],
                                             [self.clte, 0*self.clte],
                                             [self.clte, 0*self.clte],
                                             [self.clee, 0*self.clee,
                                              0*self.clee, self.clbb],
                                             self.w02, wb=self.w02)
        covar_bench = np.loadtxt("test/benchmarks/bm_f_nc_np_cov0202.txt")
        compare_covars(covar, covar_bench)
        # [0,0 ; 0,2]
        covar = nmt.gaussian_covariance_flat(cw, 0, 0, 0, 2, self.ll,
                                             [self.cltt],
                                             [self.clte, 0*self.clte],
                                             [self.cltt],
                                             [self.clte, 0*self.clte],
                                             self.w, wb=self.w02)
        covar_bench = np.loadtxt("test/benchmarks/bm_f_nc_np_cov0002.txt")
        compare_covars(covar, covar_bench)
        # [0,0 ; 2,2]
        covar = nmt.gaussian_covariance_flat(cw, 0, 0, 2, 2, self.ll,
                                             [self.clte, 0*self.clte],
                                             [self.clte, 0*self.clte],
                                             [self.clte, 0*self.clte],
                                             [self.clte, 0*self.clte],
                                             self.w, wb=self.w22)
        covar_bench = np.loadtxt("test/benchmarks/bm_f_nc_np_cov0022.txt")
        compare_covars(covar, covar_bench)
        # [2,2 ; 2,2]
        covar = nmt.gaussian_covariance_flat(cw, 2, 2, 2, 2, self.ll,
                                             [self.clee, 0*self.clee,
                                              0*self.clee, self.clbb],
                                             [self.clee, 0*self.clee,
                                              0*self.clee, self.clbb],
                                             [self.clee, 0*self.clee,
                                              0*self.clee, self.clbb],
                                             [self.clee, 0*self.clee,
                                              0*self.clee, self.clbb],
                                             self.w22, wb=self.w22)
        covar_bench = np.loadtxt("test/benchmarks/bm_f_nc_np_cov2222.txt")
        compare_covars(covar, covar_bench)

    def test_workspace_covar_flat_errors(self):
        cw = nmt.NmtCovarianceWorkspaceFlat()

        with self.assertRaises(ValueError):  # Write uninitialized
            cw.write_to("wsp.fits")

        cw.compute_coupling_coefficients(self.f0, self.f0, self.b)  # All good
        self.assertEqual(cw.wsp.bin.n_bands, self.w.wsp.bin.n_bands)

        with self.assertRaises(RuntimeError):  # Write uninitialized
            cw.write_to("tests/wsp.fits")

        cw.read_from('test/benchmarks/bm_f_nc_np_cw00.fits')  # Correct reading
        self.assertEqual(cw.wsp.bin.n_bands, self.w.wsp.bin.n_bands)

        # gaussian_covariance
        with self.assertRaises(ValueError):  # Wrong input power spectra
            nmt.gaussian_covariance_flat(cw, 0, 0, 0, 0, self.ll,
                                         [self.cltt], [self.cltt],
                                         [self.cltt], [self.cltt[:15]],
                                         self.w)
        with self.assertRaises(ValueError):  # Wrong input power shapes
            nmt.gaussian_covariance_flat(cw, 0, 0, 0, 0, self.ll,
                                         [self.cltt, self.cltt],
                                         [self.cltt], [self.cltt],
                                         [self.cltt[:15]], self.w)
        with self.assertRaises(ValueError):  # Wrong input spins
            nmt.gaussian_covariance_flat(cw, 0, 2, 0, 0, self.ll,
                                         [self.cltt], [self.cltt],
                                         [self.cltt], [self.cltt],
                                         self.w)

        with self.assertRaises(RuntimeError):  # Incorrect reading
            cw.read_from('none')
        with self.assertRaises(ValueError):  # Incompatible resolutions
            cw.compute_coupling_coefficients(self.f0, self.f0_half, self.b)
        with self.assertRaises(RuntimeError):  # Incompatible bandpowers
            cw.compute_coupling_coefficients(self.f0, self.f0, self.b,
                                             self.f0, self.f0, self.b_half)


class TestCovarSph(unittest.TestCase):
    def setUp(self):
        # This is to avoid showing an ugly warning
        # that has nothing to do with pymaster
        if (sys.version_info > (3, 1)):
            warnings.simplefilter("ignore", ResourceWarning)

        self.nside = 64
        self.nlb = 16
        self.npix = hp.nside2npix(self.nside)
        msk = hp.read_map("test/benchmarks/msk.fits", verbose=False)
        mps = np.array(hp.read_map("test/benchmarks/mps.fits",
                                   verbose=False, field=[0, 1, 2]))
        self.b = nmt.NmtBin.from_nside_linear(self.nside, self.nlb)
        self.f0 = nmt.NmtField(msk, [mps[0]])
        self.f2 = nmt.NmtField(msk, [mps[1], mps[2]])
        self.f0_half = nmt.NmtField(msk[:self.npix//4],
                                    [mps[0, :self.npix//4]])  # Half nside
        self.w = nmt.NmtWorkspace()
        self.w.read_from("test/benchmarks/bm_nc_np_w00.fits")
        self.w02 = nmt.NmtWorkspace()
        self.w02.read_from("test/benchmarks/bm_nc_np_w02.fits")
        self.w22 = nmt.NmtWorkspace()
        self.w22.read_from("test/benchmarks/bm_nc_np_w22.fits")

        cls = np.loadtxt("test/benchmarks/cls_lss.txt", unpack=True)
        l, cltt, clee, clbb, clte, nltt, nlee, nlbb, nlte = cls
        self.ll = l[:3*self.nside]
        self.cltt = cltt[:3*self.nside]+nltt[:3*self.nside]
        self.clee = clee[:3*self.nside]+nlee[:3*self.nside]
        self.clbb = clbb[:3*self.nside]+nlbb[:3*self.nside]
        self.clte = clte[:3*self.nside]

    def test_workspace_covar_benchmark(self):
        def compare_covars(c, cb):
            # Check first and second diagonals
            for k in [0, 1]:
                d = np.diag(c, k=k)
                db = np.diag(cb, k=k)
                self.assertTrue((np.fabs(d-db) <=
                                 np.fmin(np.fabs(d),
                                         np.fabs(db))*1E-4).all())

        # Check against a benchmark
        cw = nmt.NmtCovarianceWorkspace()
        cw.compute_coupling_coefficients(self.f0, self.f0)

        # [0,0 ; 0,0]
        covar = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                        [self.cltt], [self.cltt],
                                        [self.cltt], [self.cltt],
                                        self.w)
        covar_bench = np.loadtxt("test/benchmarks/bm_nc_np_cov.txt",
                                 unpack=True)
        compare_covars(covar, covar_bench)
        # Check coupled
        covar_rc = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                           [self.cltt], [self.cltt],
                                           [self.cltt], [self.cltt],
                                           self.w, coupled=True)  # [nl, nl]
        covar_c = np.array([self.w.decouple_cell([row])[0]
                            for row in covar_rc])  # [nl, nbpw]
        covar = np.array([self.w.decouple_cell([col])[0]
                          for col in covar_c.T]).T  # [nbpw, nbpw]
        compare_covars(covar, covar_bench)

        # [0,2 ; 0,2]
        covar = nmt.gaussian_covariance(cw, 0, 2, 0, 2,
                                        [self.cltt],
                                        [self.clte, 0*self.clte],
                                        [self.clte, 0*self.clte],
                                        [self.clee, 0*self.clee,
                                         0*self.clee, self.clbb],
                                        self.w02, wb=self.w02)
        covar_bench = np.loadtxt("test/benchmarks/bm_nc_np_cov0202.txt")
        compare_covars(covar, covar_bench)
        # [0,0 ; 0,2]
        covar = nmt.gaussian_covariance(cw, 0, 0, 0, 2,
                                        [self.cltt],
                                        [self.clte, 0*self.clte],
                                        [self.cltt],
                                        [self.clte, 0*self.clte],
                                        self.w, wb=self.w02)
        covar_bench = np.loadtxt("test/benchmarks/bm_nc_np_cov0002.txt")
        compare_covars(covar, covar_bench)
        # [0,0 ; 2,2]
        covar = nmt.gaussian_covariance(cw, 0, 0, 2, 2,
                                        [self.clte, 0*self.clte],
                                        [self.clte, 0*self.clte],
                                        [self.clte, 0*self.clte],
                                        [self.clte, 0*self.clte],
                                        self.w, wb=self.w22)
        covar_bench = np.loadtxt("test/benchmarks/bm_nc_np_cov0022.txt")
        compare_covars(covar, covar_bench)
        # [2,2 ; 2,2]
        covar = nmt.gaussian_covariance(cw, 2, 2, 2, 2,
                                        [self.clee, 0*self.clee,
                                         0*self.clee, self.clbb],
                                        [self.clee, 0*self.clee,
                                         0*self.clee, self.clbb],
                                        [self.clee, 0*self.clee,
                                         0*self.clee, self.clbb],
                                        [self.clee, 0*self.clee,
                                         0*self.clee, self.clbb],
                                        self.w22, wb=self.w22)
        covar_bench = np.loadtxt("test/benchmarks/bm_nc_np_cov2222.txt")
        compare_covars(covar, covar_bench)

    def test_workspace_covar_errors(self):
        cw = nmt.NmtCovarianceWorkspace()

        with self.assertRaises(ValueError):  # Write uninitialized
            cw.write_to("wsp.fits")

        cw.compute_coupling_coefficients(self.f0, self.f0)  # All good
        self.assertEqual(cw.wsp.lmax, self.w.wsp.lmax)
        self.assertEqual(cw.wsp.lmax, self.w.wsp.lmax)
        with self.assertRaises(RuntimeError):  # Write uninitialized
            cw.write_to("tests/wsp.fits")

        cw.read_from('test/benchmarks/bm_nc_np_cw00.fits')  # Correct reading
        self.assertEqual(cw.wsp.lmax, self.w.wsp.lmax)
        self.assertEqual(cw.wsp.lmax, self.w.wsp.lmax)

        # gaussian_covariance
        with self.assertRaises(ValueError):  # Wrong input cl size
            nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                    [self.cltt], [self.cltt],
                                    [self.cltt], [self.cltt[:15]],
                                    self.w)
        with self.assertRaises(ValueError):  # Wrong input cl shapes
            nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                    [self.cltt], [self.cltt],
                                    [self.cltt], [self.cltt, self.cltt],
                                    self.w)
        with self.assertRaises(ValueError):  # Wrong input spins
            nmt.gaussian_covariance(cw, 0, 2, 0, 0,
                                    [self.cltt], [self.cltt],
                                    [self.cltt], [self.cltt, self.cltt],
                                    self.w)

        with self.assertRaises(RuntimeError):  # Incorrect reading
            cw.read_from('none')

        with self.assertRaises(ValueError):  # Incompatible resolutions
            cw.compute_coupling_coefficients(self.f0, self.f0_half)


if __name__ == '__main__':
    unittest.main()
