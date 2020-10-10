import unittest
import numpy as np
import pymaster as nmt
import healpy as hp
import warnings
import sys
from .testutils import normdiff, read_flat_map

# Unit tests associated with the NmtField and NmtFieldFlat classes


class TestFieldCAR(unittest.TestCase):
    def setUp(self):
        # This is to avoid showing an ugly warning that
        # has nothing to do with pymaster
        if (sys.version_info > (3, 1)):
            warnings.simplefilter("ignore", ResourceWarning)

        from astropy.io import fits
        from astropy.wcs import WCS

        hdul = fits.open("test/benchmarks/mps_car_small.fits")
        self.wcs = WCS(hdul[0].header)
        self.ny, self.nx = hdul[0].data.shape
        hdul.close()

        self.wt = nmt.NmtWCSTranslator(self.wcs, (self.ny, self.nx))
        self.lmax = self.wt.get_lmax()
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

    def test_field_lite(self):
        # Lite field
        fl = nmt.NmtField(self.msk, [self.mps[0]], wcs=self.wcs,
                          beam=self.beam, lite=True)
        # Empty field
        with self.assertRaises(ValueError):  # No maps and no spin
            fe = nmt.NmtField(self.msk, None, wcs=self.wcs,
                              beam=self.beam)
        fe = nmt.NmtField(self.msk, None, wcs=self.wcs,
                          beam=self.beam, spin=1)

        # Error checks
        for f in [fl, fe]:
            with self.assertRaises(ValueError):  # Query maps
                f.get_maps()
            with self.assertRaises(ValueError):  # Query templates
                f.get_templates()

    def test_field_alloc(self):
        # No templates
        f0 = nmt.NmtField(self.msk, [self.mps[0]],
                          beam=self.beam, wcs=self.wcs)
        f2 = nmt.NmtField(self.msk, [self.mps[1], self.mps[2]],
                          beam=self.beam, wcs=self.wcs)
        f2p = nmt.NmtField(self.msk, [self.mps[1], self.mps[2]],
                           beam=self.beam, wcs=self.wcs,
                           purify_e=True, purify_b=True, n_iter_mask_purify=10)
        self.assertTrue(normdiff(f0.get_maps()[0],
                                 (self.mps[0]*self.msk).flatten()) < 1E-10)
        self.assertTrue(normdiff(f2.get_maps()[0],
                                 (self.mps[1]*self.msk).flatten()) < 1E-10)
        self.assertTrue(normdiff(f2.get_maps()[1],
                                 (self.mps[2]*self.msk).flatten()) < 1E-10)
        self.assertTrue(1E-5*np.mean(np.fabs(f2p.get_maps()[0])) >
                        np.mean(np.fabs(f2p.get_maps()[0] -
                                        (self.mps[1]*self.msk).flatten())))
        self.assertTrue(1E-5*np.mean(np.fabs(f2p.get_maps()[1])) >
                        np.mean(np.fabs(f2p.get_maps()[1] -
                                        (self.mps[2]*self.msk).flatten())))
        self.assertEqual(len(f0.get_templates()), 0)
        self.assertEqual(len(f2.get_templates()), 0)
        self.assertEqual(len(f2p.get_templates()), 0)

        # With templates
        f0 = nmt.NmtField(self.msk, [self.mps[0]],
                          templates=np.array([[t[0]] for t in self.tmp]),
                          beam=self.beam, wcs=self.wcs)
        f2 = nmt.NmtField(self.msk, [self.mps[1], self.mps[2]],
                          templates=np.array([[t[1], t[2]] for t in self.tmp]),
                          beam=self.beam, wcs=self.wcs)
        # Map should be zero, since template =  map
        self.assertTrue(normdiff(f0.get_maps()[0],
                                 0*self.msk.flatten()) < 1E-10)
        self.assertTrue(normdiff(f2.get_maps()[0],
                                 0*self.msk.flatten()) < 1E-10)
        self.assertTrue(normdiff(f2.get_maps()[1],
                                 0*self.msk.flatten()) < 1E-10)
        self.assertEqual(len(f0.get_templates()), 5)
        self.assertEqual(len(f2.get_templates()), 5)

    def test_field_error(self):
        with self.assertRaises(ValueError):  # Not passing WCS
            nmt.NmtField(self.msk, [self.mps[0]], beam=self.beam)
        with self.assertRaises(ValueError):  # Passing 1D maps
            nmt.NmtField(self.msk.flatten(),
                         [self.mps[0]], beam=self.beam,
                         wcs=self.wcs)
        with self.assertRaises(AttributeError):  # Passing incorrect WCS
            nmt.NmtField(self.msk, [self.mps[0]], beam=self.beam, wcs=1)
        # Incorrect sky projection
        wcs = self.wcs.deepcopy()
        wcs.wcs.ctype[0] = 'RA---TAN'
        with self.assertRaises(ValueError):
            nmt.NmtField(self.msk, [self.mps[0]], beam=self.beam, wcs=wcs)
        # Incorrect reference pixel coords
        wcs = self.wcs.deepcopy()
        wcs.wcs.crval[1] = 96.
        with self.assertRaises(ValueError):
            nmt.NmtField(self.msk, [self.mps[0]], beam=self.beam, wcs=wcs)
        # Incorrect pixel sizes
        wcs = self.wcs.deepcopy()
        wcs.wcs.cdelt[1] = 1.01
        with self.assertRaises(ValueError):
            nmt.NmtField(self.msk, [self.mps[0]], beam=self.beam, wcs=wcs)
        # Input maps are too big
        msk = np.ones([self.ny, self.nx+10])
        with self.assertRaises(ValueError):
            nmt.NmtField(msk, [self.mps[0]], beam=self.beam, wcs=self.wcs)
        with self.assertRaises(ValueError):
            msk = np.ones([self.ny+100, self.nx])
            nmt.NmtField(msk, [msk], beam=self.beam, wcs=self.wcs)
        # Reference pixel has wrong pixel coordinates
        wcs = self.wcs.deepcopy()
        wcs.wcs.crpix[0] = 1.
        wcs.wcs.cdelt[0] = -1.
        with self.assertRaises(ValueError):
            nmt.NmtField(self.msk, [self.mps[0]],
                         beam=self.beam, wcs=wcs)
        with self.assertRaises(ValueError):  # Incorrect mask size
            nmt.NmtField(self.msk[:90], [self.mps[0]],
                         beam=self.beam, wcs=self.wcs)
        with self.assertRaises(ValueError):  # Incorrect maps size
            nmt.NmtField(self.msk, [self.mps[0, :90]],
                         beam=self.beam, wcs=self.wcs)
        with self.assertRaises(ValueError):  # Too many maps
            nmt.NmtField(self.msk, self.mps, wcs=self.wcs)
        with self.assertRaises(ValueError):  # Too many maps per template
            nmt.NmtField(self.msk, [self.mps[0]],
                         templates=self.tmp, beam=self.beam, wcs=self.wcs)
        with self.assertRaises(ValueError):
            # Number of maps per template does not match spin
            nmt.NmtField(self.msk, [self.mps[0]],
                         templates=[[t[0], t[1]] for t in self.tmp],
                         beam=self.beam, wcs=self.wcs)
        with self.assertRaises(ValueError):  # Incorrect template size
            nmt.NmtField(self.msk, [self.mps[0]],
                         templates=[[t[0, :90]] for t in self.tmp],
                         beam=self.beam, wcs=self.wcs)
        with self.assertRaises(ValueError):  # Passing crap as templates
            nmt.NmtField(self.msk, [self.mps[0]],
                         templates=1, beam=self.beam, wcs=self.wcs)
        with self.assertRaises(ValueError):  # Passing wrong beam
            nmt.NmtField(self.msk, [self.mps[0]],
                         templates=[[t[0]] for t in self.tmp],
                         beam=self.beam[:90], wcs=self.wcs)
        with self.assertRaises(ValueError):  # Passing crap as beam
            nmt.NmtField(self.msk, [self.mps[0]],
                         templates=[[t[0]] for t in self.tmp],
                         beam=1, wcs=self.wcs)


class TestFieldHPX(unittest.TestCase):
    def setUp(self):
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

    def test_field_masked(self):
        nside = 64
        b = nmt.NmtBin.from_nside_linear(nside, 16)
        msk = hp.read_map("test/benchmarks/msk.fits", verbose=False)
        mps = np.array(hp.read_map("test/benchmarks/mps.fits",
                                   verbose=False,
                                   field=[0, 1, 2]))
        mps_msk = np.array([m * msk for m in mps])
        f0 = nmt.NmtField(msk, [mps[0]])
        f0_msk = nmt.NmtField(msk, [mps_msk[0]],
                              masked_on_input=True)
        f2 = nmt.NmtField(msk, [mps[1], mps[2]])
        f2_msk = nmt.NmtField(msk, [mps_msk[1], mps_msk[2]],
                              masked_on_input=True)
        w00 = nmt.NmtWorkspace()
        w00.compute_coupling_matrix(f0, f0, b)
        w02 = nmt.NmtWorkspace()
        w02.compute_coupling_matrix(f0, f2, b)
        w22 = nmt.NmtWorkspace()
        w22.compute_coupling_matrix(f2, f2, b)

        def mkcl(w, f, g):
            return w.decouple_cell(nmt.compute_coupled_cell(f, g))

        c00 = mkcl(w00, f0, f0).flatten()
        c02 = mkcl(w02, f0, f2).flatten()
        c22 = mkcl(w22, f2, f2).flatten()
        c00_msk = mkcl(w00, f0_msk, f0_msk).flatten()
        c02_msk = mkcl(w02, f0_msk, f2_msk).flatten()
        c22_msk = mkcl(w22, f2_msk, f2_msk).flatten()
        self.assertTrue(np.all(np.fabs(c00-c00_msk) /
                               np.mean(c00) < 1E-10))
        self.assertTrue(np.all(np.fabs(c02-c02_msk) /
                               np.mean(c02) < 1E-10))
        self.assertTrue(np.all(np.fabs(c22-c22_msk) /
                               np.mean(c22) < 1E-10))

    def test_field_alloc(self):
        # No templates
        f0 = nmt.NmtField(self.msk, [self.mps[0]],
                          beam=self.beam)
        f2 = nmt.NmtField(self.msk, [self.mps[1], self.mps[2]],
                          beam=self.beam)
        f2p = nmt.NmtField(self.msk, [self.mps[1], self.mps[2]],
                           beam=self.beam,
                           purify_e=True, purify_b=True,
                           n_iter_mask_purify=10)
        self.assertTrue(normdiff(f0.get_maps()[0],
                                 self.mps[0]*self.msk) < 1E-10)
        self.assertTrue(normdiff(f2.get_maps()[0],
                                 self.mps[1]*self.msk) < 1E-10)
        self.assertTrue(normdiff(f2.get_maps()[1],
                                 self.mps[2]*self.msk) < 1E-10)
        self.assertTrue(1E-5*np.mean(np.fabs(f2p.get_maps()[0])) >
                        np.mean(np.fabs(f2p.get_maps()[0] -
                                        self.mps[1]*self.msk)))
        self.assertTrue(1E-5*np.mean(np.fabs(f2p.get_maps()[1])) >
                        np.mean(np.fabs(f2p.get_maps()[1] -
                                        self.mps[2]*self.msk)))
        self.assertEqual(len(f0.get_templates()), 0)
        self.assertEqual(len(f2.get_templates()), 0)
        self.assertEqual(len(f2p.get_templates()), 0)

        # With templates
        f0 = nmt.NmtField(self.msk, [self.mps[0]],
                          templates=np.array([[t[0]] for t in self.tmp]),
                          beam=self.beam)
        f2 = nmt.NmtField(self.msk, [self.mps[1], self.mps[2]],
                          templates=np.array([[t[1], t[2]] for t in self.tmp]),
                          beam=self.beam)
        # Map should be zero, since template =  map
        self.assertTrue(normdiff(f0.get_maps()[0], 0*self.msk) < 1E-10)
        self.assertTrue(normdiff(f2.get_maps()[0], 0*self.msk) < 1E-10)
        self.assertTrue(normdiff(f2.get_maps()[1], 0*self.msk) < 1E-10)
        self.assertEqual(len(f0.get_templates()), 5)
        self.assertEqual(len(f2.get_templates()), 5)

    def test_field_lite(self):
        # Lite field
        fl = nmt.NmtField(self.msk, [self.mps[0]],
                          beam=self.beam, lite=True)
        # Empty field
        with self.assertRaises(ValueError):  # No maps and no spin
            fe = nmt.NmtField(self.msk, None, beam=self.beam)
        fe = nmt.NmtField(self.msk, None, beam=self.beam, spin=1)

        # Error checks
        for f in [fl, fe]:
            with self.assertRaises(ValueError):  # Query maps
                f.get_maps()
            with self.assertRaises(ValueError):  # Query templates
                f.get_templates()

    def test_field_error(self):
        with self.assertRaises(ValueError):  # Incorrect mask size
            nmt.NmtField(self.msk[:15], self.mps)
        with self.assertRaises(ValueError):  # Incorrect map size
            nmt.NmtField(self.msk, [self.mps[0, :15]])
        with self.assertRaises(ValueError):  # Incorrect template size
            nmt.NmtField(self.msk, [self.mps[0]],
                         templates=[[self.tmp[0, 0, :15]]])
        with self.assertRaises(ValueError):  # Passing 3 maps!
            nmt.NmtField(self.msk, self.mps)
        with self.assertRaises(ValueError):  # Passing 3 template maps!
            nmt.NmtField(self.msk, [self.mps[1], self.mps[2]],
                         templates=self.tmp)
        with self.assertRaises(ValueError):  # Passing crap as templates
            nmt.NmtField(self.msk, [self.mps[1], self.mps[2]],
                         templates=1)
        with self.assertRaises(ValueError):  # Passing wrong beam
            nmt.NmtField(self.msk, [self.mps[0]], beam=self.beam[:30])
        with self.assertRaises(ValueError):  # Passing crap as beam
            nmt.NmtField(self.msk, [self.mps[0]], beam=1)

        # Automatically assign spin = 0 for a single map
        f = nmt.NmtField(self.msk, [self.mps[0]], n_iter=0)
        self.assertTrue(f.fl.spin == 0)
        # Automatically assign spin = 2 for 2 maps
        f = nmt.NmtField(self.msk, [self.mps[1], self.mps[2]], n_iter=0)
        self.assertTrue(f.fl.spin == 2)
        with self.assertRaises(ValueError):  # Spin=0 but 2 maps
            f = nmt.NmtField(self.msk, [self.mps[1], self.mps[2]],
                             spin=0, n_iter=0)
        with self.assertRaises(ValueError):  # Spin=1 but 1 maps
            f = nmt.NmtField(self.msk, [self.mps[0]], spin=1, n_iter=0)
        with self.assertRaises(ValueError):
            f = nmt.NmtField(self.msk, [self.mps[1], self.mps[2]], spin=1,
                             purify_b=True, n_iter=0)


class TestFieldFsk(unittest.TestCase):
    def setUp(self):
        # This is to avoid showing an ugly warning that
        # has nothing to do with pymaster
        if (sys.version_info > (3, 1)):
            warnings.simplefilter("ignore", ResourceWarning)

        self.ntemp = 5
        self.nx = 141
        self.ny = 311
        self.lx = np.radians(1.)
        self.ly = np.radians(1.)
        self.npix = self.nx*self.ny
        self.msk = np.ones([self.ny, self.nx])
        self.lmax = np.sqrt((self.nx*np.pi/self.lx)**2 +
                            (self.ny*np.pi/self.ly)**2)
        self.nell = 30
        xarr = np.arange(self.nx)*self.lx/self.nx
        yarr = np.arange(self.ny)*self.ly/self.ny
        i0_x = 2
        i0_y = 3
        k0_x = i0_x*2*np.pi/self.lx
        k0_y = i0_y*2*np.pi/self.ly
        phase = k0_x*xarr[None, :]+k0_y*yarr[:, None]
        cphi0 = k0_x/np.sqrt(k0_x**2+k0_y**2)
        sphi0 = k0_y/np.sqrt(k0_x**2+k0_y**2)
        c2phi0 = cphi0**2-sphi0**2
        s2phi0 = 2*sphi0*cphi0
        self.mps = np.array([2*np.pi*np.cos(phase)/(self.lx*self.ly),
                             2*np.pi*c2phi0*np.cos(phase)/(self.lx*self.ly),
                             -2*np.pi*s2phi0*np.cos(phase)/(self.lx*self.ly)])
        self.tmp = np.array([self.mps.copy() for i in range(self.ntemp)])
        self.beam = np.array([np.arange(self.nell)*self.lmax/(self.nell-1.),
                              np.ones(self.nell)])

    def test_field_masked(self):
        wcs, msk = read_flat_map("test/benchmarks/msk_flat.fits")
        ny, nx = msk.shape
        lx = np.radians(np.fabs(nx*wcs.wcs.cdelt[0]))
        ly = np.radians(np.fabs(ny*wcs.wcs.cdelt[1]))
        mps = np.array([read_flat_map("test/benchmarks/mps_flat.fits",
                                      i_map=i)[1] for i in range(3)])
        mps_msk = np.array([m * msk for m in mps])
        d_ell = 20
        lmax = 500.
        ledges = np.arange(int(lmax/d_ell)+1)*d_ell+2
        b = nmt.NmtBinFlat(ledges[:-1], ledges[1:])

        f0 = nmt.NmtFieldFlat(lx, ly, msk, [mps[0]])
        f0_msk = nmt.NmtFieldFlat(lx, ly, msk, [mps_msk[0]],
                                  masked_on_input=True)
        f2 = nmt.NmtFieldFlat(lx, ly, msk, [mps[1], mps[2]])
        f2_msk = nmt.NmtFieldFlat(lx, ly, msk,
                                  [mps_msk[1], mps_msk[2]],
                                  masked_on_input=True)
        w00 = nmt.NmtWorkspaceFlat()
        w00.compute_coupling_matrix(f0, f0, b)
        w02 = nmt.NmtWorkspaceFlat()
        w02.compute_coupling_matrix(f0, f2, b)
        w22 = nmt.NmtWorkspaceFlat()
        w22.compute_coupling_matrix(f2, f2, b)

        def mkcl(w, f, g):
            return w.decouple_cell(nmt.compute_coupled_cell_flat(f, g, b))

        c00 = mkcl(w00, f0, f0).flatten()
        c02 = mkcl(w02, f0, f2).flatten()
        c22 = mkcl(w22, f2, f2).flatten()
        c00_msk = mkcl(w00, f0_msk, f0_msk).flatten()
        c02_msk = mkcl(w02, f0_msk, f2_msk).flatten()
        c22_msk = mkcl(w22, f2_msk, f2_msk).flatten()
        self.assertTrue(np.all(np.fabs(c00-c00_msk) /
                               np.mean(c00) < 1E-10))
        self.assertTrue(np.all(np.fabs(c02-c02_msk) /
                               np.mean(c02) < 1E-10))
        self.assertTrue(np.all(np.fabs(c22-c22_msk) /
                               np.mean(c22) < 1E-10))

    def test_field_lite(self):
        # Lite field
        fl = nmt.NmtFieldFlat(self.lx, self.ly, self.msk,
                              [self.mps[0]], beam=self.beam,
                              lite=True)
        # Empty field
        with self.assertRaises(ValueError):  # No maps and no spin
            fe = nmt.NmtFieldFlat(self.lx, self.ly, self.msk,
                                  None, beam=self.beam)
        fe = nmt.NmtFieldFlat(self.lx, self.ly, self.msk,
                              None, beam=self.beam, spin=1)

        # Error checks
        for f in [fl, fe]:
            with self.assertRaises(ValueError):  # Query maps
                f.get_maps()
            with self.assertRaises(ValueError):  # Query templates
                f.get_templates()

    def test_field_flat_alloc(self):
        # No templates
        f0 = nmt.NmtFieldFlat(self.lx, self.ly, self.msk,
                              [self.mps[0]], beam=self.beam)
        f2 = nmt.NmtFieldFlat(self.lx, self.ly, self.msk,
                              [self.mps[1], self.mps[2]], beam=self.beam)
        f2p = nmt.NmtFieldFlat(self.lx, self.ly, self.msk,
                               [self.mps[1], self.mps[2]], beam=self.beam,
                               purify_e=True, purify_b=True)
        self.assertTrue(normdiff(f0.get_maps()[0],
                                 self.mps[0]*self.msk) < 1E-10)
        self.assertTrue(normdiff(f2.get_maps()[0],
                                 self.mps[1]*self.msk) < 1E-10)
        self.assertTrue(normdiff(f2.get_maps()[1],
                                 self.mps[2]*self.msk) < 1E-10)
        self.assertTrue(normdiff(f2p.get_maps()[0],
                                 self.mps[1]*self.msk) < 1E-10)
        self.assertTrue(normdiff(f2p.get_maps()[1],
                                 self.mps[2]*self.msk) < 1E-10)
        self.assertEqual(len(f0.get_templates()), 0)
        self.assertEqual(len(f2.get_templates()), 0)
        self.assertEqual(len(f2p.get_templates()), 0)

        # With templates
        f0 = nmt.NmtFieldFlat(self.lx, self.ly, self.msk,
                              [self.mps[0]], beam=self.beam,
                              templates=np.array([[t[0]]
                                                  for t in self.tmp]))
        f2 = nmt.NmtFieldFlat(self.lx, self.ly, self.msk,
                              [self.mps[1], self.mps[2]], beam=self.beam,
                              templates=np.array([[t[1], t[2]]
                                                  for t in self.tmp]))
        # Map should be zero, since template =  map
        self.assertTrue(normdiff(f0.get_maps()[0], 0*self.msk) < 1E-10)
        self.assertTrue(normdiff(f2.get_maps()[0], 0*self.msk) < 1E-10)
        self.assertTrue(normdiff(f2.get_maps()[1], 0*self.msk) < 1E-10)
        self.assertEqual(len(f0.get_templates()), 5)
        self.assertEqual(len(f2.get_templates()), 5)

    def test_field_flat_errors(self):
        with self.assertRaises(ValueError):  # Incorrect map sizes
            nmt.NmtFieldFlat(self.lx, -self.ly, self.msk, [self.mps[0]])
        with self.assertRaises(ValueError):
            nmt.NmtFieldFlat(-self.lx, self.ly, self.msk, [self.mps[0]])
        with self.assertRaises(ValueError):  # Mismatching map dimensions
            nmt.NmtFieldFlat(self.lx, self.ly, self.msk,
                             [self.mps[0, :self.ny//2]])
        with self.assertRaises(ValueError):  # Mismatching template dimensions
            nmt.NmtFieldFlat(self.lx, self.ly, self.msk, [self.mps[0]],
                             templates=np.array([[t[0, :self.ny//2]]
                                                 for t in self.tmp]))
        with self.assertRaises(ValueError):  # Passing 3 templates!
            nmt.NmtFieldFlat(self.lx, self.ly, self.msk, self.mps)
        with self.assertRaises(ValueError):  # Passing 3 templates!
            nmt.NmtFieldFlat(self.lx, self.ly, self.msk,
                             [self.mps[0]], templates=self.tmp)
        with self.assertRaises(ValueError):  # Passing crap templates
            nmt.NmtFieldFlat(self.lx, self.ly, self.msk, [self.mps[0]],
                             templates=1)
        with self.assertRaises(ValueError):  # Passing crap beam
            nmt.NmtFieldFlat(self.lx, self.ly, self.msk, [self.mps[0]], beam=1)

        # Automatically assign spin = 0 for a single map
        f = nmt.NmtFieldFlat(self.lx, self.ly, self.msk,
                             [self.mps[0]])
        self.assertTrue(f.fl.spin == 0)
        # Automatically assign spin = 2 for 2 maps
        f = nmt.NmtFieldFlat(self.lx, self.ly, self.msk,
                             [self.mps[1], self.mps[2]])
        self.assertTrue(f.fl.spin == 2)
        with self.assertRaises(ValueError):  # Spin=0 but 2 maps
            f = nmt.NmtFieldFlat(self.lx, self.ly, self.msk,
                                 [self.mps[1], self.mps[2]], spin=0)
        with self.assertRaises(ValueError):  # Spin=1 but 1 maps
            f = nmt.NmtFieldFlat(self.lx, self.ly, self.msk,
                                 [self.mps[0]], spin=1)
        with self.assertRaises(ValueError):
            f = nmt.NmtFieldFlat(self.lx, self.ly, self.msk,
                                 [self.mps[1], self.mps[2]], spin=1,
                                 purify_b=True)


if __name__ == '__main__':
    unittest.main()
