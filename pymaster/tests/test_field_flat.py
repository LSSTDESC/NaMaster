import numpy as np
import pymaster as nmt
import warnings
import sys
import pytest
from .utils import normdiff, read_flat_map


class FieldTesterFlat(object):
    def __init__(self):
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


FT = FieldTesterFlat()


def test_field_masked():
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
    assert (np.all(np.fabs(c00-c00_msk) /
                   np.mean(c00) < 1E-10))
    assert (np.all(np.fabs(c02-c02_msk) /
                   np.mean(c02) < 1E-10))
    assert (np.all(np.fabs(c22-c22_msk) /
                   np.mean(c22) < 1E-10))


def test_field_lite():
    # Lite field
    fl = nmt.NmtFieldFlat(FT.lx, FT.ly, FT.msk,
                          [FT.mps[0]], beam=FT.beam,
                          lite=True)
    # Empty field
    with pytest.raises(ValueError):  # No maps and no spin
        fe = nmt.NmtFieldFlat(FT.lx, FT.ly, FT.msk,
                              None, beam=FT.beam)
    fe = nmt.NmtFieldFlat(FT.lx, FT.ly, FT.msk,
                          None, beam=FT.beam, spin=1)

    # Error checks
    for f in [fl, fe]:
        with pytest.raises(ValueError):  # Query maps
            f.get_maps()
        with pytest.raises(ValueError):  # Query templates
            f.get_templates()


def test_field_flat_get_mask():
    msk = np.random.rand(FT.ny, FT.nx)
    f0 = nmt.NmtFieldFlat(FT.lx, FT.ly, msk, [FT.mps[0]], beam=FT.beam)
    mskb = f0.get_mask()
    assert np.amax(np.fabs(msk-mskb)/np.std(msk)) < 1E-5


def test_field_flat_alloc():
    # No templates
    f0 = nmt.NmtFieldFlat(FT.lx, FT.ly, FT.msk,
                          [FT.mps[0]], beam=FT.beam)
    f2 = nmt.NmtFieldFlat(FT.lx, FT.ly, FT.msk,
                          [FT.mps[1], FT.mps[2]], beam=FT.beam)
    f2p = nmt.NmtFieldFlat(FT.lx, FT.ly, FT.msk,
                           [FT.mps[1], FT.mps[2]], beam=FT.beam,
                           purify_e=True, purify_b=True)
    assert (normdiff(f0.get_maps()[0],
                     FT.mps[0]*FT.msk) < 1E-10)
    assert (normdiff(f2.get_maps()[0],
                     FT.mps[1]*FT.msk) < 1E-10)
    assert (normdiff(f2.get_maps()[1],
                     FT.mps[2]*FT.msk) < 1E-10)
    assert (normdiff(f2p.get_maps()[0],
                     FT.mps[1]*FT.msk) < 1E-10)
    assert (normdiff(f2p.get_maps()[1],
                     FT.mps[2]*FT.msk) < 1E-10)
    assert (len(f0.get_templates()) == 0)
    assert (len(f2.get_templates()) == 0)
    assert (len(f2p.get_templates()) == 0)

    # With templates
    f0 = nmt.NmtFieldFlat(FT.lx, FT.ly, FT.msk,
                          [FT.mps[0]], beam=FT.beam,
                          templates=np.array([[t[0]]
                                              for t in FT.tmp]))
    f2 = nmt.NmtFieldFlat(FT.lx, FT.ly, FT.msk,
                          [FT.mps[1], FT.mps[2]], beam=FT.beam,
                          templates=np.array([[t[1], t[2]]
                                              for t in FT.tmp]))
    # Map should be zero, since template =  map
    assert (normdiff(f0.get_maps()[0], 0*FT.msk) < 1E-10)
    assert (normdiff(f2.get_maps()[0], 0*FT.msk) < 1E-10)
    assert (normdiff(f2.get_maps()[1], 0*FT.msk) < 1E-10)
    assert (len(f0.get_templates()) == 5)
    assert (len(f2.get_templates()) == 5)


def test_field_flat_errors():
    with pytest.raises(ValueError):  # Incorrect map sizes
        nmt.NmtFieldFlat(FT.lx, -FT.ly, FT.msk, [FT.mps[0]])
    with pytest.raises(ValueError):
        nmt.NmtFieldFlat(-FT.lx, FT.ly, FT.msk, [FT.mps[0]])
    with pytest.raises(ValueError):  # Mismatching map dimensions
        nmt.NmtFieldFlat(FT.lx, FT.ly, FT.msk,
                         [FT.mps[0, :FT.ny//2]])
    with pytest.raises(ValueError):  # Mismatching template dimensions
        nmt.NmtFieldFlat(FT.lx, FT.ly, FT.msk, [FT.mps[0]],
                         templates=np.array([[t[0, :FT.ny//2]]
                                             for t in FT.tmp]))
    with pytest.raises(ValueError):  # Passing 3 templates!
        nmt.NmtFieldFlat(FT.lx, FT.ly, FT.msk, FT.mps)
    with pytest.raises(ValueError):  # Passing 3 templates!
        nmt.NmtFieldFlat(FT.lx, FT.ly, FT.msk,
                         [FT.mps[0]], templates=FT.tmp)
    with pytest.raises(ValueError):  # Passing crap templates
        nmt.NmtFieldFlat(FT.lx, FT.ly, FT.msk, [FT.mps[0]],
                         templates=1)
    with pytest.raises(ValueError):  # Passing crap beam
        nmt.NmtFieldFlat(FT.lx, FT.ly, FT.msk, [FT.mps[0]], beam=1)

    # Automatically assign spin = 0 for a single map
    f = nmt.NmtFieldFlat(FT.lx, FT.ly, FT.msk,
                         [FT.mps[0]])
    assert (f.fl.spin == 0)
    # Automatically assign spin = 2 for 2 maps
    f = nmt.NmtFieldFlat(FT.lx, FT.ly, FT.msk,
                         [FT.mps[1], FT.mps[2]])
    assert (f.fl.spin == 2)
    with pytest.raises(ValueError):  # Spin=0 but 2 maps
        f = nmt.NmtFieldFlat(FT.lx, FT.ly, FT.msk,
                             [FT.mps[1], FT.mps[2]], spin=0)
    with pytest.raises(ValueError):  # Spin=1 but 1 maps
        f = nmt.NmtFieldFlat(FT.lx, FT.ly, FT.msk,
                             [FT.mps[0]], spin=1)
    with pytest.raises(ValueError):
        f = nmt.NmtFieldFlat(FT.lx, FT.ly, FT.msk,
                             [FT.mps[1], FT.mps[2]], spin=1,
                             purify_b=True)


def test_field_flat_ell_sampling():
    fl = nmt.NmtFieldFlat(FT.lx, FT.ly, FT.msk, None, spin=0)
    ells = fl.get_ell_sampling()
    assert ells[0] == fl.fl.fs.dell*0.5
    assert np.all(np.diff(ells) == fl.fl.fs.dell)
