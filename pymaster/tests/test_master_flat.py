import pytest
import numpy as np
import pymaster as nmt
import warnings
import sys
from .utils import read_flat_map


class WorkspaceTesterFlat(object):
    def __init__(self):
        # This is to avoid showing an ugly warning
        # that has nothing to do with pymaster
        if (sys.version_info > (3,  1)):
            warnings.simplefilter("ignore",  ResourceWarning)

        self.wcs, self.msk = read_flat_map("test/benchmarks/msk_flat.fits")
        (self.ny, self.nx) = self.msk.shape
        self.lx = np.radians(np.fabs(self.nx*self.wcs.wcs.cdelt[0]))
        self.ly = np.radians(np.fabs(self.ny*self.wcs.wcs.cdelt[1]))
        self.mps = np.array([read_flat_map("test/benchmarks/mps_flat.fits",
                                           i_map=i)[1] for i in range(3)])
        self.mps_s1 = np.array([read_flat_map("test/benchmarks/"
                                              "mps_sp1_flat.fits",
                                              i_map=i)[1] for i in range(3)])
        self.tmp = np.array([read_flat_map("test/benchmarks/tmp_flat.fits",
                                           i_map=i)[1] for i in range(3)])
        self.d_ell = 20
        self.lmax = 500.
        ledges = np.arange(int(self.lmax/self.d_ell)+1)*self.d_ell+2
        self.b = nmt.NmtBinFlat(ledges[:-1], ledges[1:])
        self.leff = self.b.get_effective_ells()
        self.f0 = nmt.NmtFieldFlat(self.lx, self.ly, self.msk,
                                   [self.mps[0]])
        self.f0_half = nmt.NmtFieldFlat(self.lx, self.ly,
                                        self.msk[:self.ny//2,
                                                 :self.nx//2],
                                        [self.mps[0,
                                                  :self.ny//2,
                                                  :self.nx//2]])
        ledges_half = ledges[:len(ledges)//2]
        self.b_half = nmt.NmtBinFlat(ledges_half[:-1],
                                     ledges_half[1:])

        dd = np.loadtxt("test/benchmarks/cls_lss.txt",
                        unpack=True)
        l, cltt, clee, clbb, clte, nltt, nlee, nlbb, nlte = dd
        self.ll = l[:]
        self.cltt = cltt[:]
        self.clee = clee[:]
        self.clbb = clbb[:]
        self.clte = clte[:]
        self.nltt = nltt[:]
        self.nlee = nlee[:]
        self.nlbb = nlbb[:]
        self.nlte = nlte[:]
        self.n_good = np.zeros([1, len(l)])
        self.n_bad = np.zeros([2, len(l)])
        self.nb_good = np.zeros([1, self.b.bin.n_bands])
        self.nb_bad = np.zeros([2, self.b.bin.n_bands])


WT = WorkspaceTesterFlat()


def test_lite_pure():
    f0 = nmt.NmtFieldFlat(WT.lx, WT.ly, WT.msk,
                          [WT.mps[0]])
    f2l = nmt.NmtFieldFlat(WT.lx, WT.ly, WT.msk,
                           [WT.mps[1], WT.mps[2]],
                           purify_b=True, lite=True)
    f2e = nmt.NmtFieldFlat(WT.lx, WT.ly, WT.msk,
                           None, purify_b=True,
                           lite=True, spin=2)
    nlth = np.array([WT.nlte, 0*WT.nlte])
    w = nmt.NmtWorkspaceFlat()
    w.compute_coupling_matrix(f0, f2e, WT.b)
    clb = w.couple_cell(WT.ll, nlth)
    cl = w.decouple_cell(nmt.compute_coupled_cell_flat(f0,
                                                       f2l,
                                                       WT.b),
                         cl_bias=clb)
    tl = np.loadtxt("test/benchmarks/bm_f_nc_yp_c02.txt",
                    unpack=True)[1:, :]
    assert (np.fabs(cl-tl) <=
            np.fmin(np.fabs(cl),
                    np.fabs(tl))*1E-5).all()


def test_lite_cont():
    f0 = nmt.NmtFieldFlat(WT.lx, WT.ly, WT.msk,
                          [WT.mps[0]],
                          templates=[[WT.tmp[0]]])
    tmps = [[WT.tmp[1], WT.tmp[2]]]
    f2 = nmt.NmtFieldFlat(WT.lx, WT.ly, WT.msk,
                          [WT.mps[1], WT.mps[2]],
                          templates=tmps)
    f2l = nmt.NmtFieldFlat(WT.lx, WT.ly, WT.msk,
                           [WT.mps[1], WT.mps[2]],
                           templates=tmps, lite=True)
    f2e = nmt.NmtFieldFlat(WT.lx, WT.ly, WT.msk,
                           None, lite=True, spin=2)
    clth = np.array([WT.clte, 0*WT.clte])
    nlth = np.array([WT.nlte, 0*WT.nlte])
    w = nmt.NmtWorkspaceFlat()
    w.compute_coupling_matrix(f0, f2e, WT.b)
    clb = w.couple_cell(WT.ll, nlth)
    dlb = nmt.deprojection_bias_flat(f0, f2,
                                     WT.b, WT.ll,
                                     clth+nlth)
    clb += dlb
    cl = w.decouple_cell(nmt.compute_coupled_cell_flat(f0,
                                                       f2l,
                                                       WT.b),
                         cl_bias=clb)
    tl = np.loadtxt("test/benchmarks/bm_f_yc_np_c02.txt",
                    unpack=True)[1:, :]
    tlb = np.loadtxt("test/benchmarks/bm_f_yc_np_cb02.txt",
                     unpack=True)[1:, :]
    assert (np.fabs(dlb-tlb) <=
            np.fmin(np.fabs(dlb),
                    np.fabs(tlb))*1E-5).all()
    assert (np.fabs(cl-tl) <=
            np.fmin(np.fabs(cl),
                    np.fabs(tl))*1E-5).all()


def test_spin1():
    prefix = "test/benchmarks/bm_f_sp1"
    f0 = nmt.NmtFieldFlat(WT.lx, WT.ly, WT.msk,
                          [WT.mps_s1[0]])
    f1 = nmt.NmtFieldFlat(WT.lx, WT.ly, WT.msk,
                          [WT.mps_s1[1], WT.mps_s1[2]],
                          spin=1)
    f = [f0, f1]

    for ip1 in range(2):
        for ip2 in range(ip1, 2):
            w = nmt.NmtWorkspaceFlat()
            w.compute_coupling_matrix(f[ip1], f[ip2], WT.b)
            cl = w.decouple_cell(nmt.compute_coupled_cell_flat(f[ip1],
                                                               f[ip2],
                                                               WT.b))[0]
            tl = np.loadtxt(prefix+'_c%d%d.txt' % (ip1, ip2),
                            unpack=True)[1]
            assert (np.fabs(cl-tl) <=
                    np.fmin(np.fabs(cl),
                            np.fabs(tl))*1E-5).all()


def mastest(wtemp, wpure, do_teb=False):
    prefix = "test/benchmarks/bm_f"
    if wtemp:
        prefix += "_yc"
        f0 = nmt.NmtFieldFlat(WT.lx, WT.ly, WT.msk,
                              [WT.mps[0]],
                              templates=[[WT.tmp[0]]])
        f2 = nmt.NmtFieldFlat(WT.lx, WT.ly, WT.msk,
                              [WT.mps[1], WT.mps[2]],
                              templates=[[WT.tmp[1],
                                          WT.tmp[2]]],
                              purify_b=wpure)
    else:
        prefix += "_nc"
        f0 = nmt.NmtFieldFlat(WT.lx, WT.ly, WT.msk,
                              [WT.mps[0]])
        f2 = nmt.NmtFieldFlat(WT.lx, WT.ly, WT.msk,
                              [WT.mps[1], WT.mps[2]],
                              purify_b=wpure)
    f = [f0, f2]

    if wpure:
        prefix += "_yp"
    else:
        prefix += "_np"

    for ip1 in range(2):
        for ip2 in range(ip1, 2):
            if ip1 == ip2 == 0:
                clth = np.array([WT.cltt])
                nlth = np.array([WT.nltt])
            elif ip1 == ip2 == 1:
                clth = np.array([WT.clee, 0*WT.clee,
                                 0*WT.clbb, WT.clbb])
                nlth = np.array([WT.nlee, 0*WT.nlee,
                                 0*WT.nlbb, WT.nlbb])
            else:
                clth = np.array([WT.clte, 0*WT.clte])
                nlth = np.array([WT.nlte, 0*WT.nlte])
            w = nmt.NmtWorkspaceFlat()
            w.compute_coupling_matrix(f[ip1], f[ip2], WT.b)
            clb = w.couple_cell(WT.ll, nlth)
            if wtemp:
                dlb = nmt.deprojection_bias_flat(f[ip1], f[ip2],
                                                 WT.b, WT.ll,
                                                 clth+nlth)
                tlb = np.loadtxt(prefix+'_cb%d%d.txt' % (2*ip1, 2*ip2),
                                 unpack=True)[1:, :]
                assert (np.fabs(dlb-tlb) <=
                        np.fmin(np.fabs(dlb),
                                np.fabs(tlb))*1E-5).all()
                clb += dlb
            cl = w.decouple_cell(nmt.compute_coupled_cell_flat(f[ip1],
                                                               f[ip2],
                                                               WT.b),
                                 cl_bias=clb)
            tl = np.loadtxt(prefix+'_c%d%d.txt' % (2*ip1, 2*ip2),
                            unpack=True)[1:, :]
            assert (np.fabs(cl-tl) <=
                    np.fmin(np.fabs(cl),
                            np.fabs(tl))*1E-5).all()

    # TEB
    if do_teb:
        clth = np.array([WT.cltt, WT.clte, 0*WT.clte, WT.clee,
                         0*WT.clee, 0*WT.clbb, WT.clbb])
        nlth = np.array([WT.nltt, WT.nlte, 0*WT.nlte, WT.nlee,
                         0*WT.nlee, 0*WT.nlbb, WT.nlbb])
        w = nmt.NmtWorkspaceFlat()
        w.compute_coupling_matrix(f[0], f[1], WT.b, is_teb=True)
        c00 = nmt.compute_coupled_cell_flat(f[0], f[0], WT.b)
        c02 = nmt.compute_coupled_cell_flat(f[0], f[1], WT.b)
        c22 = nmt.compute_coupled_cell_flat(f[1], f[1], WT.b)
        cl = np.array([c00[0], c02[0], c02[1], c22[0],
                       c22[1], c22[2], c22[3]])
        t00 = np.loadtxt(prefix+'_c00.txt', unpack=True)[1:, :]
        t02 = np.loadtxt(prefix+'_c02.txt', unpack=True)[1:, :]
        t22 = np.loadtxt(prefix+'_c22.txt', unpack=True)[1:, :]
        tl = np.array([t00[0], t02[0], t02[1], t22[0],
                       t22[1], t22[2], t22[3]])
        cl = w.decouple_cell(cl, cl_bias=w.couple_cell(WT.ll, nlth))
        assert (np.fabs(cl-tl) <=
                np.fmin(np.fabs(cl),
                        np.fabs(tl))*1E-5).all()


@pytest.mark.parametrize('wtemp,wpure,do_teb',
                         [(False, False, True),
                          (False, True, True),
                          (False, False, False),
                          (True, False, False),
                          (False, True, False),
                          (True, True, False)])
def test_workspace_flat_master_teb_np(wtemp, wpure, do_teb):
    mastest(wtemp, wpure, do_teb=do_teb)


def test_workspace_flat_io():
    w = nmt.NmtWorkspaceFlat()
    with pytest.raises(RuntimeError):  # Invalid writing
        w.write_to("test/wspc.fits")
    w.read_from("test/benchmarks/bm_f_yc_yp_w02.fits")  # OK read
    assert WT.msk.shape == (w.wsp.fs.ny, w.wsp.fs.nx)
    with pytest.raises(RuntimeError):  # Can't write on that file
        w.write_to("tests/wspc.fits")
    with pytest.raises(RuntimeError):  # File doesn't exist
        w.read_from("none")


def test_workspace_flat_methods():
    w = nmt.NmtWorkspaceFlat()
    w.compute_coupling_matrix(WT.f0, WT.f0, WT.b)  # OK init
    assert WT.msk.shape == (w.wsp.fs.ny, w.wsp.fs.nx)
    with pytest.raises(RuntimeError):  # Incompatible resolutions
        w.compute_coupling_matrix(WT.f0, WT.f0_half, WT.b)
    with pytest.raises(RuntimeError):  # Wrong fields for TEB
        w.compute_coupling_matrix(WT.f0, WT.f0, WT.b, is_teb=True)

    w.compute_coupling_matrix(WT.f0, WT.f0, WT.b)

    # Test couple_cell
    c = w.couple_cell(WT.ll, WT.n_good)
    assert c.shape == (1, WT.b.bin.n_bands)
    with pytest.raises(ValueError):
        w.couple_cell(WT.ll, WT.n_bad)
    with pytest.raises(ValueError):
        w.couple_cell(WT.ll, WT.n_good[:, :len(WT.ll)//2])

    # Test decouple_cell
    c = w.decouple_cell(WT.nb_good, WT.nb_good, WT.nb_good)
    assert c.shape == (1, WT.b.bin.n_bands)
    with pytest.raises(ValueError):
        w.decouple_cell(WT.nb_bad)
    with pytest.raises(ValueError):
        w.decouple_cell(WT.n_good)
    with pytest.raises(ValueError):
        w.decouple_cell(WT.nb_good, cl_bias=WT.nb_bad)
    with pytest.raises(ValueError):
        w.decouple_cell(WT.nb_good, cl_bias=WT.n_good)
    with pytest.raises(ValueError):
        w.decouple_cell(WT.nb_good, cl_noise=WT.nb_bad)
    with pytest.raises(ValueError):
        w.decouple_cell(WT.nb_good, cl_noise=WT.n_good)


def test_workspace_flat_full_master():
    w = nmt.NmtWorkspaceFlat()
    w.compute_coupling_matrix(WT.f0, WT.f0, WT.b)
    w_half = nmt.NmtWorkspaceFlat()
    w_half.compute_coupling_matrix(WT.f0_half, WT.f0_half, WT.b_half)

    c = nmt.compute_full_master_flat(WT.f0, WT.f0, WT.b)
    assert c.shape == (1, WT.b.bin.n_bands)

    c = nmt.compute_full_master_flat(WT.f0, WT.f0, WT.b,
                                     ells_guess=np.arange(1000),
                                     cl_guess=np.zeros([1, 1000]))
    assert c.shape == (1, WT.b.bin.n_bands)

    with pytest.raises(ValueError):  # Incompatible resolutions
        nmt.compute_full_master_flat(WT.f0, WT.f0_half, WT.b)
    c = nmt.compute_full_master_flat(WT.f0, WT.f0, WT.b_half)
    # Passing correct input workspace
    # Computing from correct wsp
    c = nmt.compute_full_master_flat(WT.f0, WT.f0, WT.b,
                                     workspace=w)
    assert c.shape == (1, WT.b.bin.n_bands)
    with pytest.raises(RuntimeError):  # Incompatible bandpowers
        nmt.compute_full_master_flat(WT.f0, WT.f0, WT.b_half,
                                     workspace=w)
    with pytest.raises(RuntimeError):  # Incompatible workspace
        nmt.compute_full_master_flat(WT.f0, WT.f0, WT.b,
                                     workspace=w_half)
    # Incorrect input spectra
    with pytest.raises(ValueError):
        nmt.compute_full_master_flat(WT.f0, WT.f0, WT.b,
                                     cl_noise=WT.nb_bad, workspace=w)
    with pytest.raises(ValueError):
        nmt.compute_full_master_flat(WT.f0, WT.f0, WT.b,
                                     cl_noise=WT.n_good, workspace=w)
    with pytest.raises(ValueError):  # Non ell-values
        nmt.compute_full_master_flat(WT.f0, WT.f0, WT.b,
                                     cl_noise=WT.nb_good,
                                     cl_guess=WT.n_good, workspace=w)
    with pytest.raises(ValueError):  # Wrong cl_guess
        nmt.compute_full_master_flat(WT.f0, WT.f0, WT.b,
                                     cl_noise=WT.nb_good,
                                     cl_guess=WT.n_bad,
                                     ells_guess=WT.ll, workspace=w)
    with pytest.raises(ValueError):  # Wrong cl_guess
        nmt.compute_full_master_flat(WT.f0, WT.f0, WT.b,
                                     cl_noise=WT.nb_good,
                                     cl_guess=WT.nb_good,
                                     ells_guess=WT.ll, workspace=w)


def test_workspace_flat_deprojection_bias():
    # Test derojection_bias_flat
    c = nmt.deprojection_bias_flat(WT.f0, WT.f0, WT.b,
                                   WT.ll, WT.n_good)
    assert c.shape == (1, WT.b.bin.n_bands)
    with pytest.raises(ValueError):  # Wrong cl_guess
        nmt.deprojection_bias_flat(WT.f0, WT.f0, WT.b,
                                   WT.ll, WT.n_bad)
    with pytest.raises(ValueError):  # Wrong cl_guess
        nmt.deprojection_bias_flat(WT.f0, WT.f0, WT.b,
                                   WT.ll, WT.nb_good)
    with pytest.raises(RuntimeError):  # Incompatible resolutions
        nmt.deprojection_bias_flat(WT.f0, WT.f0_half,
                                   WT.b, WT.ll, WT.n_good)


def test_workspace_flat_compute_coupled_cell():
    # Test compute_coupled_cell_flat
    c = nmt.compute_coupled_cell_flat(WT.f0, WT.f0, WT.b)
    assert c.shape == (1, WT.b.bin.n_bands)
    with pytest.raises(ValueError):  # Different resolutions
        nmt.compute_coupled_cell_flat(WT.f0, WT.f0_half, WT.b)
