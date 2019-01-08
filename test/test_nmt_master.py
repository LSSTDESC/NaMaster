import unittest 
import numpy as np
import pymaster as nmt
import healpy as hp
import warnings
import sys
from .testutils import normdiff, read_flat_map

#Unit tests associated with the NmtField and NmtFieldFlat classes
class TestWorkspaceSph(unittest.TestCase) :
    def setUp(self) :
        #This is to avoid showing an ugly warning that has nothing to do with pymaster
        if (sys.version_info > (3, 1)):
            warnings.simplefilter("ignore", ResourceWarning)

        self.nside=64
        self.nlb=16
        self.lmax=3*self.nside-1
        self.npix=int(hp.nside2npix(self.nside))
        self.msk=hp.read_map("test/benchmarks/msk.fits",verbose=False)
        self.mps=np.array(hp.read_map("test/benchmarks/mps.fits",verbose=False,field=[0,1,2]))
        self.tmp=np.array(hp.read_map("test/benchmarks/tmp.fits",verbose=False,field=[0,1,2]))
        self.b=nmt.NmtBin(self.nside,nlb=self.nlb)
        self.f0=nmt.NmtField(self.msk,[self.mps[0]]) #Original nside
        self.f0_half=nmt.NmtField(self.msk[:self.npix//4],[self.mps[0,:self.npix//4]]) #Half nside
        self.b_half=nmt.NmtBin(self.nside//2,nlb=self.nlb) #Small-nside bandpowers
        self.b_doub=nmt.NmtBin(2*self.nside,nlb=self.nlb) #Large-nside bandposers
        self.n_good=np.zeros([1,3*self.nside])
        self.n_bad=np.zeros([2,3*self.nside])
        self.n_half=np.zeros([1,3*(self.nside//2)])
        
        l,cltt,clee,clbb,clte,nltt,nlee,nlbb,nlte=np.loadtxt("test/benchmarks/cls_lss.txt",unpack=True)
        self.l=l[:3*self.nside]
        self.cltt=cltt[:3*self.nside]
        self.clee=clee[:3*self.nside]
        self.clbb=clbb[:3*self.nside]
        self.clte=clte[:3*self.nside]
        self.nltt=nltt[:3*self.nside]
        self.nlee=nlee[:3*self.nside]
        self.nlbb=nlbb[:3*self.nside]
        self.nlte=nlte[:3*self.nside]
        
    def mastest(self,wtemp,wpure,do_teb=False) :
        prefix="test/benchmarks/bm"
        if wtemp :
            prefix+="_yc"
            f0=nmt.NmtField(self.msk,[self.mps[0]],templates=[[self.tmp[0]]])
            f2=nmt.NmtField(self.msk,[self.mps[1],self.mps[2]],
                            templates=[[self.tmp[1],self.tmp[2]]],
                            purify_b=wpure)
        else :
            prefix+="_nc"
            f0=nmt.NmtField(self.msk,[self.mps[0]])
            f2=nmt.NmtField(self.msk,[self.mps[1],self.mps[2]],purify_b=wpure)
        f=[f0,f2]
        
        if wpure :
            prefix+="_yp"
        else :
            prefix+="_np"

        for ip1 in range(2) :
            for ip2 in range(ip1,2) :
                if ip1==ip2==0 :
                    clth=np.array([self.cltt]);
                    nlth=np.array([self.nltt]);
                elif ip1==ip2==1 :
                    clth=np.array([self.clee,0*self.clee,0*self.clbb,self.clbb]);
                    nlth=np.array([self.nlee,0*self.nlee,0*self.nlbb,self.nlbb]);
                else :
                    clth=np.array([self.clte,0*self.clte]);
                    nlth=np.array([self.nlte,0*self.nlte]);
                w=nmt.NmtWorkspace()
                w.compute_coupling_matrix(f[ip1],f[ip2],self.b)
                clb=nlth
                if wtemp :
                    dlb=nmt.deprojection_bias(f[ip1],f[ip2],clth+nlth)
                    tlb=np.loadtxt(prefix+'_cb%d%d.txt'%(2*ip1,2*ip2),unpack=True)[1:,:]
                    self.assertTrue((np.fabs(dlb-tlb)<=np.fmin(np.fabs(dlb),np.fabs(tlb))*1E-5).all())
                    clb+=dlb
                cl=w.decouple_cell(nmt.compute_coupled_cell(f[ip1],f[ip2]),cl_bias=clb)
                tl=np.loadtxt(prefix+'_c%d%d.txt'%(2*ip1,2*ip2),unpack=True)[1:,:]
                self.assertTrue((np.fabs(cl-tl)<=np.fmin(np.fabs(cl),np.fabs(tl))*1E-5).all())

        #TEB
        if do_teb :
            clth=np.array([self.cltt,self.clte,0*self.clte,self.clee,0*self.clee,0*self.clbb,self.clbb])
            nlth=np.array([self.nltt,self.nlte,0*self.nlte,self.nlee,0*self.nlee,0*self.nlbb,self.nlbb])
            w=nmt.NmtWorkspace()
            w.compute_coupling_matrix(f[0],f[1],self.b,is_teb=True)
            c00=nmt.compute_coupled_cell(f[0],f[0])
            c02=nmt.compute_coupled_cell(f[0],f[1])
            c22=nmt.compute_coupled_cell(f[1],f[1])
            cl=np.array([c00[0],c02[0],c02[1],c22[0],c22[1],c22[2],c22[3]])
            t00=np.loadtxt(prefix+'_c00.txt',unpack=True)[1:,:]
            t02=np.loadtxt(prefix+'_c02.txt',unpack=True)[1:,:]
            t22=np.loadtxt(prefix+'_c22.txt',unpack=True)[1:,:]
            tl=np.array([t00[0],t02[0],t02[1],t22[0],t22[1],t22[2],t22[3]])
            cl=w.decouple_cell(cl,cl_bias=nlth)
            self.assertTrue((np.fabs(cl-tl)<=np.fmin(np.fabs(cl),np.fabs(tl))*1E-5).all())

    def test_workspace_master_teb_np(self) :
        self.mastest(False,False,do_teb=True)
        
    def test_workspace_master_teb_yp(self) :
        self.mastest(False,True,do_teb=True)
        
    def test_workspace_master_nc_np(self) :
        self.mastest(False,False)

    def test_workspace_master_yc_np(self) :
        self.mastest(True,False)

    def test_workspace_master_nc_yp(self) :
        self.mastest(False,True)

    def test_workspace_master_yc_yp(self) :
        self.mastest(True,True)

    def test_workspace_io(self) :
        w=nmt.NmtWorkspace()
        with self.assertRaises(RuntimeError) : #Invalid writing
            w.write_to("test/wspc.dat")
        w.read_from("test/benchmarks/bm_yc_yp_w02.dat") #OK read
        self.assertEqual(w.wsp.nside,64)
        mcm_old=w.get_coupling_matrix() #Read mode coupling matrix
        mcm_new=np.identity(3*w.wsp.nside*2) #Updating mode-coupling matrix
        w.update_coupling_matrix(mcm_new)
        mcm_back=w.get_coupling_matrix() #Retireve MCM and check it's correct
        self.assertTrue(np.fabs(np.sum(np.diagonal(mcm_back))-3*w.wsp.nside*2)<=1E-16)
        with self.assertRaises(RuntimeError) :  #Can't write on that file
            w.write_to("tests/wspc.dat")
        with self.assertRaises(RuntimeError) : #File doesn't exist
            w.read_from("none")

    def test_workspace_methods(self) :
        w=nmt.NmtWorkspace()
        w.compute_coupling_matrix(self.f0,self.f0,self.b) #OK init
        self.assertEqual(w.wsp.nside,64)
        with self.assertRaises(RuntimeError) : #Incompatible bandpowers
            w.compute_coupling_matrix(self.f0,self.f0,self.b_doub)
        with self.assertRaises(RuntimeError) : #Incompatible resolutions
            w.compute_coupling_matrix(self.f0,self.f0_half,self.b)
        with self.assertRaises(RuntimeError) : #Wrong fields for TEB
            w.compute_coupling_matrix(self.f0,self.f0,self.b,is_teb=True)

        w.compute_coupling_matrix(self.f0,self.f0,self.b)

        #Test couple_cell
        c=w.couple_cell(self.n_good)
        self.assertEqual(c.shape,(1,w.wsp.lmax+1))
        with self.assertRaises(ValueError) :
            c=w.couple_cell(self.n_bad)
        with self.assertRaises(ValueError) :
            c=w.couple_cell(self.n_half)

        #Test decouple_cell
        c=w.decouple_cell(self.n_good)
        self.assertEqual(c.shape,(1,self.b.bin.n_bands))
        with self.assertRaises(ValueError) :
            c=w.decouple_cell(self.n_bad)
        with self.assertRaises(ValueError) :
            c=w.decouple_cell(self.n_half)
        c=w.decouple_cell(self.n_good,cl_bias=self.n_good,cl_noise=self.n_good)
        self.assertEqual(c.shape,(1,self.b.bin.n_bands))
        with self.assertRaises(ValueError) :
            c=w.decouple_cell(self.n_good,cl_bias=self.n_good,cl_noise=self.n_bad)
        with self.assertRaises(ValueError) :
            c=w.decouple_cell(self.n_good,cl_bias=self.n_good,cl_noise=self.n_half)
        with self.assertRaises(ValueError) :
            c=w.decouple_cell(self.n_good,cl_bias=self.n_bad,cl_noise=self.n_good)
        with self.assertRaises(ValueError) :
            c=w.decouple_cell(self.n_good,cl_bias=self.n_half,cl_noise=self.n_good)

    def test_workspace_full_master(self) :
        #Test compute_full_master
        w=nmt.NmtWorkspace()
        w.compute_coupling_matrix(self.f0,self.f0,self.b) #OK init
        w_half=nmt.NmtWorkspace();
        w_half.compute_coupling_matrix(self.f0_half,self.f0_half,self.b_half)

        c=nmt.compute_full_master(self.f0,self.f0,self.b)
        self.assertEqual(c.shape,(1,self.b.bin.n_bands))
        with self.assertRaises(RuntimeError) : #Incompatible bandpowers
            c=nmt.compute_full_master(self.f0,self.f0,self.b_doub)
        with self.assertRaises(ValueError) : #Incompatible resolutions
            c=nmt.compute_full_master(self.f0,self.f0_half,self.b)
        #Passing correct input workspace
        w.compute_coupling_matrix(self.f0,self.f0,self.b)
        c=nmt.compute_full_master(self.f0,self.f0,self.b,workspace=w) #Computing from correct workspace
        self.assertEqual(c.shape,(1,self.b.bin.n_bands))
        with self.assertRaises(RuntimeError) : #Inconsistent workspace
            c=nmt.compute_full_master(self.f0_half,self.f0_half,self.b_half,workspace=w)
        #Incorrect input spectra
        with self.assertRaises(ValueError) :
            c=nmt.compute_full_master(self.f0,self.f0,self.b,cl_noise=self.n_bad)
        with self.assertRaises(ValueError) :
            c=nmt.compute_full_master(self.f0,self.f0,self.b,cl_guess=self.n_bad)
        with self.assertRaises(RuntimeError) :
            c=nmt.compute_full_master(self.f0,self.f0,self.b,cl_noise=self.n_half)
        with self.assertRaises(RuntimeError) :
            c=nmt.compute_full_master(self.f0,self.f0,self.b,cl_guess=self.n_half)

    def test_workspace_deprojection_bias(self) :
        #Test deprojection_bias
        c=nmt.deprojection_bias(self.f0,self.f0,self.n_good)
        self.assertEqual(c.shape,(1,self.f0.fl.lmax+1)) 
        with self.assertRaises(ValueError) :
            c=nmt.deprojection_bias(self.f0,self.f0,self.n_bad)
        with self.assertRaises(ValueError) :
            c=nmt.deprojection_bias(self.f0,self.f0,self.n_half)
        with self.assertRaises(RuntimeError) :
            c=nmt.deprojection_bias(self.f0,self.f0_half,self.n_good)

    def test_workspace_uncorr_noise_deprojection_bias(self) :
        #Test uncorr_noise_deprojection_bias
        c=nmt.uncorr_noise_deprojection_bias(self.f0,np.zeros(self.npix))
        self.assertEqual(c.shape,(1,self.f0.fl.lmax+1)) 
        with self.assertRaises(ValueError) :
            c=nmt.uncorr_noise_deprojection_bias(self.f0,self.n_good)

    def test_workspace_compute_coupled_cell(self) :
        #Test compute_coupled_cell
        c=nmt.compute_coupled_cell(self.f0,self.f0)
        self.assertEqual(c.shape,(1,self.f0.fl.lmax+1)) 
        with self.assertRaises(ValueError) : #Different resolutions
            c=nmt.compute_coupled_cell(self.f0,self.f0_half)

class TestWorkspaceFsk(unittest.TestCase) :
    def setUp(self) :
        #This is to avoid showing an ugly warning that has nothing to do with pymaster
        if (sys.version_info > (3, 1)):
            warnings.simplefilter("ignore", ResourceWarning)

        self.wcs,self.msk=read_flat_map("test/benchmarks/msk_flat.fits")
        (self.ny,self.nx)=self.msk.shape
        self.lx=np.radians(np.fabs(self.nx*self.wcs.wcs.cdelt[0]))
        self.ly=np.radians(np.fabs(self.ny*self.wcs.wcs.cdelt[1]))
        self.mps=np.array([read_flat_map("test/benchmarks/mps_flat.fits",i_map=i)[1] for i in range(3)])
        self.tmp=np.array([read_flat_map("test/benchmarks/tmp_flat.fits",i_map=i)[1] for i in range(3)])
        self.d_ell=20;
        self.lmax=500.;
        ledges=np.arange(int(self.lmax/self.d_ell)+1)*self.d_ell+2
        self.b=nmt.NmtBinFlat(ledges[:-1],ledges[1:])
        self.leff=self.b.get_effective_ells();
        self.f0=nmt.NmtFieldFlat(self.lx,self.ly,self.msk,[self.mps[0]])
        self.f0_half=nmt.NmtFieldFlat(self.lx,self.ly,self.msk[:self.ny//2,:self.nx//2],
                                      [self.mps[0,:self.ny//2,:self.nx//2]])
        ledges_half=ledges[:len(ledges)//2]
        self.b_half=nmt.NmtBinFlat(ledges_half[:-1],ledges_half[1:])

        l,cltt,clee,clbb,clte,nltt,nlee,nlbb,nlte=np.loadtxt("test/benchmarks/cls_lss.txt",unpack=True)
        self.l=l[:]
        self.cltt=cltt[:]
        self.clee=clee[:]
        self.clbb=clbb[:]
        self.clte=clte[:]
        self.nltt=nltt[:]
        self.nlee=nlee[:]
        self.nlbb=nlbb[:]
        self.nlte=nlte[:]
        self.n_good=np.zeros([1,len(l)])
        self.n_bad=np.zeros([2,len(l)])
        self.nb_good=np.zeros([1,self.b.bin.n_bands])
        self.nb_bad=np.zeros([2,self.b.bin.n_bands])
        
        
    def mastest(self,wtemp,wpure,do_teb=False) :
        prefix="test/benchmarks/bm_f"
        if wtemp :
            prefix+="_yc"
            f0=nmt.NmtFieldFlat(self.lx,self.ly,self.msk,[self.mps[0]],templates=[[self.tmp[0]]])
            f2=nmt.NmtFieldFlat(self.lx,self.ly,self.msk,[self.mps[1],self.mps[2]],
                                templates=[[self.tmp[1],self.tmp[2]]],
                                purify_b=wpure)
        else :
            prefix+="_nc"
            f0=nmt.NmtFieldFlat(self.lx,self.ly,self.msk,[self.mps[0]])
            f2=nmt.NmtFieldFlat(self.lx,self.ly,self.msk,[self.mps[1],self.mps[2]],purify_b=wpure)
        f=[f0,f2]
        
        if wpure :
            prefix+="_yp"
        else :
            prefix+="_np"

        for ip1 in range(2) :
            for ip2 in range(ip1,2) :
                if ip1==ip2==0 :
                    clth=np.array([self.cltt]);
                    nlth=np.array([self.nltt]);
                elif ip1==ip2==1 :
                    clth=np.array([self.clee,0*self.clee,0*self.clbb,self.clbb]);
                    nlth=np.array([self.nlee,0*self.nlee,0*self.nlbb,self.nlbb]);
                else :
                    clth=np.array([self.clte,0*self.clte]);
                    nlth=np.array([self.nlte,0*self.nlte]);
                w=nmt.NmtWorkspaceFlat()
                w.compute_coupling_matrix(f[ip1],f[ip2],self.b)
                clb=w.couple_cell(self.l,nlth)
                if wtemp :
                    dlb=nmt.deprojection_bias_flat(f[ip1],f[ip2],self.b,self.l,clth+nlth)
                    tlb=np.loadtxt(prefix+'_cb%d%d.txt'%(2*ip1,2*ip2),unpack=True)[1:,:]
                    self.assertTrue((np.fabs(dlb-tlb)<=np.fmin(np.fabs(dlb),np.fabs(tlb))*1E-5).all())
                    clb+=dlb
                cl=w.decouple_cell(nmt.compute_coupled_cell_flat(f[ip1],f[ip2],self.b),cl_bias=clb)
                tl=np.loadtxt(prefix+'_c%d%d.txt'%(2*ip1,2*ip2),unpack=True)[1:,:]
                self.assertTrue((np.fabs(cl-tl)<=np.fmin(np.fabs(cl),np.fabs(tl))*1E-5).all())

        #TEB
        if do_teb :
            clth=np.array([self.cltt,self.clte,0*self.clte,self.clee,0*self.clee,0*self.clbb,self.clbb])
            nlth=np.array([self.nltt,self.nlte,0*self.nlte,self.nlee,0*self.nlee,0*self.nlbb,self.nlbb])
            w=nmt.NmtWorkspaceFlat()
            w.compute_coupling_matrix(f[0],f[1],self.b,is_teb=True)
            c00=nmt.compute_coupled_cell_flat(f[0],f[0],self.b)
            c02=nmt.compute_coupled_cell_flat(f[0],f[1],self.b)
            c22=nmt.compute_coupled_cell_flat(f[1],f[1],self.b)
            cl=np.array([c00[0],c02[0],c02[1],c22[0],c22[1],c22[2],c22[3]])
            t00=np.loadtxt(prefix+'_c00.txt',unpack=True)[1:,:]
            t02=np.loadtxt(prefix+'_c02.txt',unpack=True)[1:,:]
            t22=np.loadtxt(prefix+'_c22.txt',unpack=True)[1:,:]
            tl=np.array([t00[0],t02[0],t02[1],t22[0],t22[1],t22[2],t22[3]])
            cl=w.decouple_cell(cl,cl_bias=w.couple_cell(self.l,nlth))
            self.assertTrue((np.fabs(cl-tl)<=np.fmin(np.fabs(cl),np.fabs(tl))*1E-5).all())

    def test_workspace_flat_master_teb_np(self) :
        self.mastest(False,False,do_teb=True)
        
    def test_workspace_flat_master_teb_yp(self) :
        self.mastest(False,True,do_teb=True)
        
    def test_workspace_flat_master_nc_np(self) :
        self.mastest(False,False)

    def test_workspace_flat_master_nc_yp(self) :
        self.mastest(False,True)

    def test_workspace_flat_master_yc_np(self) :
        self.mastest(True,False)

    def test_workspace_flat_master_yc_yp(self) :
        self.mastest(True,True)

    def test_workspace_flat_io(self) :
        w=nmt.NmtWorkspaceFlat()
        with self.assertRaises(RuntimeError) : #Invalid writing
            w.write_to("test/wspc.dat")
        w.read_from("test/benchmarks/bm_f_yc_yp_w02.dat") #OK read
        self.assertEqual(self.msk.shape,(w.wsp.fs.ny,w.wsp.fs.nx))
        with self.assertRaises(RuntimeError) :  #Can't write on that file
            w.write_to("tests/wspc.dat")
        with self.assertRaises(RuntimeError) : #File doesn't exist
            w.read_from("none")

    def test_workspace_flat_methods(self) :
        w=nmt.NmtWorkspaceFlat()
        w.compute_coupling_matrix(self.f0,self.f0,self.b) #OK init
        self.assertEqual(self.msk.shape,(w.wsp.fs.ny,w.wsp.fs.nx))
        with self.assertRaises(RuntimeError) : #Incompatible resolutions
            w.compute_coupling_matrix(self.f0,self.f0_half,self.b)
        with self.assertRaises(RuntimeError) : #Wrong fields for TEB
            w.compute_coupling_matrix(self.f0,self.f0,self.b,is_teb=True)

        w.compute_coupling_matrix(self.f0,self.f0,self.b)

        #Test couple_cell
        c=w.couple_cell(self.l,self.n_good)
        self.assertEqual(c.shape,(1,self.b.bin.n_bands))
        with self.assertRaises(ValueError) :
            c=w.couple_cell(self.l,self.n_bad)
        with self.assertRaises(ValueError) :
            c=w.couple_cell(self.l,self.n_good[:,:len(self.l)//2])

        #Test decouple_cell
        c=w.decouple_cell(self.nb_good,self.nb_good,self.nb_good)
        self.assertEqual(c.shape,(1,self.b.bin.n_bands))
        with self.assertRaises(ValueError) :
            c=w.decouple_cell(self.nb_bad)
        with self.assertRaises(ValueError) :
            c=w.decouple_cell(self.n_good)
        with self.assertRaises(ValueError) :
            c=w.decouple_cell(self.nb_good,cl_bias=self.nb_bad)
        with self.assertRaises(ValueError) :
            c=w.decouple_cell(self.nb_good,cl_bias=self.n_good)
        with self.assertRaises(ValueError) :
            c=w.decouple_cell(self.nb_good,cl_noise=self.nb_bad)
        with self.assertRaises(ValueError) :
            c=w.decouple_cell(self.nb_good,cl_noise=self.n_good)

    def test_workspace_flat_full_master(self) :
        w=nmt.NmtWorkspaceFlat()
        w.compute_coupling_matrix(self.f0,self.f0,self.b)
        w_half=nmt.NmtWorkspaceFlat()
        w_half.compute_coupling_matrix(self.f0_half,self.f0_half,self.b_half)

        c=nmt.compute_full_master_flat(self.f0,self.f0,self.b)
        self.assertEqual(c.shape,(1,self.b.bin.n_bands))
        with self.assertRaises(ValueError) : #Incompatible resolutions
            c=nmt.compute_full_master_flat(self.f0,self.f0_half,self.b)
        c=nmt.compute_full_master_flat(self.f0,self.f0,self.b_half)
        #Passing correct input workspace
        c=nmt.compute_full_master_flat(self.f0,self.f0,self.b,workspace=w) #Computing from correct wsp
        self.assertEqual(c.shape,(1,self.b.bin.n_bands))
        with self.assertRaises(RuntimeError) : #Incompatible bandpowers
            c=nmt.compute_full_master_flat(self.f0,self.f0,self.b_half,workspace=w)
        with self.assertRaises(RuntimeError) : #Incompatible workspace
            c=nmt.compute_full_master_flat(self.f0,self.f0,self.b,workspace=w_half)
        #Incorrect input spectra
        with self.assertRaises(ValueError) :
            c=nmt.compute_full_master_flat(self.f0,self.f0,self.b,cl_noise=self.nb_bad,workspace=w)
        with self.assertRaises(ValueError) :
            c=nmt.compute_full_master_flat(self.f0,self.f0,self.b,cl_noise=self.n_good,workspace=w)
        with self.assertRaises(ValueError) : #Non ell-values
            c=nmt.compute_full_master_flat(self.f0,self.f0,self.b,cl_noise=self.nb_good,
                                           cl_guess=self.n_good,workspace=w)
        with self.assertRaises(ValueError) : #Wrong cl_guess
            c=nmt.compute_full_master_flat(self.f0,self.f0,self.b,cl_noise=self.nb_good,
                                           cl_guess=self.n_bad,ells_guess=self.l,workspace=w)
        with self.assertRaises(ValueError) : #Wrong cl_guess
            c=nmt.compute_full_master_flat(self.f0,self.f0,self.b,cl_noise=self.nb_good,
                                           cl_guess=self.nb_good,ells_guess=self.l,workspace=w)

    def test_workspace_flat_deprojection_bias(self) :
        #Test derojection_bias_flat
        c=nmt.deprojection_bias_flat(self.f0,self.f0,self.b,self.l,self.n_good)
        self.assertEqual(c.shape,(1,self.b.bin.n_bands))
        with self.assertRaises(ValueError) : #Wrong cl_guess
            c=nmt.deprojection_bias_flat(self.f0,self.f0,self.b,self.l,self.n_bad)
        with self.assertRaises(ValueError) : #Wrong cl_guess
            c=nmt.deprojection_bias_flat(self.f0,self.f0,self.b,self.l,self.nb_good)
        with self.assertRaises(RuntimeError) : #Incompatible resolutions
            c=nmt.deprojection_bias_flat(self.f0,self.f0_half,self.b,self.l,self.n_good)

    def test_workspace_flat_compute_coupled_cell(self) :
        #Test compute_coupled_cell_flat
        c=nmt.compute_coupled_cell_flat(self.f0,self.f0,self.b)
        self.assertEqual(c.shape,(1,self.b.bin.n_bands))
        with self.assertRaises(ValueError) : #Different resolutions
            c=nmt.compute_coupled_cell_flat(self.f0,self.f0_half,self.b)

        
if __name__ == '__main__':
    unittest.main()
