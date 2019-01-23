import unittest 
import numpy as np
import pymaster as nmt
import healpy as hp
from .testutils import normdiff

#Unit tests associated with the NmtField and NmtFieldFlat classes

class TestFieldCAR(unittest.TestCase) :
    def setUp(self) :
        from astropy.io import fits
        from astropy.wcs import WCS

        hdul=fits.open("test/benchmarks/mps_car.fits")
        self.wcs=WCS(hdul[0].header)
        self.ny,self.nx=hdul[0].data.shape
        hdul.close()

        self.wt=nmt.NmtWCSTranslator(self.wcs,(self.ny,self.nx))
        self.lmax=nmt.lib.get_lmax_py(self.wt.is_healpix,self.wt.nside,self.wt.nx,self.wt.ny,
                                      self.wt.d_phi,self.wt.d_theta,self.wt.phi0,self.wt.theta_max)
        self.ntemp=5
        self.npix=self.ny*self.nx
        self.msk=np.ones([self.ny,self.nx])
        self.mps=np.ones([3,self.ny,self.nx])
        self.tmp=np.ones([self.ntemp,3,self.ny,self.nx])
        self.beam=np.ones(self.lmax+1)
        ph,th=self.wcs.wcs_pix2world(np.array(np.meshgrid(np.arange(self.nx),np.arange(self.ny))).reshape([2,self.ny*self.nx]).T,0).T.reshape([2,self.ny,self.nx])
        ph=np.radians(ph)
        th=np.radians(90-th)
        sth=np.sin(th); cth=np.cos(th)
        self.mps[0,:,:]=np.sqrt(15./2./np.pi)*sth**2*np.cos(2*ph) #Re(Y_22)
        self.mps[1,:,:]=-np.sqrt(15./2./np.pi)*sth**2/4.          #_2Y^E_20 + _2Y^B_30
        self.mps[2,:,:]=-np.sqrt(105./2./np.pi)*cth*sth**2/2.
        for i in range(self.ntemp) :
            self.tmp[i][0,:,:]=np.sqrt(15./2./np.pi)*sth**2*np.cos(2*ph) #Re(Y_22)
            self.tmp[i][1,:,:]=-np.sqrt(15./2./np.pi)*sth**2/4.          #_2Y^E_20 + _2Y^B_30
            self.tmp[i][2,:,:]=-np.sqrt(105./2./np.pi)*cth*sth**2/2.

    def test_field_alloc(self) :
        #No templates
        f0=nmt.NmtField(self.msk,[self.mps[0]],beam=self.beam,wcs=self.wcs)
        f2=nmt.NmtField(self.msk,[self.mps[1],self.mps[2]],beam=self.beam,wcs=self.wcs)
        f2p=nmt.NmtField(self.msk,[self.mps[1],self.mps[2]],beam=self.beam,wcs=self.wcs,
                         purify_e=True,purify_b=True,n_iter_mask_purify=10)
        self.assertTrue(normdiff(f0.get_maps()[0],(self.mps[0]*self.msk).flatten())<1E-10)
        self.assertTrue(normdiff(f2.get_maps()[0],(self.mps[1]*self.msk).flatten())<1E-10)
        self.assertTrue(normdiff(f2.get_maps()[1],(self.mps[2]*self.msk).flatten())<1E-10)
        self.assertTrue(1E-5*np.mean(np.fabs(f2p.get_maps()[0]))>np.mean(np.fabs(f2p.get_maps()[0]-
                                                                                 (self.mps[1]*self.msk).flatten())))
        self.assertTrue(1E-5*np.mean(np.fabs(f2p.get_maps()[1]))>np.mean(np.fabs(f2p.get_maps()[1]-
                                                                                 (self.mps[2]*self.msk).flatten())))
        self.assertEqual(len(f0.get_templates()),0)
        self.assertEqual(len(f2.get_templates()),0)
        self.assertEqual(len(f2p.get_templates()),0)

        #With templates
        f0=nmt.NmtField(self.msk,[self.mps[0]],
                        templates=np.array([[t[0]] for t in self.tmp]),
                        beam=self.beam,wcs=self.wcs)
        f2=nmt.NmtField(self.msk,[self.mps[1],self.mps[2]],
                        templates=np.array([[t[1],t[2]] for t in self.tmp]),
                        beam=self.beam,wcs=self.wcs)
        #Map should be zero, since template =  map
        self.assertTrue(normdiff(f0.get_maps()[0],0*self.msk.flatten())<1E-10)
        self.assertTrue(normdiff(f2.get_maps()[0],0*self.msk.flatten())<1E-10)
        self.assertTrue(normdiff(f2.get_maps()[1],0*self.msk.flatten())<1E-10)
        self.assertEqual(len(f0.get_templates()),5)
        self.assertEqual(len(f2.get_templates()),5)

    def test_field_error(self) :
        with self.assertRaises(ValueError) : #Not passing WCS
            f=nmt.NmtField(self.msk,[self.mps[0]],beam=self.beam)
        with self.assertRaises(ValueError) : #Passing 1D maps
            f=nmt.NmtField(self.msk.flatten(),[self.mps[0]],beam=self.beam,wcs=self.wcs)
        with self.assertRaises(AttributeError) : #Passing incorrect WCS
            f=nmt.NmtField(self.msk,[self.mps[0]],beam=self.beam,wcs=1)
        #Incorrect sky projection
        wcs=self.wcs.deepcopy(); wcs.wcs.ctype[0]='RA---TAN'
        with self.assertRaises(ValueError) :
            f=nmt.NmtField(self.msk,[self.mps[0]],beam=self.beam,wcs=wcs)
        #Incorrect reference pixel coords
        wcs=self.wcs.deepcopy(); wcs.wcs.crval[1]=96.
        with self.assertRaises(ValueError) :
            f=nmt.NmtField(self.msk,[self.mps[0]],beam=self.beam,wcs=wcs)
        #Incorrect pixel sizes
        wcs=self.wcs.deepcopy(); wcs.wcs.cdelt[1]=1.01
        with self.assertRaises(ValueError) :
            f=nmt.NmtField(self.msk,[self.mps[0]],beam=self.beam,wcs=wcs)
        #Input maps are too big
        msk=np.ones([self.ny,self.nx+10])
        with self.assertRaises(ValueError) :
            f=nmt.NmtField(msk,[self.mps[0]],beam=self.beam,wcs=self.wcs)
        with self.assertRaises(ValueError) :
            msk=np.ones([self.ny+100,self.nx])
            f=nmt.NmtField(msk,[msk],beam=self.beam,wcs=self.wcs)
        #Reference pixel has wrong pixel coordinates
        wcs=self.wcs.deepcopy(); wcs.wcs.crpix[0]=1.; wcs.wcs.cdelt[0]=-1.
        with self.assertRaises(ValueError) :
            f=nmt.NmtField(self.msk,[self.mps[0]],beam=self.beam,wcs=wcs)
        with self.assertRaises(ValueError) : #Incorrect mask size
            f=nmt.NmtField(self.msk[:90],[self.mps[0]],beam=self.beam,wcs=self.wcs)
        with self.assertRaises(ValueError) : #Incorrect maps size
            f=nmt.NmtField(self.msk,[self.mps[0,:90]],beam=self.beam,wcs=self.wcs)
        with self.assertRaises(ValueError) : #Too many maps
            f=nmt.NmtField(self.msk,self.mps,wcs=self.wcs)
        with self.assertRaises(ValueError) : #Too many maps per template
            f=nmt.NmtField(self.msk,[self.mps[0]],templates=self.tmp,beam=self.beam,wcs=self.wcs)
        with self.assertRaises(ValueError) : #Number of maps per template does not match spin
            f=nmt.NmtField(self.msk,[self.mps[0]],templates=[[t[0],t[1]] for t in self.tmp],
                           beam=self.beam,wcs=self.wcs)
        with self.assertRaises(ValueError) : #Incorrect template size
            f=nmt.NmtField(self.msk,[self.mps[0]],templates=[[t[0,:90]] for t in self.tmp],
                           beam=self.beam,wcs=self.wcs)
        with self.assertRaises(ValueError) : #Passing crap as templates
            f=nmt.NmtField(self.msk,[self.mps[0]],templates=1,beam=self.beam,wcs=self.wcs)
        with self.assertRaises(ValueError) : #Passing wrong beam
            f=nmt.NmtField(self.msk,[self.mps[0]],templates=[[t[0]] for t in self.tmp],
                           beam=self.beam[:90],wcs=self.wcs)
        with self.assertRaises(ValueError) : #Passing crap as beam
            f=nmt.NmtField(self.msk,[self.mps[0]],templates=[[t[0]] for t in self.tmp],
                           beam=1,wcs=self.wcs)
        
class TestFieldHPX(unittest.TestCase) :
    def setUp(self) :
        self.nside=64
        self.lmax=3*self.nside-1
        self.ntemp=5
        self.npix=int(hp.nside2npix(self.nside))
        self.msk=np.ones(self.npix)
        self.mps=np.zeros([3,self.npix])
        self.tmp=np.zeros([self.ntemp,3,self.npix])
        self.beam=np.ones(self.lmax+1)

        th,ph=hp.pix2ang(self.nside,np.arange(self.npix))
        sth=np.sin(th); cth=np.cos(th)
        self.mps[0]=np.sqrt(15./2./np.pi)*sth**2*np.cos(2*ph) #Re(Y_22)
        self.mps[1]=-np.sqrt(15./2./np.pi)*sth**2/4.          #_2Y^E_20 + _2Y^B_30
        self.mps[2]=-np.sqrt(105./2./np.pi)*cth*sth**2/2.
        for i in range(self.ntemp) :
            self.tmp[i][0]=np.sqrt(15./2./np.pi)*sth**2*np.cos(2*ph) #Re(Y_22)
            self.tmp[i][1]=-np.sqrt(15./2./np.pi)*sth**2/4.          #_2Y^E_20 + _2Y^B_30
            self.tmp[i][2]=-np.sqrt(105./2./np.pi)*cth*sth**2/2.

    def test_field_alloc(self) :
        #No templates
        f0=nmt.NmtField(self.msk,[self.mps[0]],beam=self.beam)
        f2=nmt.NmtField(self.msk,[self.mps[1],self.mps[2]],beam=self.beam)
        f2p=nmt.NmtField(self.msk,[self.mps[1],self.mps[2]],beam=self.beam,
                         purify_e=True,purify_b=True,n_iter_mask_purify=10)
        self.assertTrue(normdiff(f0.get_maps()[0],self.mps[0]*self.msk)<1E-10)
        self.assertTrue(normdiff(f2.get_maps()[0],self.mps[1]*self.msk)<1E-10)
        self.assertTrue(normdiff(f2.get_maps()[1],self.mps[2]*self.msk)<1E-10)
        self.assertTrue(1E-5*np.mean(np.fabs(f2p.get_maps()[0]))>np.mean(np.fabs(f2p.get_maps()[0]-
                                                                                 self.mps[1]*self.msk)))
        self.assertTrue(1E-5*np.mean(np.fabs(f2p.get_maps()[1]))>np.mean(np.fabs(f2p.get_maps()[1]-
                                                                                 self.mps[2]*self.msk)))
        self.assertEqual(len(f0.get_templates()),0)
        self.assertEqual(len(f2.get_templates()),0)
        self.assertEqual(len(f2p.get_templates()),0)

        #With templates
        f0=nmt.NmtField(self.msk,[self.mps[0]],
                        templates=np.array([[t[0]] for t in self.tmp]),
                        beam=self.beam)
        f2=nmt.NmtField(self.msk,[self.mps[1],self.mps[2]],
                        templates=np.array([[t[1],t[2]] for t in self.tmp]),
                        beam=self.beam)
        #Map should be zero, since template =  map
        self.assertTrue(normdiff(f0.get_maps()[0],0*self.msk)<1E-10)
        self.assertTrue(normdiff(f2.get_maps()[0],0*self.msk)<1E-10)
        self.assertTrue(normdiff(f2.get_maps()[1],0*self.msk)<1E-10)
        self.assertEqual(len(f0.get_templates()),5)
        self.assertEqual(len(f2.get_templates()),5)

    def test_field_error(self) :
        with self.assertRaises(ValueError) : #Incorrect mask size
            f=nmt.NmtField(self.msk[:15],self.mps)
        with self.assertRaises(ValueError) : #Incorrect map size
            f=nmt.NmtField(self.msk,[self.mps[0,:15]])
        with self.assertRaises(ValueError) : #Incorrect template size
            f=nmt.NmtField(self.msk,[self.mps[0]],templates=[[self.tmp[0,0,:15]]])
        with self.assertRaises(ValueError) : #Passing 3 maps!
            f=nmt.NmtField(self.msk,self.mps)
        with self.assertRaises(ValueError) : #Passing 3 template maps!
            f=nmt.NmtField(self.msk,[self.mps[1],self.mps[2]],templates=self.tmp)
        with self.assertRaises(ValueError) : #Passing crap as templates
            f=nmt.NmtField(self.msk,[self.mps[1],self.mps[2]],templates=1)
        with self.assertRaises(ValueError) : #Passing wrong beam
            f=nmt.NmtField(self.msk,[self.mps[0]],beam=self.beam[:30])
        with self.assertRaises(ValueError) : #Passing crap as beam
            f=nmt.NmtField(self.msk,[self.mps[0]],beam=1)

class TestFieldFsk(unittest.TestCase) :
    def setUp(self) :
        self.ntemp=5
        self.nx=141
        self.ny=311
        self.lx=np.radians(1.)
        self.ly=np.radians(1.)
        self.npix=self.nx*self.ny
        self.msk=np.ones([self.ny,self.nx])
        self.lmax=np.sqrt((self.nx*np.pi/self.lx)**2+(self.ny*np.pi/self.ly)**2)
        self.nell=30
        xarr=np.arange(self.nx)*self.lx/self.nx
        yarr=np.arange(self.ny)*self.ly/self.ny
        i0_x=2; i0_y=3;
        k0_x=i0_x*2*np.pi/self.lx
        k0_y=i0_y*2*np.pi/self.ly
        phase=k0_x*xarr[None,:]+k0_y*yarr[:,None]
        cphi0=k0_x/np.sqrt(k0_x**2+k0_y**2)
        sphi0=k0_y/np.sqrt(k0_x**2+k0_y**2)
        c2phi0=cphi0**2-sphi0**2
        s2phi0=2*sphi0*cphi0
        self.mps=np.array([2*np.pi*np.cos(phase)/(self.lx*self.ly),
                           2*np.pi*c2phi0*np.cos(phase)/(self.lx*self.ly),
                           -2*np.pi*s2phi0*np.cos(phase)/(self.lx*self.ly)])
        self.tmp=np.array([self.mps.copy() for i in range(self.ntemp)])
        self.beam=np.array([np.arange(self.nell)*self.lmax/(self.nell-1.),np.ones(self.nell)])

    def test_field_flat_alloc(self) :
        #No templates
        f0=nmt.NmtFieldFlat(self.lx,self.ly,self.msk,[self.mps[0]],beam=self.beam)
        f2=nmt.NmtFieldFlat(self.lx,self.ly,self.msk,[self.mps[1],self.mps[2]],beam=self.beam)
        f2p=nmt.NmtFieldFlat(self.lx,self.ly,self.msk,[self.mps[1],self.mps[2]],beam=self.beam,
                             purify_e=True,purify_b=True)
        self.assertTrue(normdiff(f0.get_maps()[0],self.mps[0]*self.msk)<1E-10)
        self.assertTrue(normdiff(f2.get_maps()[0],self.mps[1]*self.msk)<1E-10)
        self.assertTrue(normdiff(f2.get_maps()[1],self.mps[2]*self.msk)<1E-10)
        self.assertTrue(normdiff(f2p.get_maps()[0],self.mps[1]*self.msk)<1E-10)
        self.assertTrue(normdiff(f2p.get_maps()[1],self.mps[2]*self.msk)<1E-10)
        self.assertEqual(len(f0.get_templates()),0)
        self.assertEqual(len(f2.get_templates()),0)
        self.assertEqual(len(f2p.get_templates()),0)

        #With templates
        f0=nmt.NmtFieldFlat(self.lx,self.ly,self.msk,[self.mps[0]],beam=self.beam,
                            templates=np.array([[t[0]] for t in self.tmp]))
        f2=nmt.NmtFieldFlat(self.lx,self.ly,self.msk,[self.mps[1],self.mps[2]],beam=self.beam,
                            templates=np.array([[t[1],t[2]] for t in self.tmp]))
        #Map should be zero, since template =  map
        self.assertTrue(normdiff(f0.get_maps()[0],0*self.msk)<1E-10)
        self.assertTrue(normdiff(f2.get_maps()[0],0*self.msk)<1E-10)
        self.assertTrue(normdiff(f2.get_maps()[1],0*self.msk)<1E-10)
        self.assertEqual(len(f0.get_templates()),5)
        self.assertEqual(len(f2.get_templates()),5)

    def test_field_flat_errors(self) :
        with self.assertRaises(ValueError) : #Incorrect map sizes
            f=nmt.NmtFieldFlat(self.lx,-self.ly,self.msk,[self.mps[0]])
        with self.assertRaises(ValueError) :
            f=nmt.NmtFieldFlat(-self.lx,self.ly,self.msk,[self.mps[0]])
        with self.assertRaises(ValueError) : #Mismatching map dimensions
            f=nmt.NmtFieldFlat(self.lx,self.ly,self.msk,[self.mps[0,:self.ny//2]])
        with self.assertRaises(ValueError) : #Mismatching template dimensions
            f=nmt.NmtFieldFlat(self.lx,self.ly,self.msk,[self.mps[0]],
                               templates=np.array([[t[0,:self.ny//2]] for t in self.tmp]))
        with self.assertRaises(ValueError) : #Passing 3 templates!
            f=nmt.NmtFieldFlat(self.lx,self.ly,self.msk,self.mps)
        with self.assertRaises(ValueError) : #Passing 3 templates!
            f=nmt.NmtFieldFlat(self.lx,self.ly,self.msk,[self.mps[0]],templates=self.tmp)
        with self.assertRaises(ValueError) : #Passing crap templates
            f=nmt.NmtFieldFlat(self.lx,self.ly,self.msk,[self.mps[0]],
                               templates=1)
        with self.assertRaises(ValueError) : #Passing crap beam
            f=nmt.NmtFieldFlat(self.lx,self.ly,self.msk,[self.mps[0]],beam=1)
            
if __name__ == '__main__':
    unittest.main()
