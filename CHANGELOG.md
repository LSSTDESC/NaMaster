# v2.5
- Implementation of mode deprojection for catalog fields complete (#236)

# v2.4
- Mode deprojection enabled for `NmtFieldCatalogClustering` fields (#227)

# v2.3.1
- Fixed docstrings (#224)

# v2.3.2
- Fixed azimuth wrapping issue in CAR (#222)

# v2.3.1
- Fixed API-breaking change in ducc0 (#220)

# v2.3
- Support for anisotropically-weighted fields (#217)

# v2.2.3
- Bugfix: saving `norm_type` in workspace fits files (#212)

# v2.2.2
- Improved bug fix (#204)

# v2.2.1
- Fixed small bug due to floating point error in CAR initialisation (#204)

# v2.2
- More convenient constructors for workspaces and FKP normalisation (#201)

# v2.1
- Catalog-based pseudo-C_ells (#184)
- Tutorial notebooks and improved documentation (#192, #195, #196, #197)
- Unit tests for mac-OS (#194)

# v2.0
- Improved error messages (#187)
- Migrated to ducc from libsharp (#183)

# v1.6
- Bugfix: sign of Wigner 3j symbols (#180)
- Faster method for spin-0 Wigner 3j symbols (#178)
- No deprojection bias for lite fields (#177)
- Bugfix for rectangular covariances (#175)

# v1.5.1
- Bugfix in Toeplitz calculation (#169)
- Bugfix for rectangular covariances (#175)

# v1.5
- Compilation refactor (#158, #162)
- Updated installation instructions for NERSC (#160)
- Ensure correct mask endianness (#161)

# v1.4
- Option to read only the binned MCM (#151)

# v1.3
- Spin-0-only covariance workspaces (#147)
- Added functionality to return field mask (#142)
- Fixed a couple of bugs (#147, #144, #140)

# v1.2
- Uses most recent version of libsharp (libsharp2) (#121)
- Various bug fixes (#123, #124, #126)
- Added functionality to return the field's alms (#131)
- Migrated tests to pytest and github actions (#132)

# v1.1
- Faster MCM via symmetry (thanks to Thibaut Louis) (#111)
- Covariance calculable for coupled spectra (#113)
- Arbitrary spins (#115)
- Additional binning functionality (thanks to Joe Zuntz) (#116)
- Lightweight and map-less fields (#117)

# v1.0.2 changes
- Pip-installable (#95).
- Installation instructions on rtd (#95).

# v1.0.1 changes
- Input maps and templates can be masked (#93).
- Input theory power spectra are culled to lmax automatically (#90).
- Libsharp, HEALPix and libnmt automatically installed (#84).
- Code is now PEP8 compliant (#86).
- Workspaces are now stored as FITS files (which should make them easier to read externally) (#83).
- More convenient NmtBin constructors (#80).
- Lmax_mask, lmax_sht, lmax_bins (#74).
- Parallelized Gaussian covariance (#73).
- Ability to recover bandpower window functions easily (#55)


# v1.0 changes
- Support for rectangular pixelizations (CAR pixelization with Clenshaw-Curtis separation).
- Support for spin-0 and spin-2 covariance matrices.
- Convenience functions to recover bandpower windows and the unbinned mode-coupling matrix.
- Arbitrary ell-weighting (e.g. can compute D_l = l*(l+1)*C_l/2pi).
- Simultaneous calculation of the MCM for all spin-0 and spin-2 combinations.
