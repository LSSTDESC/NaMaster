"""
:mod:`pymaster` contains three basic classes:

- :class:`pymaster.field.NmtField`
- :class:`pymaster.bins.NmtBin`
- :class:`pymaster.workspaces.NmtWorkspace`
- :class:`pymaster.covariance.NmtCovarianceWorkspace`

and a number of functions

- :func:`pymaster.workspaces.deprojection_bias`
- :func:`pymaster.workspaces.compute_coupled_cell`
- :func:`pymaster.workspaces.compute_full_master`
- :func:`pymaster.covariance.gaussian_covariance`
- :func:`pymaster.utils.mask_apodization`
- :func:`pymaster.utils.synfast_spherical`

:mod:`pymaster` also comes with a flat-sky version with most of the same functionality:

- :class:`pymaster.field.NmtFieldFlat`
- :class:`pymaster.bins.NmtBinFlat`
- :class:`pymaster.workspaces.NmtWorkspaceFlat`

- :func:`pymaster.workspaces.deprojection_bias_flat`
- :func:`pymaster.workspaces.compute_coupled_cell_flat`
- :func:`pymaster.workspaces.compute_full_master_flat`
- :func:`pymaster.covariance.gaussian_covariance_flat`
- :func:`pymaster.utils.mask_apodization_flat`
- :func:`pymaster.utils.synfast_flat`

Many of these function accept or return sets of power spectra (arrays with one element per angular multipole) or bandpowers (binned versions of power spectra). In all cases, these are returned and provided as 2D arrays with shape [n_cls][nl], where n_cls is the number of power spectra and nl is either the number of multipoles or bandpowers. In all cases, n_cls should correspond with the spins of the two fields being correlated, and the ordering is as follows:

- Two spin-0 fields: n_cls=1, [C_T1T2]
- One spin-0 field and one spin-2 field: n_cls=2, [C_TE,C_TB]
- Two spin-2 fields: n_cls=4, [C_E1E2,C_E1B2,C_E2B1,C_B1B2]

All sky maps accepted and returned by these functions are in the form of HEALPix maps exclusively with RING ordering.

"""
from pymaster import nmtlib as lib
import numpy as np
from pymaster.utils import (
    mask_apodization,
    mask_apodization_flat,
    synfast_spherical,
    synfast_flat,
)
from pymaster.field import NmtField, NmtFieldFlat
from pymaster.bins import NmtBin, NmtBinFlat
from pymaster.workspaces import (
    NmtWorkspace,
    NmtWorkspaceFlat,
    deprojection_bias,
    compute_coupled_cell,
    compute_full_master,
    deprojection_bias_flat,
    compute_coupled_cell_flat,
    compute_full_master_flat,
    uncorr_noise_deprojection_bias,
)
from pymaster.covariance import (
    NmtCovarianceWorkspace,
    gaussian_covariance,
    NmtCovarianceWorkspaceFlat,
    gaussian_covariance_flat,
)
