"""
:mod:`pymaster` contains four basic classes:

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

:mod:`pymaster` also comes with a flat-sky version with \
    most of the same functionality:

- :class:`pymaster.field.NmtFieldFlat`
- :class:`pymaster.bins.NmtBinFlat`
- :class:`pymaster.workspaces.NmtWorkspaceFlat`
- :class:`pymaster.covariance.NmtCovarianceWorkspaceFlat`

- :func:`pymaster.workspaces.deprojection_bias_flat`
- :func:`pymaster.workspaces.compute_coupled_cell_flat`
- :func:`pymaster.workspaces.compute_full_master_flat`
- :func:`pymaster.covariance.gaussian_covariance_flat`
- :func:`pymaster.utils.mask_apodization_flat`
- :func:`pymaster.utils.synfast_flat`

Finally, :mod:`pymaster` has functionality to calculate \
the power spectra of fields defined at the arbitrary \
positions of a discrete set of sources. These should be \
defined using the following two field classes:

- :class:`pymaster.field.NmtFieldCatalog`
- :class:`pymaster.field.NmtFieldCatalogClustering`


Many of the NaMaster functions above accept or return sets \
of power spectra (arrays with one element per angular multipole) \
or bandpowers (binned versions of power spectra). In \
all cases, these are returned and provided as 2D arrays \
with shape ``[n_cls][nl]``, where ``n_cls`` is the number of \
power spectra and ``nl`` is either the number of multipoles \
or bandpowers. In all cases, ``n_cls`` should correspond \
with the spins of the two fields being correlated, and \
the ordering is as follows:

- Two spin-0 fields: ``n_cls`` = 1, [C_T1T2]
- One spin-0 field and one spin>0 field: ``n_cls`` = 2, [C_TE,C_TB]
- Two spin>0 fields: ``n_cls`` = 4, [C_E1E2,C_E1B2,C_E2B1,C_B1B2]

By defaut, all sky maps accepted and returned by the curved-sky \
functions are in the form of HEALPix maps exclusively with RING \
ordering. Note that NaMaster also supports CAR (Plate Carree) \
pixelization (see Example 9 in documentation), as well as catalog-based \
fields in a pixel-less manner (see Examples 11 and 12).
"""
try:
    from importlib.metadata import version
    __version__ = version(__name__)
except:  # noqa
    # This will happen on RTD, but that's fine
    __version__ = 'RTD'
    pass

from pymaster import nmtlib as lib  # noqa
import numpy as np  # noqa
from pymaster.utils import (  # noqa
    nmt_params,
    set_sht_calculator,
    set_n_iter_default,
    set_tol_pinv_default,
    get_default_params,
    NmtMapInfo,
    NmtAlmInfo,
    mask_apodization,
    mask_apodization_flat,
    synfast_spherical,
    synfast_flat,
    moore_penrose_pinvh,
    map2alm, alm2map,
)
from pymaster.field import (  # noqa
    NmtField, NmtFieldFlat,
    NmtFieldCatalog, NmtFieldCatalogClustering
)
from pymaster.bins import NmtBin, NmtBinFlat  # noqa
from pymaster.workspaces import (  # noqa
    NmtWorkspace,
    NmtWorkspaceFlat,
    deprojection_bias,
    compute_coupled_cell,
    compute_full_master,
    deprojection_bias_flat,
    compute_coupled_cell_flat,
    compute_full_master_flat,
    uncorr_noise_deprojection_bias,
    get_general_coupling_matrix,
)
from pymaster.covariance import (  # noqa
    NmtCovarianceWorkspace,
    gaussian_covariance,
    NmtCovarianceWorkspaceFlat,
    gaussian_covariance_flat,
)
