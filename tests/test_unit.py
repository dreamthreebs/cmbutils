import numpy as np

from astropy import units as u
from cmbutils.unit import mapdepth2sigma, uKpix2uKamin


def test_mapdepth2sigma():
    freq = 30
    mapdepth = 10.6
    nstd = mapdepth2sigma(delta=mapdepth, nside=512)
    print(f"{nstd=}")
    assert np.allclose(nstd.value, 1.5427, rtol=1e-4)


def test_uKpix2uKamin():
    nside = 512
    mapdepth = uKpix2uKamin(1.5427, npix=12 * nside**2)
    print(f"{mapdepth=}")
    assert np.allclose(mapdepth, 10.6, rtol=1e-4)
