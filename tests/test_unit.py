import numpy as np
import healpy as hp

from astropy import units as u
from cmbutils.unit import mapdepth2sigma, uKpix2uKamin


def test_mapdepth2sigma():
    freq = 30
    mapdepth = 10.6
    nstd = mapdepth2sigma(delta=mapdepth, nside=512)
    print(f"{nstd=}")
    assert np.allclose(nstd, 1.5427, rtol=1e-4)

    nstd = mapdepth2sigma(delta=mapdepth, nside=350)
    print(f"nside = 350, {nstd=}")

    nstd = mapdepth2sigma(delta=mapdepth, pixel_size=10)
    print(f"pixel_size = 10, {nstd=}")


def test_uKpix2uKamin():
    nside = 512
    mapdepth = uKpix2uKamin(1.5427, nside=nside)
    print(f"{mapdepth=}")
    assert np.allclose(mapdepth, 10.6, rtol=1e-4)

    mapdepth = uKpix2uKamin(sigma_pix=1.06, nside=350)
    print(f"{mapdepth=}")

    mapdepth = uKpix2uKamin(sigma_pix=1.06, pixel_size=10)
    print(f"{mapdepth=}")
