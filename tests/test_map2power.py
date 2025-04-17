import numpy as np
import healpy as hp
import pymaster as nmt
import matplotlib.pyplot as plt


from cmbutils.map2power import calc_power
from cmbutils.map import galactic_latitude_mask


def test_calc_power():
    nside = 512
    lmax = 3 * nside - 1
    mask = galactic_latitude_mask(nside=nside)
    apo_mask = nmt.mask_apodization(mask, aposize=6, apotype="C1")

    cl_cmb = np.load(f"./data/cmbcl_8k.npy").T

    # no beam scalar test
    m = hp.synfast(cls=cl_cmb, lmax=lmax, pol=True, new=True, nside=nside)
    ell_arr, dl_tt = calc_power(m[0], apo_mask=apo_mask, lmax=lmax)
    plt.plot(ell_arr, dl_tt)
    plt.show()

    # with beam scalar test
    beam = 9  # arcmin
    m = hp.synfast(
        cls=cl_cmb,
        lmax=lmax,
        pol=True,
        new=True,
        nside=nside,
        fwhm=np.deg2rad(beam) / 60,
    )
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam) / 60, lmax=lmax)
    ell_arr, dl_tt = calc_power(m[0], apo_mask=apo_mask, lmax=lmax, bl=bl)
    plt.plot(ell_arr, dl_tt)
    plt.show()
