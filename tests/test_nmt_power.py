import numpy as np
import healpy as hp
import pymaster as nmt
import matplotlib.pyplot as plt


from cmbutils.nmt_power import calc_power
from cmbutils.map import galactic_latitude_mask


def test_calc_power_sca(plot_flag=False):
    nside = 512
    lmax = 3 * nside - 1
    mask = galactic_latitude_mask(nside=nside)
    apo_mask = nmt.mask_apodization(mask, aposize=6, apotype="C1")

    cl_cmb = np.load(f"../data/cmbcl_8k.npy").T

    # no beam scalar test
    m = hp.synfast(cls=cl_cmb, lmax=lmax, pol=True, new=True, nside=nside)
    ell_arr, dl_tt = calc_power(m[0], apo_mask=apo_mask, lmax=lmax)

    if plot_flag is True:
        l = np.arange(lmax + 1)
        plt.plot(
            l, l * (l + 1) * cl_cmb[0, : lmax + 1] / (2 * np.pi), label="fiducial cmb"
        )
        plt.plot(ell_arr, dl_tt, label="TT")
        plt.legend()
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

    if plot_flag is True:
        l = np.arange(lmax + 1)
        plt.plot(
            l, l * (l + 1) * cl_cmb[0, : lmax + 1] / (2 * np.pi), label="fiducial cmb"
        )
        plt.plot(ell_arr, dl_tt, label="TT")
        plt.legend()
        plt.show()


def test_calc_power_pol(plot_flag=False):
    nside = 512
    lmax = 3 * nside - 1
    mask = galactic_latitude_mask(nside=nside)
    apo_mask = nmt.mask_apodization(mask, aposize=6, apotype="C1")

    cl_cmb = np.load(f"../data/cmbcl_8k.npy").T

    # no beam scalar test
    m = hp.synfast(cls=cl_cmb, lmax=lmax, pol=True, new=True, nside=nside)
    ell_arr, (dl_ee, dl_bb) = calc_power(m, apo_mask=apo_mask, lmax=lmax)

    if plot_flag is True:
        l = np.arange(lmax + 1)
        plt.plot(
            l,
            l * (l + 1) * cl_cmb[1, : lmax + 1] / (2 * np.pi),
            label="fiducial cmb EE",
        )
        plt.plot(
            l,
            l * (l + 1) * cl_cmb[2, : lmax + 1] / (2 * np.pi),
            label="fiducial cmb BB",
        )

        plt.plot(ell_arr, dl_ee, label="EE")
        plt.plot(ell_arr, dl_bb, label="BB")
        plt.loglog()
        plt.legend()
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
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam) / 60, lmax=lmax, pol=True)[:, 2]
    ell_arr, (dl_ee, dl_bb) = calc_power(m, apo_mask=apo_mask, lmax=lmax, bl=bl)

    if plot_flag is True:
        l = np.arange(lmax + 1)
        plt.plot(
            l,
            l * (l + 1) * cl_cmb[1, : lmax + 1] / (2 * np.pi),
            label="fiducial cmb EE",
        )
        plt.plot(
            l,
            l * (l + 1) * cl_cmb[2, : lmax + 1] / (2 * np.pi),
            label="fiducial cmb BB",
        )

        plt.plot(ell_arr, dl_ee, label="EE")
        plt.plot(ell_arr, dl_bb, label="BB")
        plt.loglog()
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # test_calc_power_sca(plot_flag=True)
    test_calc_power_pol(plot_flag=True)
