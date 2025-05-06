import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt
import pytest
import time

from cmbutils.map import galactic_latitude_mask, calc_fsky


def test_galactic_latitude_mask(plot_flag=False):
    nside = 64
    mask = galactic_latitude_mask(nside, lat_max_deg=20.0)
    apo_mask = nmt.mask_apodization(mask, aposize=6, apotype="C1")

    # Optional: plot the mask
    if plot_flag is True:
        hp.mollview(mask, title="Mask with |b| > 20° removed")
        hp.mollview(apo_mask, title="APODIZED Mask with |b| > 20° removed")
        hp.graticule()
        plt.show()


def test_calc_fsky():
    nside = 64
    mask = galactic_latitude_mask(nside, lat_max_deg=20.0)
    apo_mask = nmt.mask_apodization(mask, aposize=6, apotype="C1")

    fsky_bin = calc_fsky(mask=mask)
    fsky_apo = calc_fsky(mask=apo_mask)

    print(f"{fsky_bin=}")
    print(f"{fsky_apo=}")


def benchmark_fsky():
    mask = np.random.rand(12 * 512**2)  # NSIDE=512 full-sky map

    # Old version
    start = time.time()
    out1 = np.sum(mask**2) / np.size(mask)
    print("sum/size:", time.time() - start)

    # Faster version
    start = time.time()
    out2 = np.mean(mask**2)
    print("mean   :", time.time() - start)

    print(f"{out1=}, {out2=}")


if __name__ == "__main__":
    test_galactic_latitude_mask(plot_flag=True)
    test_calc_fsky()
    benchmark_fsky()
