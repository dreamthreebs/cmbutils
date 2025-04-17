import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt
import pytest

from cmbutils.map import galactic_latitude_mask


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


if __name__ == "__main__":
    test_galactic_latitude_mask(plot_flag=True)
