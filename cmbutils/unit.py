import numpy as np
from astropy import units as u


def mJysr_to_uKCMB(intensity_mJy, frequency_GHz):
    # Constants
    c = 2.99792458e8  # Speed of light in m/s
    h = 6.62607015e-34  # Planck constant in J*s
    k = 1.380649e-23  # Boltzmann constant in J/K
    T_CMB = 2.725  # CMB temperature in Kelvin
    frequency_Hz = frequency_GHz * 1e9  # Convert frequency to Hz from GHz
    x = (h * frequency_Hz) / (k * T_CMB)  # Calculate x = h*nu/(k*T)
    # Calculate the derivative of the Planck function with respect to temperature, dB/dT
    dBdT = (2.0 * h * frequency_Hz**3 / c**2 / T_CMB) * (
        x * np.exp(x) / (np.exp(x) - 1) ** 2
    )
    intensity_Jy = intensity_mJy * 1e-3  # Convert intensity from mJy to Jy
    intensity_W_m2_sr_Hz = intensity_Jy * 1e-26  # Convert Jy/sr to W/m^2/sr/Hz
    uK_CMB = (
        intensity_W_m2_sr_Hz / dBdT * 1e6
    )  # Convert to uK_CMB, taking the inverse of dB/dT
    return uK_CMB


def mapdepth2sigma(delta, nside):
    Delta = delta * u.uK * u.arcmin
    # Calculate pixel area in steradian
    pixel_area_sr = 4 * np.pi / (12 * nside**2) * u.sr

    # Convert pixel area to arcmin^2
    pixel_area_arcmin2 = pixel_area_sr.to(u.arcmin**2)

    # Calculate sigma
    sigma = Delta / np.sqrt(pixel_area_arcmin2)

    return sigma


def uKpix2uKamin(n: float, npix: int) -> float:
    """
    Convert noise level from μK/pixel to μK/arcmin.

    Parameters
    ----------
    n : float
        Noise per pixel, in μK/pixel.
    npix : int
        Total number of pixels in the full sky map.

    Returns
    -------
    float
        Noise level in μK/arcmin.
    """
    factor = np.sqrt((360 * 60) ** 2 / (np.pi * npix))
    return n * factor
