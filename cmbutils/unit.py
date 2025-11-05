import numpy as np
import healpy as hp
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


def mapdepth2sigma(delta: float, nside: int = None, pixel_size: float = None) -> float:
    """
    Convert map depth (μK·arcmin) to per-pixel noise σ (μK),
    given either HEALPix nside or pixel angular size.

    Parameters
    ----------
    delta : float
        Map depth in μK·arcmin.
    nside : int, optional
        HEALPix nside (if provided, overrides pixel_size).
    pixel_size : float, optional
        Pixel size (FWHM or resolution) in arcmin.

    Returns
    -------
    sigma_pix : float
        Noise standard deviation per pixel (μK).
    """
    Delta = delta * u.uK * u.arcmin

    if nside is not None:
        # HEALPix case
        pixel_area_sr = hp.nside2pixarea(nside) * u.sr
        pixel_area_arcmin2 = pixel_area_sr.to(u.arcmin**2)
    elif pixel_size is not None:
        # Flat-sky case
        pixel_area_arcmin2 = (pixel_size * u.arcmin) ** 2
    else:
        raise ValueError("Either nside or pixel_size must be provided.")

    sigma = (Delta / np.sqrt(pixel_area_arcmin2)).to(u.uK)
    return sigma.value


def uKpix2uKamin(
    sigma_pix: float, nside: int = None, pixel_size: float = None
) -> float:
    """
    Convert noise level from μK/pixel to μK·arcmin.
    (Inverse of mapdepth2sigma)

    Parameters
    ----------
    sigma_pix : float
        Noise per pixel, in μK/pixel.
    nside : int, optional
        HEALPix NSIDE. If provided, overrides pixel_size.
    pixel_size : float, optional
        Pixel angular resolution in arcmin (flat-sky approximation).

    Returns
    -------
    delta : float
        Map depth in μK·arcmin.
    """
    sigma_pix = sigma_pix * u.uK

    # Determine pixel area
    if nside is not None:
        pixel_area_sr = hp.nside2pixarea(nside) * u.sr
        pixel_area_arcmin2 = pixel_area_sr.to(u.arcmin**2)
    elif pixel_size is not None:
        pixel_area_arcmin2 = (pixel_size * u.arcmin) ** 2
    else:
        raise ValueError("Either nside or pixel_size must be provided.")

    # Convert μK/pix → μK·arcmin
    delta = (sigma_pix * np.sqrt(pixel_area_arcmin2)).to(u.uK * u.arcmin)
    return delta.value
