import healpy as hp
import numpy as np


def adjust_lat(lat):
    if lat < -90 or lat > 90:
        lat = lat % 360
        if lat < -90:
            lat = -180 - lat
        if (lat > 90) and (lat <= 270):
            lat = 180 - lat
        elif lat > 270:
            lat = lat - 360
    return lat


def galactic_latitude_mask(nside: int, lat_max_deg: float = 20.0) -> np.ndarray:
    """
    Create a HEALPix mask where pixels with |b| > lat_max_deg are masked (set to 0).

    Parameters:
        nside (int): HEALPix resolution parameter.
        lat_max_deg (float): Latitude cutoff in degrees (default is 20).

    Returns:
        mask (np.ndarray): Binary mask (1 = keep, 0 = mask).
    """
    npix = hp.nside2npix(nside)

    # Get theta, phi for all pixels in equatorial coordinates
    theta, phi = hp.pix2ang(nside, np.arange(npix))

    # Convert theta to Galactic latitude b
    vec = hp.ang2vec(theta, phi)
    gal_lon, gal_lat = hp.vec2ang(vec, lonlat=True)

    # Apply mask: keep only pixels within |b| <= lat_max_deg
    mask = (np.abs(gal_lat) > lat_max_deg).astype(np.int8)

    return mask
