import numpy as np
import healpy as hp


def gen_noise(nstd, nside, seed=None):
    """
    Generate Gaussian noise maps for temperature and polarization on a HEALPix sphere.

    Parameters
    ----------
    nstd : float
        Standard deviation of the Gaussian noise in μK per pixel.
    nside : int
        HEALPix Nside resolution parameter.
    seed : int or None, optional
        Seed for the random number generator (default: None).

    Returns
    -------
    noise_map : ndarray
        Noise maps in μK as a NumPy array of shape (3, npix), where
        npix = 12 * nside**2. The three rows correspond to
        [T, Q, U] noise components, respectively.
    """
    rng = np.random.default_rng(seed)
    npix = hp.nside2npix(nside)
    return nstd * rng.standard_normal(size=(3, npix))


def genCmb(nside, clFile, beamFwhmArcmin=0.0, seed=0, lmax=None):
    """
    Generate a CMB temperature anisotropy map from a theoretical power spectrum.

    Parameters
    ----------
    nside : int
        HEALPix Nside resolution parameter.
    clFile : str
        Path to a .npy file containing the CMB power spectra in μK².
        The array should be 2D with rows in the order [TT, EE, BB, TE].
    beamFwhmArcmin : float, optional
        Full-width at half-maximum of the Gaussian beam in arcminutes
        (default: 0.0 for no smoothing).
    seed : int, optional
        Seed for reproducibility (default: 0).
    lmax : int or None, optional
        Maximum multipole moment to synthesize.
        If None, defaults to `3 * nside - 1`.

    Returns
    -------
    cmb_map : ndarray
        Simulated CMB temperature anisotropy map in μK as a 1D array
        of length `npix = 12 * nside**2`.
    """
    if lmax is None:
        lmax = 3 * nside - 1

    np.random.seed(seed)
    cl = np.load(clFile)

    return hp.synfast(
        cls=cl, nside=nside, fwhm=np.deg2rad(beamFwhmArcmin / 60.0), lmax=lmax, new=True
    )
