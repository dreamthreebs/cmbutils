from healpy.sphtfunc import smoothing
import numpy as np
import healpy as hp


def gen_noise(nstd, nside, seed=None):
    """
    Generate Gaussian noise maps for temperature and polarization on a HEALPix sphere.

    Parameters
    ----------
    nstd : float or np.ndarray
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


def gen_cmb(nside, cls, beamFwhmArcmin=0.0, seed=0, lmax=None):
    """
    Generate a CMB TQU map from a theoretical power spectrum.

    Parameters
    ----------
    nside : int
        HEALPix Nside resolution parameter.
    cls : str
        The CMB power spectra in μK².
        The array should be 2D with rows in the order [TT, EE, BB, TE], which consistent with 'new' parameter in synfast.
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
        Simulated CMB temperature anisotropy map in μK as a 2D array with dimension (3, npix)
    """
    if lmax is None:
        lmax = 3 * nside - 1

    np.random.seed(seed)

    return hp.synfast(
        cls=cls,
        nside=nside,
        fwhm=np.deg2rad(beamFwhmArcmin / 60.0),
        lmax=lmax,
        new=True,
    )


def gen_test_ps(nside, lon, lat, flux_i=1000, flux_q=1000, flux_u=1000, beam=9):
    """
    Generate a smoothed test point source map in I, Q, U Stokes components.

    Parameters
    ----------
    nside : int
        HEALPix NSIDE resolution parameter.
    lon : float
        Longitude (in degrees) of the point source in Galactic or Equatorial coordinates, depending on map convention.
    lat : float
        Latitude (in degrees) of the point source in the same coordinate system as `lon`.
    flux_i : float, optional
        Flux value (arbitrary units) for the I (intensity) component. Default is 1000.
    flux_q : float, optional
        Flux value for the Q polarization component. Default is 1000.
    flux_u : float, optional
        Flux value for the U polarization component. Default is 1000.
    beam : float, optional
        Full width at half maximum (FWHM) of the Gaussian smoothing beam in arcminutes. Default is 9 arcmin.

    Returns
    -------
    sm : ndarray
        A (3, N_pix) NumPy array containing the smoothed I, Q, U maps with a single point source inserted
        at the specified location and convolved with a symmetric Gaussian beam.
    """
    pix_idx = hp.ang2pix(nside=nside, theta=lon, phi=lat, lonlat=True)
    m = np.zeros(shape=(3, hp.nside2npix(nside=nside)))

    m[0, pix_idx] = flux_i
    m[1, pix_idx] = flux_q
    m[2, pix_idx] = flux_u
    sm = hp.smoothing(m, fwhm=np.deg2rad(beam) / 60)
    return sm


def beam_model(norm_beam, theta, FWHM):
    """
    Gaussian beam model.
    FWHM in arcmin
    """
    sigma = np.deg2rad(FWHM) / 60 / (np.sqrt(8 * np.log(2)))
    return norm_beam / (2 * np.pi * sigma**2) * np.exp(-((theta) ** 2) / (2 * sigma**2))


def beta_model(norm_beam, theta, theta_c, beta):
    """
    Beta model profile for the tSZ effect.
    theta_c in arcmin
    """
    temp = (1 + theta**2 / np.deg2rad(theta_c / 60) ** 2) ** (-(3 * beta - 1) / 2)
    return norm_beam * temp


def beam2bl(lmax, fwhm, factor=2):
    """
    Convert beam(theta) to b(l).
    fwhm in arcmin
    """
    # fwhm = np.deg2rad(fwhm/60)  # arcmin
    # sigma = fwhm / (np.sqrt(8 * np.log(2)))
    theta = np.linspace(0, factor * np.deg2rad(fwhm) / 60, 30000)
    btheta = beam_model(1, theta, fwhm)
    b_ell = hp.beam2bl(btheta, theta, lmax=lmax)
    b_ell /= b_ell[0]  # normalize
    return b_ell


def beta2bl(lmax, theta_ac, beta=1.0, factor=2):
    """
    Convert Compton-y(theta) to b(l).
    fwhm in arcmin
    """
    # theta_ac = np.deg2rad(theta_ac/60)  # arcmin
    theta = np.linspace(0, factor * np.deg2rad(theta_ac) / 60, 30000)
    btheta = beta_model(1, theta, theta_ac, beta=beta)
    b_ell = hp.beam2bl(btheta, theta, lmax=lmax)
    b_ell /= b_ell[0]  # normalize
    return b_ell


def gen_test_tsz(nside, fwhm, theta_ac=1, beta=2 / 3):
    lmax = 3 * nside - 1
    npix = hp.nside2npix(nside=nside)
    m = np.zeros(shape=npix)
    ctr_val = 1
    ctr_pix = hp.ang2pix(nside=nside, theta=0, phi=0, lonlat=True)
    m[ctr_pix] = ctr_val

    bl_sz = beta2bl(lmax=lmax, theta_ac=theta_ac, beta=beta)
    bl_beam = beam2bl(lmax=lmax, fwhm=fwhm)
    m_sz = hp.smoothing(m, beam_window=bl_sz * bl_beam, lmax=lmax)
    m_sz = m_sz / np.max(m_sz)
    # hp.gnomview(m_sz)
    # plt.show()

    return m_sz
