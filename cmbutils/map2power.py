import numpy as np
import pymaster as nmt


def dl2cl(D_ell: np.ndarray) -> np.ndarray:
    """
    Convert D_ell to C_ell using the relation:
        C_ell = D_ell * 2π / (ell * (ell + 1))

    Parameters
    ----------
    D_ell : np.ndarray
        Array of D_ell values.

    Returns
    -------
    C_ell : np.ndarray
        Corresponding C_ell values, with C_ell[0] and C_ell[1] set to zero.
    """
    ell = np.arange(len(D_ell))
    mask = ell > 1
    C_ell = np.zeros_like(D_ell, dtype=np.float64)
    C_ell[mask] = (2 * np.pi * D_ell[mask]) / (ell[mask] * (ell[mask] + 1))
    C_ell[~mask] = 0
    return C_ell


def generate_bins(l_min_start=30, delta_l_min=30, l_max=1500, fold=0.3):
    """
    Generate variable-width ell bin edges for power spectrum binning.

    Parameters
    ----------
    l_min_start : int
        Minimum ell to start binning.
    delta_l_min : int
        Minimum width of a bin.
    l_max : int
        Maximum ell value to bin up to.
    fold : float
        Multiplicative factor controlling bin width growth with ell.

    Returns
    -------
    lmin_arr : list[int]
        List of lower bin edges.
    lmax_arr : list[int]
        List of upper bin edges.
    """
    bins_edges = []
    l_min = l_min_start

    while l_min < l_max:
        delta_l = max(delta_l_min, int(fold * l_min))
        l_next = l_min + delta_l
        bins_edges.append(l_min)
        l_min = l_next

    bins_edges.append(l_max)
    return bins_edges[:-1], bins_edges[1:]


def calc_dl_from_scalar_map(m_t, bl, apo_mask, bin_dl, masked_on_input, lmax):
    """
    Compute binned D_ell power spectrum for a scalar map.

    Parameters
    ----------
    m_t : np.ndarray
        Temperature map on the sphere.
    bl : np.ndarray
        Beam transfer function.
    apo_mask : np.ndarray
        Apodized mask array.
    bin_dl : nmt.NmtBin
        Binning scheme.
    masked_on_input : bool
        Whether the input map is already masked.
    lmax : int
        Maximum multipole.

    Returns
    -------
    dl : np.ndarray
        Binned D_ell power spectrum (TT).
    """
    f0 = nmt.NmtField(
        apo_mask,
        [m_t],
        beam=bl,
        masked_on_input=masked_on_input,
        lmax=lmax,
        lmax_mask=lmax,
    )
    dl = nmt.compute_full_master(f0, f0, bin_dl)
    return dl[0]


def calc_dl_from_pol_map(
    m_q, m_u, bl, apo_mask, bin_dl, masked_on_input, purify_b, lmax
):
    """
    Compute binned D_ell power spectra (EE and BB) from polarization maps.

    Parameters
    ----------
    m_q : np.ndarray
        Q Stokes parameter map.
    m_u : np.ndarray
        U Stokes parameter map.
    bl : np.ndarray
        Beam transfer function.
    apo_mask : np.ndarray
        Apodized mask array.
    bin_dl : nmt.NmtBin
        Binning scheme.
    masked_on_input : bool
        Whether the input maps are already masked.
    purify_b : bool
        Whether to apply B-mode purification.
    lmax : int
        Maximum multipole.

    Returns
    -------
    dl_ee : np.ndarray
        Binned EE power spectrum.
    dl_bb : np.ndarray
        Binned BB power spectrum.
    """
    f2p = nmt.NmtField(
        apo_mask,
        [m_q, m_u],
        beam=bl,
        masked_on_input=masked_on_input,
        purify_b=purify_b,
        lmax=lmax,
        lmax_mask=lmax,
    )
    w22p = nmt.NmtWorkspace.from_fields(f2p, f2p, bin_dl)
    dl = w22p.decouple_cell(nmt.compute_coupled_cell(f2p, f2p))
    return dl[0], dl[3]


def calc_power(
    m, apo_mask, lmax, masked_on_input=False, bl=None, bin_dl=None, purify_b=True
):
    """
    Calculate D_ell power spectra from a scalar or TQU map.

    Parameters
    ----------
    m : np.ndarray
        Input map. Should be 1D (T-only) or 2D with shape (3, Npix) for TQU.
    apo_mask : np.ndarray
        Apodized mask array.
    lmax : int
        Maximum multipole to compute.
    masked_on_input : bool, optional
        Whether the map is already masked. Default is False.
    bl : np.ndarray or None, optional
        Beam transfer function. Default is None.
    bin_dl : nmt.NmtBin or None, optional
        Binning scheme. If None, defaults to linear binning with Δl=10.
    purify_b : bool, optional
        Whether to apply B-mode purification (used only for polarization). Default is True.

    Returns
    -------
    ell_arr : np.ndarray
        Effective ell values for each bin.
    dl : np.ndarray or tuple[np.ndarray, np.ndarray]
        Binned D_ell power spectra: TT for scalar, (EE, BB) for polarization.
    """
    # check scalar or tqu maps
    if m.ndim == 1:
        pol = False
    elif (m.ndim == 2) and m.shape[0] == 3:
        pol = True
    else:
        raise ValueError("input map is not scalar or TQU, check your input please!")
    print(f"{m.ndim=}, {pol=}")

    if bin_dl is None:
        bin_dl = nmt.NmtBin.from_lmax_linear(lmax=lmax, nlb=10, is_Dell=True)

    ell_arr = bin_dl.get_effective_ells()
    if pol is False:
        dl_sca = calc_dl_from_scalar_map(
            m_t=m,
            bl=bl,
            apo_mask=apo_mask,
            bin_dl=bin_dl,
            masked_on_input=masked_on_input,
            lmax=lmax,
        )
        return ell_arr, dl_sca
    else:
        dl_ee, dl_bb = calc_dl_from_pol_map(
            m_q=m[1],
            m_u=m[2],
            bl=bl,
            apo_mask=apo_mask,
            bin_dl=bin_dl,
            masked_on_input=masked_on_input,
            purify_b=purify_b,
            lmax=lmax,
        )
        return ell_arr, (dl_ee, dl_bb)
