import healpy as hp
import numpy as np
import matplotlib.pyplot as plt


def plot_hp(
    m, proj_type, pol=False, title=None, rot=None, xsize=None, min=None, max=None
):
    """
    Plot a HEALPix map using mollview, orthview, or gnomview.

    Parameters
    ----------
    m : array-like or list/tuple of arrays
        HEALPix map(s). For polarization, should be (T, Q, U).
    proj_type : {'moll', 'orth', 'gnom'}
        Type of projection to use.
    pol : bool, default=True
        If True and m is a tuple/list of length 3, plot T, Q, U.
    title : str, optional
        Title of the plot (used for T map or scalar map).
    rot : tuple/list, optional
        Rotation for the view (lon, lat[, psi]).
    xsize : int, optional
        Only used in gnomview; defines the size in pixels.
    min, max : float, optional
        Color scale limits.
    """
    if proj_type not in {"moll", "orth", "gnom"}:
        raise ValueError(f"Unsupported projection type: {proj_type}")

    plot_func = {"moll": hp.mollview, "orth": hp.orthview, "gnom": hp.gnomview}[
        proj_type
    ]

    if pol and isinstance(m, (tuple, list)) and len(m) == 3:
        titles = ["T", "Q", "U"]
        for i in range(3):
            plot_func(m[i], title=titles[i], rot=rot, xsize=xsize, min=min, max=max)
    else:
        plot_func(m, title=title, rot=rot, xsize=xsize, min=min, max=max)

    plt.show()
