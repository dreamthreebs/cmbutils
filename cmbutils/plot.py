import healpy as hp
import matplotlib.pyplot as plt


def plot_hp(
    m,
    proj_type,
    pol=True,
    title=None,
    rot=None,
    xsize=None,
    min=None,
    max=None,
    norm="hist",
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
        Size in pixels (default depends on projection type).
    min, max : float, optional
        Color scale limits.
    norm : {'hist', 'linear', 'log'}, default='hist'
        Color normalization method.
    """
    if proj_type not in {"moll", "orth", "gnom"}:
        raise ValueError(f"Unsupported projection type: {proj_type}")

    plot_func = {"moll": hp.mollview, "orth": hp.orthview, "gnom": hp.gnomview}[
        proj_type
    ]

    if xsize is None:
        default_xsize = {"moll": 800, "orth": 800, "gnom": 400}
        xsize = default_xsize[proj_type]

    def _plot_single_map(data, map_title):
        plot_func(
            data, title=map_title, rot=rot, xsize=xsize, min=min, max=max, norm=norm
        )

    if pol and isinstance(m, (tuple, list)) and len(m) == 3:
        titles = ["T", "Q", "U"]
        for i in range(3):
            _plot_single_map(m[i], titles[i])
    else:
        _plot_single_map(m, title)

    plt.show()
