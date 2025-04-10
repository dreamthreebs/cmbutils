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


def generate_bins(l_min_start=30, delta_l_min=30, l_max=1500, fold=0.3):
    bins_edges = []
    l_min = l_min_start  # starting l_min

    while l_min < l_max:
        delta_l = max(delta_l_min, int(fold * l_min))
        l_next = l_min + delta_l
        bins_edges.append(l_min)
        l_min = l_next

    # Adding l_max to ensure the last bin goes up to l_max
    bins_edges.append(l_max)
    return bins_edges[:-1], bins_edges[1:]


def dl2cl(D_ell):
    ell = np.arange(len(D_ell))
    mask = ell > 1
    C_ell = np.zeros_like(D_ell, dtype=np.float64)
    C_ell[mask] = (2 * np.pi * D_ell[mask]) / (ell[mask] * (ell[mask] + 1))
    C_ell[~mask] = 0
    return C_ell

