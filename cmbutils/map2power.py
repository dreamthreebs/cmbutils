import numpy as np
import pymaster as nmt


def dl2cl(D_ell):
    ell = np.arange(len(D_ell))
    mask = ell > 1
    C_ell = np.zeros_like(D_ell, dtype=np.float64)
    C_ell[mask] = (2 * np.pi * D_ell[mask]) / (ell[mask] * (ell[mask] + 1))
    C_ell[~mask] = 0
    return C_ell


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


def calc_dl_from_scalar_map(m_t, bl, apo_mask, bin_dl, masked_on_input, lmax):
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
    # dl = nmt.workspaces.compute_full_master(pol_field, pol_field, b=bin_dl)
    dl = w22p.decouple_cell(nmt.compute_coupled_cell(f2p, f2p))
    return dl[0], dl[3]


def calc_power(
    m, apo_mask, lmax, masked_on_input=False, bl=None, bin_dl=None, purify_b=True
):
    # check scalar or tqu maps
    if m.ndim == 1:
        pol = False
    elif (m.ndim == 2) and m.shape[0] == 3:
        pol = True
    else:
        raise ValueError("input map is not scalar or TQU, check you input please!")
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
        dl_pol = calc_dl_from_pol_map(
            m_q=m[1],
            m_u=m[2],
            bl=bl,
            apo_mask=apo_mask,
            bin_dl=bin_dl,
            masked_on_input=masked_on_input,
            purify_b=purify_b,
            lmax=lmax,
        )
        return ell_arr, dl_pol
