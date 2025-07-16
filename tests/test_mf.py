import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from cmbutils.sim import gen_cmb, gen_test_ps
from cmbutils.mf import MatchedFilter


def test_mf(plot_flag=False):
    nside = 1024
    beam = 9  # arcmin
    lmax = 3 * nside - 1
    cls = np.load("../data/cmbcl_8k.npy").T
    m_cmb = gen_cmb(nside=nside, cls=cls, beamFwhmArcmin=beam, seed=2)
    m_ps = gen_test_ps(nside=nside, lon=0, lat=0, beam=beam)
    m = m_cmb + m_ps

    if plot_flag:
        hp.gnomview(m[0], reso=1, xsize=100, title="m obs")
        hp.gnomview(m_ps[0], reso=1, xsize=100, title="m_ps")
        plt.show()

    cl_tot = hp.anafast(m_cmb, lmax=lmax)[0]
    bl = hp.gauss_beam(fwhm=np.deg2rad(9) / 60, lmax=lmax)
    ell = np.arange(len(cl_tot))

    if plot_flag:
        plt.plot(ell, ell * (ell + 1) * cl_tot / (2 * np.pi) / bl**2)
        plt.show()

    obj_mf = MatchedFilter(nside=nside, lmax=lmax, cl_tot=cl_tot, beam=beam)
    obs_out, tot_out, snr, sigma, wl = obj_mf.run_mf(m[0].copy(), m_tot=m_cmb[0].copy())

    if plot_flag:
        hp.gnomview(obs_out, title="obs", reso=1, xsize=100)
        hp.gnomview(tot_out, title="tot", reso=1, xsize=100)
        hp.gnomview(snr, title="snr", reso=1, xsize=100)
        plt.show()


def mf_mean_value():
    """result:
    mean value from simulation: 128.20
    true value: 128.22
    """
    nside = 1024
    beam = 9  # arcmin
    lmax = 3 * nside - 1
    cls = np.load("../data/cmbcl_8k.npy").T

    m_ps = gen_test_ps(nside=nside, lon=0, lat=0, beam=beam)
    pix_idx = hp.ang2pix(nside=nside, theta=0, phi=0, lonlat=True)

    center_val_list = []
    for seed in np.arange(20):
        print(f"{seed=}")
        m_cmb = gen_cmb(nside=nside, cls=cls, beamFwhmArcmin=beam, seed=seed)
        m = m_cmb + m_ps

        cl_tot = hp.anafast(m_cmb, lmax=lmax)[0]
        # ell = np.arange(len(cl_tot))

        obj_mf = MatchedFilter(nside=nside, lmax=lmax, cl_tot=cl_tot, beam=beam)
        obs_out, tot_out, snr, sigma, wl = obj_mf.run_mf(
            m[0].copy(), m_tot=m_cmb[0].copy()
        )

        center_val = obs_out[pix_idx]
        print(f"{center_val=}")
        center_val_list.append(center_val)
    center_val_mean = np.mean(center_val_list)
    print(f"{center_val_mean=}")
    print(f"{m_ps[0, pix_idx]=}")


if __name__ == "__main__":
    test_mf(plot_flag=True)
    mf_mean_value()
