import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from cmbutils.sim import gen_cmb, gen_test_ps, gen_test_tsz, beta2bl
from cmbutils.mf import MatchedFilter


def _test_mf_ps(plot_flag=False):
    nside = 1024
    beam = 9  # arcmin
    lmax = 3 * nside - 1
    nstd = 1
    cls = np.load("../data/cmbcl_8k.npy").T
    m_cmb = gen_cmb(nside=nside, cls=cls, beamFwhmArcmin=beam, seed=2)
    m_ps = gen_test_ps(nside=nside, lon=0, lat=0, beam=beam)
    m_noise = nstd * np.random.normal(
        loc=0, scale=1, size=(3, hp.nside2npix(nside=nside))
    )
    m = m_cmb + m_ps + m_noise

    if plot_flag:
        hp.gnomview(m[0], reso=1, xsize=100, title="m obs")
        hp.gnomview(m_ps[0], reso=1, xsize=100, title="m_ps")
        hp.gnomview(m_noise[0], reso=1, xsize=100, title="m_noise")
        plt.show()

    cl_tot = hp.anafast(m_cmb + m_noise, lmax=lmax)[0]
    bl = hp.gauss_beam(fwhm=np.deg2rad(9) / 60, lmax=lmax)
    ell = np.arange(len(cl_tot))

    if plot_flag:
        plt.plot(ell, ell * (ell + 1) * cl_tot / (2 * np.pi) / bl**2)
        plt.show()

    obj_mf = MatchedFilter(nside=nside, lmax=lmax, cl_tot=cl_tot, beam=beam)
    obs_out, tot_out, snr, sigma, wl = obj_mf.run_mf(
        m[0].copy(), m_tot=m_cmb[0].copy() + m_noise[0].copy()
    )

    if plot_flag:
        hp.gnomview(obs_out, title="obs", reso=1, xsize=100)
        hp.gnomview(tot_out, title="tot", reso=1, xsize=100)
        hp.gnomview(snr, title="snr", reso=1, xsize=100)
        plt.show()


def mf_ps_mean_value():
    """result:
    mean value from simulation: 128.20, if CMB is the only total component
    mean value from simulation: 129.62, considering the CMB + noise

    true value: 128.22
    """
    nside = 1024
    beam = 9  # arcmin
    lmax = 3 * nside - 1
    nstd = 1
    cls = np.load("../data/cmbcl_8k.npy").T

    m_ps = gen_test_ps(nside=nside, lon=0, lat=0, beam=beam)
    pix_idx = hp.ang2pix(nside=nside, theta=0, phi=0, lonlat=True)

    center_val_list = []
    for seed in np.arange(20):
        print(f"{seed=}")
        m_cmb = gen_cmb(nside=nside, cls=cls, beamFwhmArcmin=beam, seed=seed)
        m_noise = nstd * np.random.normal(
            loc=0, scale=1, size=(3, hp.nside2npix(nside=nside))
        )
        m = m_cmb + m_ps + m_noise

        cl_tot = hp.anafast(m_cmb + m_noise, lmax=lmax)[0]
        # ell = np.arange(len(cl_tot))

        obj_mf = MatchedFilter(nside=nside, lmax=lmax, cl_tot=cl_tot, beam=beam)
        obs_out, tot_out, snr, sigma, wl = obj_mf.run_mf(
            m[0].copy(), m_tot=m_cmb[0].copy() + m_noise[0].copy()
        )

        center_val = obs_out[pix_idx]
        print(f"{center_val=}")
        center_val_list.append(center_val)
    center_val_mean = np.mean(center_val_list)
    print(f"{center_val_mean=}")
    print(f"{m_ps[0, pix_idx]=}")


def mf_ps_Q_theta(plot_flag=False):
    """
    test if I use different beam to fit, how the Q changes
    results: seems the Q theta varies intensely, maybe coming from the gaussian profile, so it is not similar with Q theta in SZ profile
    """
    nside = 1024
    beam = 9  # arcmin
    lmax = 3 * nside - 1
    nstd = 5

    cls = np.load("../data/cmbcl_8k.npy").T
    m_cmb = gen_cmb(nside=nside, cls=cls, beamFwhmArcmin=beam, seed=3)
    m_ps = gen_test_ps(nside=nside, lon=0, lat=0, beam=beam)
    m_noise = nstd * np.random.normal(
        loc=0, scale=1, size=(3, hp.nside2npix(nside=nside))
    )
    m = m_cmb + m_ps + m_noise

    pix_idx = hp.ang2pix(nside=nside, theta=0, phi=0, lonlat=True)

    if plot_flag:
        hp.gnomview(m[0], reso=1, xsize=100, title="m obs")
        hp.gnomview(m_ps[0], reso=1, xsize=100, title="m_ps")
        plt.show()

    print(f"{np.max(m)=}")
    print(f"{np.max(m_ps)=}")

    cl_tot = hp.anafast(m_cmb + m_noise, lmax=lmax)[0]
    bl = hp.gauss_beam(fwhm=np.deg2rad(9) / 60, lmax=lmax)

    obs_list = []
    for beam in [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]:
        obj_mf = MatchedFilter(nside=nside, lmax=lmax, cl_tot=cl_tot, beam=beam)
        obs_out, tot_out, snr, sigma, wl = obj_mf.run_mf(
            m_ps[0].copy(), m_tot=m_cmb[0].copy() + m_noise[0].copy(), normalize=False
        )

        if plot_flag:
            hp.gnomview(obs_out, reso=1, xsize=100, title="m obs")
            hp.gnomview(snr, reso=1, xsize=100, title="m_snr")
            plt.show()

        print(f"{beam=}, {obs_out[pix_idx] / beam**2=}")
        obs_list.append(obs_out[pix_idx] / beam**2)

    np.save("./mf_data/obs_arr.npy", np.array(obs_list))
    # np.save("./mf_data/tsz_arr", np.array(obs_list))


def _test_mf_tsz(plot_flag=False):
    nside = 2048
    beam = 1.4  # arcmin
    lmax = 3 * nside - 1
    nstd = 50
    theta_ac = 0.8
    beta = 1.0

    cls = np.load("../data/cmbcl_8k.npy").T
    m_cmb = gen_cmb(nside=nside, cls=cls, beamFwhmArcmin=beam, seed=2)[0].copy()
    m_tsz = -500 * gen_test_tsz(nside=nside, fwhm=beam, theta_ac=theta_ac, beta=beta)
    m_noise = nstd * np.random.normal(loc=0, scale=1, size=hp.nside2npix(nside=nside))
    m = m_cmb + m_tsz + m_noise
    # m = m * -1.0

    if plot_flag:
        hp.gnomview(m, reso=1, xsize=100, title="m obs")
        hp.gnomview(m_tsz, reso=1, xsize=100, title="m_ps")
        hp.gnomview(m_noise, reso=1, xsize=100, title="m_noise")
        plt.show()

    cl_tot = hp.anafast(m_cmb + m_noise, lmax=lmax)
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam) / 60, lmax=lmax)

    bl_beta = beta2bl(lmax=lmax, theta_ac=theta_ac, beta=beta)
    # bl_beta = beta2bl(lmax=lmax, theta_ac=3, beta=beta)
    ell = np.arange(len(cl_tot))

    if plot_flag:
        plt.loglog(ell, ell * (ell + 1) * cl_tot / (2 * np.pi) / bl**2)
        plt.show()

    obj_mf = MatchedFilter(
        nside=nside, lmax=lmax, cl_tot=cl_tot, beam=beam, beam_window=bl_beta
    )
    obs_out, tot_out, snr, sigma, wl = obj_mf.run_mf(m.copy(), m_tot=m_cmb + m_noise)

    if plot_flag:
        hp.gnomview(obs_out, title="obs", reso=1, xsize=100)
        hp.gnomview(tot_out, title="tot", reso=1, xsize=100)
        hp.gnomview(snr, title="snr", reso=1, xsize=100)
        plt.show()

    obs_out, tot_out, snr, sigma, wl = obj_mf.run_mf(
        m_tsz.copy(), m_tot=m_cmb + m_noise, overwrite_wl=False
    )

    if plot_flag:
        hp.gnomview(obs_out, title="obs", reso=1, xsize=100)
        plt.show()


def mf_tzs_Q_theta(plot_flag=False):
    nside = 2048
    beam = 1.4  # arcmin
    # lmax = 3 * nside - 1
    lmax = 2 * nside
    nstd = 50
    theta_ac = 1.5
    beta = 1.2

    pix_idx = hp.ang2pix(nside=nside, theta=0, phi=0, lonlat=True)
    cls = np.load("../data/cmbcl_8k.npy").T
    m_cmb = gen_cmb(nside=nside, cls=cls, beamFwhmArcmin=beam, seed=2)[0].copy()
    m_tsz = -400 * gen_test_tsz(nside=nside, fwhm=beam, theta_ac=theta_ac, beta=beta)
    m_noise = nstd * np.random.normal(loc=0, scale=1, size=hp.nside2npix(nside=nside))
    m = m_cmb + m_tsz + m_noise

    if plot_flag:
        hp.gnomview(m, reso=1, xsize=100, title="m obs")
        hp.gnomview(m_tsz, reso=1, xsize=100, title="m_ps")
        hp.gnomview(m_noise, reso=1, xsize=100, title="m_noise")
        plt.show()

    cl_tot = hp.anafast(m_cmb + m_noise, lmax=lmax)
    bl = hp.gauss_beam(fwhm=np.deg2rad(beam) / 60, lmax=lmax)
    ell = np.arange(len(cl_tot))

    obs_list = []
    for theta_ac in np.arange(0.1, 3, 0.4):
        bl_beta = beta2bl(lmax=lmax, theta_ac=theta_ac, beta=beta)
        obj_mf = MatchedFilter(
            nside=nside, lmax=lmax, cl_tot=cl_tot, beam=beam, beam_window=bl_beta
        )
        obs_out, tot_out, snr, sigma, wl = obj_mf.run_mf(
            m_tsz.copy(), m_tot=m_cmb + m_noise
        )

        hp.gnomview(obs_out, title="obs_out")
        hp.gnomview(snr, title="snr")
        plt.show()

        print(f"{theta_ac=}, {-obs_out[pix_idx]=}")
        obs_list.append(-obs_out[pix_idx])

    np.save("./mf_data/obs_arr.npy", np.array(obs_list))
    # np.save("./mf_data/tsz_arr", np.array(obs_list))


def plot_tsz_q_theta():
    theta = np.arange(0.1, 3, 0.4)
    Q_theta = np.load("./mf_data/obs_arr.npy")
    plt.plot(theta, Q_theta)
    plt.xlabel("theta")
    plt.ylabel("Q_theta")
    plt.show()


if __name__ == "__main__":
    # test_mf_ps(plot_flag=True)
    # mf_ps_mean_value()
    mf_ps_Q_theta(plot_flag=False)
    # _test_mf_tsz(plot_flag=True)
    # mf_tzs_Q_theta(plot_flag=True)
    # plot_tsz_q_theta()
