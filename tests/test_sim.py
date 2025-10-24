import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from pylab import plot

from cmbutils.sim import (
    gen_noise,
    gen_cmb,
    gen_test_ps,
    gen_test_tsz,
    beta_model,
    beam_model,
    beta2bl,
    beam2bl,
)


def test_gen_noise(plot_flag=False):
    m_noise = gen_noise(nstd=1, nside=1024)

    if plot_flag:
        hp.mollview(m_noise[0], title="T noise")
        plt.show()


def test_gen_cmb(plot_flag=False):
    cls = np.load(f"../data/cmbcl_8k.npy")
    print(f"{cls.shape=}")

    m_cmb = gen_cmb(nside=512, cls=cls.T, beamFwhmArcmin=9)

    if plot_flag:
        hp.mollview(m_cmb[0], title="T")
        hp.mollview(m_cmb[1], title="Q")
        hp.mollview(m_cmb[2], title="U")
        plt.show()


def test_gen_test_ps(plot_flag=False):
    m_ps = gen_test_ps(nside=1024, lon=0, lat=0)

    if plot_flag:
        hp.gnomview(m_ps[0], title="T")
        hp.gnomview(m_ps[1], title="Q")
        hp.gnomview(m_ps[2], title="U")
        plt.show()


def test_beta_model(plot_flag=False):
    beta = 1.0
    theta_ac = 0.8
    fwhm = 1.4
    nside = 2048
    lmax = 3 * nside - 1

    theta = np.deg2rad(np.linspace(0, 20 * theta_ac, 3000)) / 60
    btheta = beta_model(1, theta, theta_ac, beta=beta)

    theta_beam = np.deg2rad(np.linspace(0, 2 * fwhm, 3000)) / 60
    btheta_beam = beam_model(1, theta_beam, fwhm)
    btheta_beam = btheta_beam / np.max(btheta_beam)
    wl = np.load("./mf_data/normalized_wl_default.npy")
    btheta_wl = hp.bl2beam(wl, theta_beam)
    btheta_wl = btheta_wl / np.max(btheta_wl)

    theta_1 = np.rad2deg(theta) * 60
    theta_2 = np.rad2deg(theta_beam) * 60
    if plot_flag:
        plt.plot(theta_1, btheta, label="beta")
        plt.plot(theta_2, btheta_beam, label="beam")
        plt.plot(theta_2, btheta_wl, label="wl")
        plt.legend()
        plt.show()

    bl_sz = beta2bl(lmax=lmax, theta_ac=theta_ac, beta=beta)
    bl_beam = beam2bl(lmax=lmax, fwhm=fwhm)
    l = np.arange(len(bl_sz))

    if plot_flag:
        plt.plot(l, bl_sz, label="beta")
        plt.plot(l, bl_beam, label="beam")
        plt.plot(l, wl, label="wl")
        plt.show()


def test_bl_beta(plot_flag=False):
    beta = 0.8
    theta_ac = 0.2
    fwhm = 1.4
    nside = 2048
    lmax = 3 * nside - 1

    theta_beam = np.deg2rad(np.linspace(0, 10 * fwhm, 30000)) / 60

    btheta_beam = beam_model(1, theta_beam, fwhm)
    btheta_beam = btheta_beam / np.max(btheta_beam)

    wl = np.load("./mf_data/normalized_wl_default.npy")
    btheta_wl = hp.bl2beam(wl, theta_beam)
    btheta_wl = btheta_wl / np.max(btheta_wl)

    bl_sz = beta2bl(lmax=lmax, theta_ac=theta_ac, beta=beta)
    bl_sz_1 = beta2bl(lmax=lmax, theta_ac=theta_ac, beta=beta, factor=0.5)
    bl_sz_2 = beta2bl(lmax=lmax, theta_ac=theta_ac, beta=beta, factor=1)
    bl_sz_3 = beta2bl(lmax=lmax, theta_ac=theta_ac, beta=beta, factor=3)
    bl_beam = beam2bl(lmax=lmax, fwhm=fwhm)
    l = np.arange(len(bl_sz))

    if plot_flag:
        plt.plot(l, bl_sz, label="beta 2 factor")
        plt.plot(l, bl_sz_1, label="beta 0.5 factor")
        plt.plot(l, bl_sz_2, label="beta 1 factor")
        plt.plot(l, bl_sz_3, label="beta 3 factor")
        plt.plot(l, bl_beam, label="beam fwhm 1.4")
        # plt.plot(l, (bl_sz - bl_sz_1) / bl_sz_1)
        # plt.plot(l, bl_beam, label="beam")
        # plt.plot(l, wl, label="wl")
        plt.legend()
        plt.show()


def test_bl_beam(plot_flag=False):
    beta = 0.8
    theta_ac = 0.8
    beam = 1.4
    nside = 2048
    lmax = 3 * nside - 1

    bl_th = hp.gauss_beam(fwhm=np.deg2rad(beam) / 60, lmax=lmax)
    bl_beam = beam2bl(lmax=lmax, fwhm=beam, factor=1.5)
    bl_beam_1 = beam2bl(lmax=lmax, fwhm=beam, factor=2)
    l = np.arange(len(bl_th))

    if plot_flag:
        plt.plot(l, bl_th, label="beam th")
        plt.plot(l, bl_beam, label="beam factor 1.5")
        plt.plot(l, bl_beam_1, label="beam factor 2")
        plt.legend()
        plt.show()


def _test_gen_test_tsz(plot_flag=False):
    nside = 2048
    m_tsz = gen_test_tsz(nside=nside, fwhm=1.5, theta_ac=0.5, beta=2 / 3)

    if plot_flag:
        hp.gnomview(m_tsz, title="tSZ")
        plt.show()


def _test_wl():
    wl_1 = np.load("./mf_data/normalized_wl_7arcmin.npy")
    wl_1 = wl_1 / np.max(wl_1)
    wl_2 = np.load("./mf_data/normalized_wl_9aarcmin.npy")
    wl_2 = wl_2 / np.max(wl_2)
    wl_3 = np.load("./mf_data/normalized_wl_default.npy")
    wl_3 = wl_3 / np.max(wl_3)
    l = np.arange(len(wl_1))

    plt.plot(l, wl_1, label="wl 7 arcmin")
    plt.plot(l, wl_2, label="wl 9 arcmin")
    plt.plot(l, wl_3, label="wl 11 arcmin")
    plt.loglog()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # test_gen_noise(plot_flag=True)
    # test_gen_cmb(plot_flag=True)
    # test_gen_test_ps(plot_flag=True)
    # test_beta_model(plot_flag=True)
    test_bl_beta(plot_flag=True)
    # test_bl_beam(plot_flag=True)
    # _test_gen_test_tsz(plot_flag=True)
    # _test_wl()
