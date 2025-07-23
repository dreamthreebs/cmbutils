import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from cmbutils.sim import gen_noise, gen_cmb, gen_test_ps, gen_test_tsz


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


def _test_gen_test_tsz(plot_flag=False):
    nside = 2048
    m_tsz = gen_test_tsz(nside=nside, fwhm=1.0, theta_ac=2.0, beta=2 / 3)

    if plot_flag:
        hp.gnomview(m_tsz, title="tSZ")
        plt.show()


if __name__ == "__main__":
    test_gen_noise(plot_flag=True)
    test_gen_cmb(plot_flag=True)
    test_gen_test_ps(plot_flag=True)
    _test_gen_test_tsz(plot_flag=True)
