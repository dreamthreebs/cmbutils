import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from cmbutils.sim import gen_noise, gen_cmb


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


if __name__ == "__main__":
    test_gen_noise(plot_flag=True)
    test_gen_cmb(plot_flag=True)
