#%%
import cupy as cp
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from phase_field_2d.free_energy import get_free_energy
from phase_field_2d.initial_distribution import make_initial_distribution
from phase_field_2d.prepare_fft import prepare_fft
import phase_field_2d.plot as myplt
import phase_field_2d.save as mysave

class CDArray(cp.ndarray): ...

class PhaseField:
    """2D phase field modeling of non elastic system"""

    def __init__(self, w: float, temperature: float, c0: float) -> None:
        """__init__

        Args:
            self (Self): Self
            w (float): Margulus parameter
            temperature (float): annealing temperature (K)
        """
        self.T = temperature
        self.w = w
        self.c0 = c0

        self.Nx = 256
        self.Ny = 256
        self.dx: float = 1.0
        self.dy: float = 1.0
        self.nstep: int = 20000
        self.nprint: int = 1000
        self.dtime: float = 1e-2
        self.ttime: float = 0.0
        self.coefA: float = 1.0
        self.mobility: float = 1.0
        self.grad_coef: float = 1.0
        self.noise: float = 0.01

        # constant
        self.R:float = 8.31446262
        self.istep:int = 0

    def update(self) -> None:
        """update properties computed from the variables defined in __init__().
        """
        self.prepare_result_array()
        self.calc_fft_parameters()
        self.set_initial_distribution()

    def start(self) -> None:
        """start all computation.
        """
        self.update()
        self.compute_phase_field()

    def prepare_result_array(self) -> None:
        """prepare the computation result matrix and array.
        """
        self.energy_g = np.zeros(self.nstep) + np.nan
        self.energy_el = np.zeros(self.nstep) + np.nan

        self.con: CDArray = cp.zeros((self.Nx, self.Ny))
        self.conk: CDArray = cp.zeros((self.Nx, self.Ny))
        self.g: CDArray = cp.zeros((self.Nx, self.Ny))
        self.dgdcon: CDArray = cp.zeros((self.Nx, self.Ny))
        self.dgdconk: CDArray = cp.zeros((self.Nx, self.Ny))

    def calc_fft_parameters(
        self,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """calculate parameters which need to FFT.

        ## Args:
            self (Self): Self

        ## Returns:
            tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: kx, ky, k2, k4
        """
        kx, ky, k2, k4 = prepare_fft(self.Nx, self.Ny, self.dx, self.dy)
        self.kx = cp.array(kx)
        self.ky = cp.array(ky)
        self.k2 = cp.array(k2)
        self.k4 = cp.array(k4)

        return kx, ky, k2, k4

    def set_initial_distribution(self) -> None:
        """calculate initial composition and set to the property.
        """
        con = make_initial_distribution(self.Nx, self.Ny, self.c0, self.noise)
        self.con = cp.array(con)

    def compute_phase_field(self) -> None:
        """compute main part of phase field.
        """
        for istep in range(1, self.nstep + 1):
            self.istep = istep

            self.dfdcon, self.g = get_free_energy(self.con, self.w)

            self.energy_g[istep - 1] = cp.sum(self.g)

            self.conk = cp.fft.fft2(self.con)
            self.dfdconk = cp.fft.fft2(self.dfdcon)

            # Time integration
            numer = self.dtime * self.mobility * self.k2 * (self.dfdconk)
            denom = (
                1.0 + self.dtime * self.coefA * self.mobility * self.grad_coef * self.k4
            )

            self.conk = (self.conk - numer) / denom
            self.con = np.real(cp.fft.ifft2(self.conk))

            # con = con * (1 + np.sum(con[con < 0])/bulk)
            # con = con * (1 + np.sum(con[con > 1])/bulk)
            # con[con < 0] = 0
            # con[con > 1] = 1

            if (istep % self.nprint == 0) or (istep == 1):
                # plt.imshow(con_disp)の図の向きは、
                # y
                # ↑
                # |
                # + --→ x [100]
                # となる。
                con_res = self.con.transpose()
                con_disp = np.flipud(con_res)
                myplt.get_matrix_image(cp.asnumpy(con_disp), show=False)
                plt.savefig("test.pdf")

    def summary (self)-> None:
        """ result summary at t=istep
        """
        print("-------------------------------------")
        print(f"phase filed result at t={self.istep}")
        print("-------------------------------------")

        con = cp.asnumpy(self.con)

        print("composition")
        myplt.get_matrix_image(con)

        print(f"composition distribution result at t={self.istep}")
        myplt.plot_con_hist(con)

    def save (self)->None:
        mysave.create_directory("result")
        dirname = mysave.make_dir_name()
        mysave.create_directory(f"result/{dirname}")
        np.save(f"result/{dirname}/con_c0_{self.c0}-w_{self.w}-T_{self.T}-t_{int(self.istep*self.dtime)}.npy", self.con)


if __name__ == "__main__":
    phase_field = PhaseField(3.0, 300, 0.4)
    phase_field.start()


# %%
