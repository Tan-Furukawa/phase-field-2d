# %%
import cupy as cp
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from phase_field_2d.free_energy import get_free_energy
from phase_field_2d.initial_distribution import make_initial_distribution
from phase_field_2d.prepare_fft import prepare_fft
import phase_field_2d.plot as mplt
import phase_field_2d.save as mysave
from typing import Literal


class CDArray(cp.ndarray): ...


class PhaseField:
    """2D phase field modeling of non elastic system"""

    def __init__(
        self,
        w: float,
        c0: float,
        method: Literal["isotropic", "anisotropic"] = "isotropic",
        eta: NDArray = np.array([1.0, 1.0, 0.0]),
        record: bool = True,
        save_dir_name: str = "result",
    ) -> None:
        """__init__

        Args:
            self (Self): Self
            w (float): Margulus parameter
        """
        self.seed = 123
        self.w = w
        self.c0 = c0
        self.method = method
        self.eta = eta
        self.eta11 = float(eta[0])
        self.eta22 = float(eta[1])
        self.eta12 = float(eta[2])
        self.record = record
        self.save_dir_name = save_dir_name

        self.Nx = 128
        self.Ny = 128
        self.dx: float = 1.0
        self.dy: float = 1.0
        self.nsave: int = 2000
        self.nstep: int = 40000
        self.nprint: int = 4000
        self.dtime: float = 1e-2
        self.ttime: float = 0.0
        self.coefA: float = 1
        self.mobility: float = 1.0
        self.grad_coef: float = 1.0
        self.grad_coef6: float = 0
        self.noise: float = 0.01

        # constant
        self.R: float = 8.31446262
        self.istep: int = 0
        self.dir_name = mysave.make_dir_name()

        if self.record:
            mysave.create_directory(self.save_dir_name)
            mysave.create_directory(f"{self.save_dir_name}/{self.dir_name}")

    def update(self) -> None:
        """update properties computed from the variables defined in __init__()."""
        self.prepare_result_array()
        self.calc_fft_parameters()
        self.set_initial_distribution()

    def start(self) -> None:
        """start all computation."""
        self.update()
        self.compute_phase_field()

    def prepare_result_array(self) -> None:
        """prepare the computation result matrix and array."""
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
        NDArray[np.float64],
    ]:
        """calculate parameters which need to FFT.

        ## Args:
            self (Self): Self

        ## Returns:
            tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: kx, ky, k2, k4
        """
        kx, ky, k2, k4, k6, k2_anisotropy, k4_anisotropy = prepare_fft(
            self.Nx, self.Ny, self.dx, self.dy, self.eta
        )
        self.kx = cp.array(kx)
        self.ky = cp.array(ky)
        self.k2 = cp.array(k2)
        self.k4 = cp.array(k4)
        self.k6 = cp.array(k6)
        self.k2_anisotropy = cp.array(k2_anisotropy)
        self.k4_anisotropy = cp.array(k4_anisotropy)

        return kx, ky, k2, k4, k6

    def set_initial_distribution(self) -> None:
        """calculate initial composition and set to the property."""
        con = make_initial_distribution(
            self.Nx, self.Ny, self.c0, self.noise, seed=self.seed
        )
        self.con = cp.array(con)

    def compute_phase_field(self) -> None:
        """compute main part of phase field."""
        for istep in range(1, self.nstep + 1):
            self.istep = istep

            self.dfdcon, self.g = get_free_energy(self.con, self.w)

            self.energy_g[istep - 1] = cp.sum(self.g)

            self.conk = cp.fft.fft2(self.con)
            self.dfdconk = cp.fft.fft2(self.dfdcon)

            # Time integration
            if self.method == "isotropic":
                numer = self.dtime * self.mobility * self.k2 * (self.dfdconk)
                denom = 1.0 + self.dtime * self.coefA * self.mobility * (
                    self.grad_coef * self.k4 - self.grad_coef6 * self.k6
                )
            else:
                numer = self.dtime * self.mobility * self.k2 * (self.dfdconk)
                denom = 1.0 + self.dtime * self.coefA * self.mobility * (
                    self.grad_coef * self.k4_anisotropy - self.grad_coef6 * self.k6
                )

            self.conk = (self.conk - numer) / denom
            self.con = np.real(cp.fft.ifft2(self.conk))

            # con = con * (1 + np.sum(con[con < 0])/bulk)
            # con = con * (1 + np.sum(con[con > 1])/bulk)
            # con[con < 0] = 0
            # con[con > 1] = 1
            if (istep % self.nprint == 0) or (istep == 1):
                self.print(self.con)

            if (istep % self.nsave == 0) or (istep == 1):
                if self.record:
                    self.save()

    @classmethod
    def print(cls, con: CDArray) -> None:
        con_res = con.transpose()
        con_disp = np.flipud(con_res)
        mplt.get_matrix_image(cp.asnumpy(con_disp), show=False)
        plt.show()

    # def when_save(self) -> None:
    #     if self.record:
    #         self.save(make_directory=False)

    def save(self) -> None:
        np.save(
            f"{self.save_dir_name}/{self.dir_name}/con_{int(self.istep)}.npy",
            self.con,
        )
        instance_dict = mysave.instance_to_dict(
            self,
            [
                "w",
                "c0",
                "method",
                "eta11",
                "eta12",
                "eta22",
                "record",
                "save_dir_name",
                "Nx",
                "Ny",
                "dx",
                "dy",
                "nstep",
                "nprint",
                "dtime",
                "ttime",
                "coefA",
                "mobility",
                "grad_coef",
                "grad_coef6",
                "noise",
            ],
        )
        yaml_str = mysave.dump(instance_dict)
        mysave.save_str(
            f"{self.save_dir_name}/{self.dir_name}/parameters.yaml", yaml_str
        )

    def summary(self) -> None:
        """result summary at t=istep"""
        print("-------------------------------------")
        print(f"phase filed result at t={self.istep}")
        print("-------------------------------------")

        con = cp.asnumpy(self.con)

        print("composition")
        mplt.get_matrix_image(con)

        print(f"composition distribution result at t={self.istep}")
        mplt.plot_con_hist(con)

    # def save(self) -> None:
    #     mysave.create_directory("result")
    #     dirname = mysave.make_dir_name()
    #     mysave.create_directory(f"result/{dirname}")
    #     np.save(
    #         f"result/{dirname}/con_c0_{self.c0}-w_{self.w}-t_{int(self.istep*self.dtime)}.npy",
    #         self.con,
    #     )


if __name__ == "__main__":
    phase_field = PhaseField(
        2.5, 0.4, method="anisotropic", eta=np.array([2.0, 1.0, 0.0])
    )
    # phase_field.nstep = 1000
    # phase_field.nprint = 200
    phase_field.start()
    # %%

    mat = np.load("result/output_2024-08-31-15-19-33/con_10000.npy")
    PhaseField.print(mat)
    # plt.imshow(mat)


# %%
