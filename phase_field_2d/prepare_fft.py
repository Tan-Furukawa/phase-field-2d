# %%
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def prepare_fft(
    Nx: int, Ny: int, dx: float, dy: float, eta: NDArray = np.array([1.0, 1.0, 0.0])
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """prepare wave vectors and the related tensors for FFT

    ## Args:
        Nx (int): number of kx vector
        Ny (int): number of ky vector
        dx (float): discrete size of x
        dy (float): discrete size of y
        eta (NDArray): k11, k22, k12

    ## Returns:
        Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: kx (vector), ky(vector), k2(tensor), k4(tensor). k2 = kx[i]**2 + ky[j]**2, k4 = k2**2

    ## examples:
        >>> Nx = 8
        >>> Ny = 8
        >>> dx = 1.0
        >>> dy = 1.0
        >>> kx, ky, k2, k4 = prepare_fft(Nx, Ny, dx, dy)
        >>> np.any(np.isclose(kx, np.array([ 0.        ,  0.78539816,  1.57079633,  2.35619449, -3.14159265, -2.35619449, -1.57079633, -0.78539816])))
        True
    """

    Nx21 = Nx // 2 + 1
    Ny21 = Ny // 2 + 1

    Nx2 = Nx + 2
    Ny2 = Ny + 2

    delkx = (2.0 * np.pi) / (Nx * dx)
    delky = (2.0 * np.pi) / (Ny * dy)

    kx = np.zeros(Nx2)
    ky = np.zeros(Ny2)

    for i in range(1, Nx21 + 1):
        fk1 = (i - 1) * delkx
        kx[i - 1] = fk1
        kx[Nx2 - i - 1] = -fk1

    for j in range(1, Ny21 + 1):
        fk2 = (j - 1) * delky
        ky[j - 1] = fk2
        ky[Ny2 - j - 1] = -fk2

    kx = kx[:Nx]
    ky = ky[:Ny]

    k2 = np.zeros((Nx, Ny))
    k2_anisotropy = np.zeros((Nx, Ny))
    k4 = np.zeros((Nx, Ny))
    k4_anisotropy = np.zeros((Nx, Ny))
    k6 = np.zeros((Nx, Ny))
    eta_xx = eta[0]
    eta_yy = eta[1]
    eta_xy = eta[2]

    for i in range(Nx):
        for j in range(Ny):
            k2[i, j] = kx[i] ** 2 + ky[j] ** 2
            k2_anisotropy[i, j] = (
                eta_xx * kx[i] ** 2 + eta_yy * ky[j] ** 2 + 2 * eta_xy * kx[i] * ky[j]
            )
            k4_anisotropy[i, j] = k2[i, j] * k2_anisotropy[i, j]
            k4[i, j] = k2[i, j] ** 2
            k6[i, j] = k4[i, j] * k2[i, j]

    return kx, ky, k2, k4, k6, k2_anisotropy, k4_anisotropy


if __name__ == "__main__":
    import doctest

    doctest.testmod()
# %%
