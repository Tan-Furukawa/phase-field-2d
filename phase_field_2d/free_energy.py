# %%
import cupy as cp

class CDArray(cp.ndarray): ...


def get_free_energy(con: CDArray, w: float) -> tuple[CDArray, CDArray]:
    """calculate free energy of the system. The free energy is formulated as x log x + (1-x) log (1-x) + w x (1-x) in this function.

    ## Args:
        con (CDArray): n * n cupy array
        w (float): float, Margulus parameter divided by RT

    ## Returns:
        tuple[CDArray, CDArray]: dg/dx, g where g is molar free energy, x is partial mole fraction.
    """

    # calculate dg/dx
    def get_dfdcon(con: CDArray | float) -> CDArray:
        dfdcon = w * (1 - 2 * con) + (cp.log(con) - cp.log(1 - con))
        return dfdcon

    min_c = 0.001
    max_c = 0.999

    dfdcon = cp.zeros(con.shape)
    dfdcon[con < min_c] = get_dfdcon(min_c)
    dfdcon[con > max_c] = get_dfdcon(max_c)
    dfdcon[cp.logical_and(min_c < con, con < max_c)] = get_dfdcon(
        con[cp.logical_and(min_c < con, con < max_c)]
    )

    g = w * con * (1 - con) + (con * cp.log(con) + (1 - con) * cp.log(1 - con))

    return dfdcon, g


# import numpy as np
# from numpy.typing import NDArray
# x: NDArray = np.array([1,2,3])
# get_free_energy(x, 0.2)
# %%
