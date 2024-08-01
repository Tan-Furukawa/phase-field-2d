# %%
import numpy as np
from numpy.typing import NDArray


def make_initial_distribution(
    Nx: int, Ny: int, c0: NDArray[np.float64] | float, noise: float
) -> NDArray[np.float64]:
    """Returns new array of initial composition distribution. The each element is determined by Gaussian noise.

    ## Args:
        Nx (int): matrix size (height)
        Ny (int): matrix size (width)
        c0 (NDArray[np.float64]): bulk composition
        noise (float): 1 sigma noise

    ## Returns:
        NDArray[np.float64]: initial composition distribution

    ## Examples:
        >>> import numpy as np
        >>> make_initial_distribution(3,3,0.5,0.1)
        np.array([[0.48176481, 0.5446179 , 0.52796401], [0.53156282, 0.53240941, 0.46879055], [0.4576655 , 0.52234256, 0.46802454]]) # random

    """

    con = np.zeros((Nx, Ny))

    rng = np.random.default_rng(seed=123)
    con = c0 + noise * (0.5 - rng.random((Nx, Ny)))
    # np.random.seed(123)
    # con = c0 + noise * (0.5 - np.random.rand(Nx, Ny))

    return con


if __name__ == "__main__":
    import doctest
    doctest.testmod()

# %%
