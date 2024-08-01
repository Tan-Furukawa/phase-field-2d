# %%
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def get_matrix_image(
    mat: NDArray,
    vmin: float = 0,
    vmax: float = 1,
    isnorm: bool = True,
    theme: str = "Greys",
    show: bool = True
) -> None:
    """plot the phase field result matrix

    Args:
        mat (NDArray): numpy 2d NDArray
        vmin (float, optional): min value of plot. The value below vmin is displayed as red blue. Defaults to 0.
        vmax (float, optional): max value of plot. The value above vmax is displayed as red color. Defaults to 1.
        isnorm (bool, optional): plotted as [vmin, vmax] range. Defaults to True.
        theme (str, optional): the theme color. see matplotlib.pyplot.get_cmap. Defaults to "Greys".
    """
    cmap = plt.get_cmap(theme)
    # カラーマップの正規化
    norm = Normalize(vmin=vmin, vmax=vmax)
    # con < 0 のセルを赤色に変更
    cmap.set_under("blue")  # below lower limit blue
    cmap.set_over("red")  # above upper limit red
    if isnorm:
        plt.imshow(mat, cmap=cmap, norm=norm)
        # plt.show()
    else:
        plt.imshow(mat, cmap=cmap)
        # plt.show()
    if show:
        # plt.savefig("tmp.svg")
        plt.colorbar()
        plt.show()

def plot_con_hist (con: NDArray)->None:
    """plot composition histogram

    Args:
        con (NDArray): composition matrix

    Example:
        >>> import numpy as np
        >>> x = np.array([[1,2,3,4,5],[2,3,4,5,6]])
        >>> plot_con_hist(x)
    """
    con = con.reshape(1, -1)[0]
    plt.hist(con, bins=200)
    plt.show()

if __name__ == "__main__":
    import numpy as np
    x = np.array([[1,2,3,4,5],[2,3,4,5,6]])
    plot_con_hist(x)

# %%
