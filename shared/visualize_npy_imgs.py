import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# ------------------------------------------------------
# -----  Script to visualize (png) all npy arrays  -----
# -----  in a directory                            -----
# ------------------------------------------------------


def scale(x):
    x = x - x.min()
    return x / x.max()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", default=".")
    opts = parser.parse_args()

    npys = Path(opts.dir).resolve().glob("*.npy")

    for npy in npys:
        arr = np.load(str(npy))
        plt.imsave(
            npy.parent / (npy.name + "img.png"),
            np.concatenate(
                [
                    scale(arr[:, :256, :]),
                    scale(arr[:, 256:512, :]),
                    scale(arr[:, 512:, :]),
                ],
                axis=1,
            ),
        )
