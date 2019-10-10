from pathlib import Path
import shutil
import argparse
from numpy.random import permutation

if __name__ == "__main__":
    # --------------------------------------------------------------------
    # -----   Parse Arguments ; Typical Use:                         -----
    # -----   $ python sample_dataset.py -d clouds_full -o . -n 50   -----
    # --------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", type=str, default=".")
    parser.add_argument("-o", "--output_path", type=str, default=".")
    parser.add_argument("-n", "--n_images", type=int, default=50)
    opts = parser.parse_args()

    # ------------------------------------------------------------------
    # -----   data_path must contain imgs/ and metos/ subfolders   -----
    # ------------------------------------------------------------------
    data_path = Path(opts.data_path).resolve()
    assert all(
        d in [a.name for a in data_path.iterdir() if a.is_dir()]
        for d in ["imgs", "metos"]
    ), "data_path ({}) should have imgs/ and metos/ subfolders".format(str(data_path))

    imdir = data_path / "imgs"
    metdir = data_path / "metos"

    # ------------------------------------------------------------
    # -----   list metos and data samples in the data_path   -----
    # ------------------------------------------------------------
    imgs = list(imdir.glob("*.npz"))
    metos = {m.stem.split("_Collocated_MERRA2")[0]: m for m in metdir.glob("*.npz")}

    # -------------------------------------
    # -----   Sample imgs and metos   -----
    # -------------------------------------
    perm = permutation(len(imgs))[: opts.n_images]
    imgs = [imgs[i] for i in perm]
    metos = [metos[i.stem] for i in imgs]

    # -------------------------------------------------------------------------
    # -----   Create new directory called "mini_clouds_{opts.n_images}"   -----
    # -----   in output_path output_path and subfolders imgs/ and         -----
    # -----   metos/                                                      -----
    # -------------------------------------------------------------------------
    output_path = Path(opts.output_path) / "mini_clouds_{}".format(opts.n_images)
    output_path.mkdir()
    new_imdir = output_path / "imgs"
    new_metdir = output_path / "metos"
    new_imdir.mkdir()
    new_metdir.mkdir()

    # ----------------------------------------
    # -----   Copy data to output_path   -----
    # ----------------------------------------
    for k, im in enumerate(imgs):
        print(k, end="\r")
        shutil.copyfile(im, new_imdir / im.name)
    print("Copied {} images".format(len(imgs)))

    for k, met in enumerate(metos):
        print(k, end="\r")
        shutil.copyfile(met, new_metdir / met.name)
    print("Copied {} metos".format(len(metos)))
