from src.data import EarthData
import numpy as np
import torch


class Rescale:
    def __init__(
        self, data_path, n_in_mem=50, num_workers=3, with_stats=True, verbose=1
    ):
        self.n_in_mem = n_in_mem
        self.data_path = data_path
        self.num_workers = num_workers
        self.verbose = verbose
        self.with_stats = with_stats

        self.dataset = EarthData(data_dir=self.data_path, n_in_mem=self.n_in_mem)

        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.n_in_mem,
            shuffle=False,
            num_workers=self.num_workers,
        )
        if with_stats:
            self.means, self.maxes, self.mins = self.get_stats(self.data_loader)
        else:
            self.means, self.maxes, self.mins = (
                {"coords": 0, "real_imgs": 0, "metos": 0},
                {"coords": 1, "real_imgs": 1, "metos": 1},
                {"coords": 0, "real_imgs": 0, "metos": 0},
            )

    def __call__(self, sample):
        sample["coords"] = (sample["coords"] - self.means["coords"]) / (
            self.maxes["coords"] - self.mins["coords"]
        )
        sample["real_imgs"] = (sample["real_imgs"] - self.means["real_imgs"]) / (
            self.maxes["real_imgs"] - self.mins["real_imgs"]
        )
        sample["metos"] = (sample["metos"] - self.means["metos"]) / (
            self.maxes["metos"] - self.mins["metos"]
        )
        if self.with_stats:
            sample["coords"] [np.isnan(sample["coords"] )] = 0.0
            sample["coords"] [np.isinf(sample["coords"] )] = 0.0
            sample["real_imgs"][np.isnan(sample["real_imgs"])] = 0.0
            sample["real_imgs"][np.isinf(sample["real_imgs"])] = 0.0
            sample["metos"][np.isnan(sample["metos"])] = 0.0
            sample["metos"][np.isinf(sample["metos"])] = 0.0
        return sample

    def get_stats(self, data_loader):
        means = {}
        maxes = {}
        mins = {}
        for i, batch in enumerate(data_loader):
            for k, v in batch.items():
                if i == 0:
                    means[k] = v.mean(dim=0)
                    maxes[k] = v.max(dim=0)[0]
                    mins[k] = v.min(dim=0)[0]
                else:
                    n = i * self.n_in_mem
                    m = len(v)
                    means[k] *= n / (n + m)
                    means[k] += v.sum(dim=0) / (n + m)
                    maxes[k][maxes[k] < v.max(dim=0)[0]] = v.max(dim=0)[0][
                        v.max(dim=0)[0] > maxes[k]
                    ]
                    mins[k][mins[k] > v.min(dim=0)[0]] = v.min(dim=0)[0][
                        mins[k] > v.min(dim=0)[0]
                    ]
            if self.verbose > 0:
                print(
                    "\r get_stats --- progress: {:.1f}%".format(
                        (i + 1) / len(self.data_loader) * 100
                    ),
                    end="",
                )
        print()
        return means, maxes, mins
