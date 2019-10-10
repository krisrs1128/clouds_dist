from src.data import EarthData
import numpy as np
import torch


class Rescale:
    def __init__(self, data_path, batch_size, num_workers=3, verbose=1):
        self.data_path = data_path
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataset = EarthData(data_dir=self.data_path)

        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        self.means, self.ranges = self.get_stats()

    def expand_as(self, a, b):
        """Repeat a vector b that gives 1 value per channel so that it
        can be used in elementwise computations with a. a.shape[1] should
        be the same as b.shape[0]

        Args:
            a (torch.Tensor): Input which shape should be matched
            b (torch.Tensor): Channel-wise vector

        Raises:
            ValueError: a should have either 3 or 4 dimensions

        Returns:
            torch.Tensor: repeated Tensor (t) from b into a's shape.
        For any i, j, k t[i, :, j, k] == b (if a has 4 dimensions)
                              t[:, i, j] == b (if a has 3 dimensions)
        """
        assert len(b.shape) == 1
        if len(a.shape) == 3:
            assert a.shape[0] == b.shape[0], "a.shape[0] does not match b.shape[0]"
            return b.view((b.shape[0], 1, 1)).expand(*a.shape)
        elif len(a.shape) == 4:
            assert a.shape[1] == b.shape[0], "a.shape[1] does not match b.shape[0]"
            return b.view((1, b.shape[0], 1, 1)).expand(*a.shape)
        raise ValueError(
            "First argument should have 3 or 4 dimensions, not {} ({})".format(
                len(a.shape), a.shape
            )
        )

    def __call__(self, sample):
        img_mean_expand = self.expand_as(sample["real_imgs"], self.means["real_imgs"])
        img_range_expand = self.expand_as(sample["real_imgs"], self.ranges["real_imgs"])
        metos_mean_expand = self.expand_as(sample["metos"], self.means["metos"])
        metos_range_expand = self.expand_as(sample["metos"], self.ranges["metos"])

        sample["real_imgs"] = (sample["real_imgs"] - img_mean_expand) / img_range_expand
        sample["metos"] = (sample["metos"] - metos_mean_expand) / metos_range_expand

        sample["real_imgs"][torch.isnan(sample["real_imgs"])] = 0.0
        sample["real_imgs"][torch.isinf(sample["real_imgs"])] = 0.0
        sample["metos"][torch.isnan(sample["metos"])] = 0.0
        sample["metos"][torch.isinf(sample["metos"])] = 0.0
        return sample

    def get_stats(self):
        maxes = {}
        mins = {}
        means = {}
        for i, batch in enumerate(self.data_loader):
            torch.cuda.empty_cache()
            for k, v in batch.items():
                v = v.to(self.device)
                if i == 0:
                    means[k] = v.mean(dim=(0, 2, 3))
                    maxes[k] = v.max(0)[0].max(1)[0].max(1)[0]
                    mins[k] = v.min(0)[0].min(1)[0].min(1)[0]

                else:
                    n = i * self.batch_size
                    m = len(v)
                    means[k] *= n / (n + m)

                    means[k] += v.sum(dim=(0, 2, 3)) / (
                        (n + m) * v.shape[2] * v.shape[3]
                    )
                    cur_max = v.max(0)[0].max(1)[0].max(1)[0]
                    cur_min = v.min(0)[0].min(1)[0].min(1)[0]

                    maxes[k][maxes[k] < cur_max] = cur_max[maxes[k] < cur_max]
                    mins[k][mins[k] > cur_min] = cur_min[mins[k] > cur_min]

            if self.verbose > 0:
                print(
                    "\r get_stats --- progress: {:.1f}%".format(
                        (i + 1) / len(self.data_loader) * 100
                    ),
                    end="",
                )

        print()
        # calculate ranges and avoid cuda multiprocessing by bringing tensors back to cpu
        stats = (
            {k: v.to("cpu") for k, v in means.items()},
            {k: (maxes[k] - v).to("cpu") for k, v in mins.items()},
        )
        torch.cuda.empty_cache()
        return stats


class Crop:
    def __init__(self, crop_size=20):
        self.crop_size = crop_size
        return

    def __call__(self, sample):
        result = {
            k: v[:, self.crop_size : -self.crop_size, self.crop_size : -self.crop_size]
            for k, v in sample.items()
        }
        return result

class Zoom:
    def __init__(self, crop_size=21):
        self.crop_size = crop_size
        return

    def __call__(self, sample):
        dim = list(sample.items())[0][1].size()[-1]
        upsample = torch.nn.UpsamplingNearest2d(scale_factor=dim / (dim - 2 * self.crop_size))

        crop_result = {
            k: v[:, self.crop_size : -self.crop_size, self.crop_size : -self.crop_size]
            for k, v in sample.items()
        }
        zoom_result = {k: upsample(v.unsqueeze(0)).squeeze()
                       for k, v in crop_result.items()}
        return zoom_result
