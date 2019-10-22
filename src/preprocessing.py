import numpy as np
from src.data import EarthData
import torch
from torchvision import transforms


def expand_as(a, b):
    """Repeat vector b that gives 1 value per channel so that it
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


class Rescale:
    def __init__(self, stats):
        self.means, self.ranges = stats

    def __call__(self, sample):
        for k in sample:
            mean_expand = expand_as(sample[k], self.means[k])
            range_expand = expand_as(sample[k], self.ranges[k])
            sample[k] = (sample[k] - mean_expand) / range_expand
        return sample


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
    def __init__(self, crop_size=20):
        self.crop_size = crop_size
        return

    def __call__(self, sample):
        dim = list(sample.items())[0][1].size()[-1]
        upsample = torch.nn.UpsamplingNearest2d(
            scale_factor=dim / (dim - 2 * self.crop_size)
        )
        crop_result = {
            k: v[:, self.crop_size : -self.crop_size, self.crop_size : -self.crop_size]
            if k == "real_imgs"
            else v
            for k, v in sample.items()
        }
        zoom_result = {
            k: upsample(v.unsqueeze(0)).squeeze() if k == "real_imgs" else v
            for k, v in crop_result.items()
        }
        return zoom_result


class ReplaceNans:
    def __init__(self):
        return

    def __call__(self, sample):
        sample["real_imgs"][torch.isnan(sample["real_imgs"])] = -1
        sample["real_imgs"][torch.isinf(sample["real_imgs"])] = 1
        sample["metos"][torch.isnan(sample["metos"])] = 0.0
        sample["metos"][torch.isinf(sample["metos"])] = 0.0
        return sample


class SquashChannels:
    def __init__(self):
        return

    def __call__(self, sample):
        sample["metos"] = torch.cat(
            [
                sample["metos"][:, :10, :, :].mean(dim=1).unsqueeze(1),
                sample["metos"][:, 10:20, :, :].mean(dim=1).unsqueeze(1),
                sample["metos"][:, 20:30, :, :].mean(dim=1).unsqueeze(1),
                sample["metos"][:, 30:40, :, :].mean(dim=1).unsqueeze(1),
                sample["metos"][:, 40:44, :, :],
            ],
            dim=1,
        )
        return sample


class CropInnerSquare:
    def __init__(self):
        self.i = None
        return

    def get_crop_index(self, img):
        assert (
            img.shape[0] == 3
        ), "Expected channels as first dim but got shape {}".format(img.shape)
        if self.i is not None:
            return self.i
        self.i = 0
        while any(torch.isnan(img[:, self.i, self.i])):
            self.i += 1
        assert self.i > 0, "Error in CropInnerSquare: i is 0"
        assert self.i <= img.shape[-1] // 2, "Error in CropInnerSquare: i is {}".format(
            self.i
        )
        return self.i

    def __call__(self, sample):
        i = self.get_crop_index(sample["real_imgs"])
        for name, tensor in sample.items():
            sample[name] = tensor[:, i:-i, i:-i]
        return sample
