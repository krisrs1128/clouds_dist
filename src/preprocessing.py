import numpy as np
import torch



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
        assert a.shape[0] == b.shape[0], "a.shape[0] != b.shape[0] ({} vs {})".format(
            a.shape[0], b.shape[0]
        )
        return b.view((b.shape[0], 1, 1)).expand(*a.shape)
    elif len(a.shape) == 4:
        assert a.shape[1] == b.shape[0], "a.shape[1] != b.shape[0] ({} vs {})".format(
            a.shape[1], b.shape[0]
        )
        return b.view((1, b.shape[0], 1, 1)).expand(*a.shape)
    raise ValueError(
        "First argument should have 3 or 4 dimensions, not {} ({})".format(
            len(a.shape), a.shape
        )
    )


class Standardize:
    def set_stats(self, stats):
        self.means, _, self.ranges, _ = stats

    def __call__(self, sample):
        for k in sample:
            mean_expand = expand_as(sample[k], self.means[k])
            range_expand = expand_as(sample[k], self.ranges[k])
            sample[k] = (sample[k] - mean_expand) / range_expand
        return sample


class CropBackground:
    def get_first_non_nan_index(self, array, axis):
        return torch.where((~torch.isnan(array)).sum(axis=0).sum(axis=axis - 1) > 0)[0][
            0
        ]

    def __call__(self, sample):

        metos = sample["metos"]
        crop_x = self.get_first_non_nan_index(metos, 1)
        crop_y = self.get_first_non_nan_index(metos, 2)
        result = {k: v[:, crop_y:-crop_y, crop_x:-crop_x] for k, v in sample.items()}
        return result


class Rescale:
    def __init__(self, resolution):
        self.res = resolution
        return

    def __call__(self, sample):
        upsample = torch.nn.UpsamplingNearest2d(size=self.res)
        rescaled_sample = {
            k: upsample(v.unsqueeze(0)).squeeze(0) for k, v in sample.items()
        }
        return rescaled_sample


class ReplaceNans:
    def __call__(self, sample):
        sample["real_imgs"][torch.isnan(sample["real_imgs"])] = -1
        sample["real_imgs"][torch.isinf(sample["real_imgs"])] = 1
        sample["metos"][torch.isnan(sample["metos"])] = 0.0
        sample["metos"][torch.isinf(sample["metos"])] = 0.0
        return sample


class SquashChannels:
    def __call__(self, sample):
        sample["metos"] = torch.cat(
            [
                sample["metos"][:10, :, :].mean(dim=0).unsqueeze(0),
                sample["metos"][10:20, :, :].mean(dim=0).unsqueeze(0),
                sample["metos"][20:30, :, :].mean(dim=0).unsqueeze(0),
                sample["metos"][30:40, :, :].mean(dim=0).unsqueeze(0),
                sample["metos"][40:44, :, :],
            ],
            dim=0,
        )
        return sample


class CropInnerSquare:
    def get_crop_index(self, img):
        assert (
            img.shape[0] == 3
        ), "Expected channels as first dim but got shape {}".format(img.shape)
        i = 0
        while any(torch.isnan(img[:, i, i])):
            i += 1
        assert i > 0, "Error in CropInnerSquare: i is 0"
        assert i <= img.shape[-1] // 2, "Error in CropInnerSquare: i is {}".format(i)
        return i

    def __call__(self, sample):
        i = self.get_crop_index(sample["real_imgs"])
        for name, tensor in sample.items():
            sample[name] = tensor[:, i:-i, i:-i]
        return sample


class Quantize:
    def set_stats(self, stats):
        _, _, _, self.quantiles = stats
        self.noq = {}
        for k in ["real_imgs", "metos"]:
            self.noq[k] = self.quantiles[k].shape[1]

    def __call__(self, sample):
        result = {"real_imgs": [], "metos": []}
        for key, tensor in sample.items():
            for c in range(tensor.shape[0]):
                channel_quantile = self.quantiles[key][c]

                nan_free_tensor = tensor[c][~torch.isnan(tensor[c])]
                digitized_flattened = torch.tensor(np.digitize(nan_free_tensor, channel_quantile), dtype=torch.float)

                digitized_reshaped = torch.zeros(tensor[c].shape)
                digitized_reshaped[torch.isnan(tensor[c])] = np.nan
                digitized_reshaped[~torch.isnan(tensor[c])] = digitized_flattened

                result[key] += [digitized_reshaped]
            result[key] = torch.stack(result[key])
        return result
