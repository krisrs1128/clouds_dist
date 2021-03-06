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


class ClipReflectance:
    """
    np.quantile(ref_680, 0.99) >>> 0.6791079149527586
    np.quantile(ref_551, 0.99) >>> 0.6702599556531738
    np.quantile(ref_443, 0.99) >>> 0.7186933615126095

    np.quantile(ref_680, 0.999) >>> 0.8614172062075665
    np.quantile(ref_551, 0.999) >>> 0.8580325935816656
    np.quantile(ref_443, 0.999) >>> 0.9071594989032588
    """

    def __init__(self, ref=0.9):
        try:
            ref = float(ref)
        except TypeError:
            raise TypeError(
                "ClipReflectance: ref can't be broadcasted to float ( {} )".format(ref)
            )
        self.ref = ref

    def __call__(self, sample):
        sample["real_imgs"][sample["real_imgs"] > self.ref] = self.ref
        return sample


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


class Resize:
    def __init__(self, resolution):
        self.res = resolution
        return

    def __call__(self, sample):
        upsample = torch.nn.UpsamplingNearest2d(size=self.res)
        resized_sample = {
            k: upsample(v.unsqueeze(0)).squeeze(0) for k, v in sample.items()
        }
        return resized_sample


class ReplaceNans:

    def __init__(self, nan_value):
        self.nan_value = nan_value

    def set_stats(self, stats):
        self.means, self.stds, _, _ = stats

    def __call__(self, sample):
        sample["real_imgs"][torch.isnan(sample["real_imgs"])] = -1 #Black pixels
        for c in range(sample["metos"].shape[0]):
            if self.nan_value == "Standardize":
                sample["metos"][c][torch.isnan(sample["metos"][c])] = -3
            elif self.nan_value == "Quantize":
                sample["metos"][c][torch.isnan(sample["metos"][c])] = -1.1
            else:
                sample["metos"][c][torch.isnan(sample["metos"][c])] = self.means["metos"][c] - 3 * self.stds["metos"][c]

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
            if key == "metos":
                for c in range(tensor.shape[0]):
                    channel_quantile = self.quantiles[key][c]

                    nan_free_tensor = tensor[c][~torch.isnan(tensor[c])]
                    digitized_flattened = torch.tensor(
                        np.digitize(nan_free_tensor, channel_quantile),
                        dtype=torch.float,
                    )

                    digitized_reshaped = torch.zeros(tensor[c].shape)
                    digitized_reshaped[torch.isnan(tensor[c])] = np.nan
                    digitized_reshaped[~torch.isnan(tensor[c])] = digitized_flattened

                    result[key] += [digitized_reshaped]
                result[key] = torch.stack(result[key])
                result[key] = 2 * (
                    (result[key] / self.noq[key]) - 0.5
                )  # rescale to [-1,1)
            else:
                result[key] = tensor
        return result
