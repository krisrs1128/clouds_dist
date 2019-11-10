import torch
from torchvision import transforms

from src.data import EarthData


def get_stats(opts, trsfs, verbose=0):

    should_compute_stats = False
    transforms_before_rescale = []
    for t in trsfs:
        if t.__class__.__name__ in ["Standardize", "Quantize"]:
            should_compute_stats = True
            break
        transforms_before_rescale.append(t)

    if not should_compute_stats:
        return None

    dataset = EarthData(
        data_dir=opts.data.path,
        load_limit=opts.data.load_limit or -1,
        transform=transforms.Compose(transforms_before_rescale),
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opts.train.batch_size,
        shuffle=False,
        num_workers=opts.data.num_workers,
    )

    maxes = {}
    mins = {}
    means = {}
    norm = {}
    for i, batch in enumerate(data_loader):
        torch.cuda.empty_cache()
        for k, v in batch.items():
            if i == 0:
                means[k] = torch.tensor(
                    [
                        v[:, c, :][~torch.isnan(v[:, c, :])].mean()
                        for c in range(v.shape[1])
                    ],
                    dtype=torch.float,
                )
                maxes[k] = torch.tensor(
                    [
                        (v[:, c, :][~torch.isnan(v[:, c, :])]).max()
                        for c in range(v.shape[1])
                    ],
                    dtype=torch.float,
                )
                mins[k] = torch.tensor(
                    [
                        (v[:, c, :][~torch.isnan(v[:, c, :])]).min()
                        for c in range(v.shape[1])
                    ],
                    dtype=torch.float,
                )
                norm[k] = torch.tensor(
                    [
                        v[:, c, :][~torch.isnan(v[:, c, :])].numel()
                        for c in range(v.shape[1])
                    ],
                    dtype=torch.float,
                )
            else:

                # count all elements that aren't nans per channel
                m = torch.tensor(
                    [
                        v[:, i, :][~torch.isnan(v[:, i, :])].numel()
                        for i in range(v.shape[1])
                    ],
                    dtype=torch.float,
                )
                means[k] *= norm[k] / (norm[k] + m)
                means[k] += torch.tensor(
                    [
                        (v[:, c, :][~torch.isnan(v[:, c, :])]).sum()
                        for c in range(v.shape[1])
                    ],
                    dtype=torch.float,
                ) / (norm[k] + m)

                norm[k] += m

                cur_max = torch.tensor(
                    [
                        v[:, i, :][~torch.isnan(v[:, i, :])].max()
                        for i in range(v.shape[1])
                    ],
                    dtype=torch.float,
                )
                cur_min = torch.tensor(
                    [
                        v[:, i, :][~torch.isnan(v[:, i, :])].min()
                        for i in range(v.shape[1])
                    ],
                    dtype=torch.float,
                )

                maxes[k][maxes[k] < cur_max] = cur_max[maxes[k] < cur_max]
                mins[k][mins[k] > cur_min] = cur_min[mins[k] > cur_min]

        if verbose > 0:
            print(
                "\r get_stats --- progress: {:.1f}%".format(
                    (i + 1) / len(data_loader) * 100
                ),
                end="",
            )

    print()
    # calculate ranges and avoid cuda multiprocessing by bringing tensors back to cpu
    stats = (
        {k: v.to("cpu") for k, v in means.items()},
        {k: (maxes[k] - v) for k, v in mins.items()},  # return range
    )
    return stats
