#!/usr/bin/env python
from comet_ml import Experiment, OfflineExperiment
from datetime import datetime
from pathlib import Path
from src.data import EarthData
from src.gan import GAN
from torch import optim
from torch.utils import data

from addict import Dict

import json
import time
import subprocess
import numpy as np
import os
import torch
import torch.nn as nn

# from tensorboardX import SummaryWriter
import multiprocessing
import argparse


def merge_defaults(opts, conf_path):
    print("Loading params from", conf_path)
    with open(conf_path, "r") as f:
        result = json.load(f)
    for group in ["model", "train"]:
        for k, v in opts[group].items():
            result[group][k] = v

    return Dict(result)


class gan_trainer:
    def __init__(self, opts, comet_exp=None, output_dir=".", n_epochs=50, verbose=1):
        self.opts = opts
        self.trainset = EarthData(
            self.opts.train.datapath,
            n_in_mem=self.opts.train.n_in_mem or 50,
            load_limit=self.opts.train.load_limit or -1,
        )
        self.trial_number = 0
        self.n_epochs = n_epochs
        self.start_time = datetime.now()
        self.verbose = verbose

        timestamp = self.start_time.strftime("%Y_%m_%d_%H_%M_%S")
        self.timestamp = timestamp

        self.results = []

        self.exp = comet_exp
        self.output_dir = Path(output_dir)

        if self.verbose > 0:
            print("-------------------------")
            print("--       Params        --")
            print("-------------------------")
            for o, d in opts.items():
                print(o)
                for k, v in d.items():
                    print("{:<30}: {:<30}".format(str(k), str(v)))
            print()
        self.make_directories()

    def save(self, epoch=0):
        torch.save(self.gan.state_dict(), str(self.ckptdir / f"gan_{epoch}.pt"))
        torch.save(self.gan.state_dict(), str(self.ckptdir / f"gan_latest.pt"))

    def make_directories(self):
        self.ckptdir = self.output_dir / "checkpoints"
        self.imgdir = self.output_dir / "images"

        self.ckptdir.mkdir(parents=True, exist_ok=True)
        self.imgdir.mkdir(exist_ok=True)

    def run_trail(self):
        if self.exp:
            self.exp.log_parameters(self.opts.train)
            self.exp.log_parameters(self.opts.model)

        # initialize objects
        self.make_directories()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.opts.train.batch_size,
            shuffle=False,
            num_workers=self.opts.train.get("num_workers", 3),
        )

        self.gan = GAN(**self.opts.model, device=self.device).to(self.device)
        self.g = self.gan.g
        self.d = self.gan.d

        # train using "regress then GAN" approach
        val_loss = self.train(
            self.opts.train.n_epochs,
            self.opts.train.lr_d,
            self.opts.train.lr_g1,
            lambda_gan=0,
            lambda_L1=1,
        )
        return {"loss": val_loss, "opts": self.opts}

    def get_noise_tensor(self, shape):
        b, h, w = shape[0], shape[2], shape[3]
        Ctot = self.opts.model.Cin + self.opts.model.Cout
        input_tensor = torch.FloatTensor(b, Ctot, h, w)
        input_tensor.uniform_(-1, 1)
        return input_tensor

    def train(self, n_epochs, lr_d=1e-2, lr_g=1e-2, lambda_gan=0.01, lambda_L1=1):
        # initialize trial
        d_optimizer = optim.Adam(self.d.parameters(), lr=lr_d)
        g_optimizer = optim.Adam(self.g.parameters(), lr=lr_g)

        L1 = nn.L1Loss()
        MSE = nn.MSELoss()
        device = self.device
        if self.verbose > 0:
            print("-------------------------")
            print("--  Starting training  --")
            print("-------------------------")
        times = []
        start_time = time.time()
        for epoch in range(n_epochs):
            torch.cuda.empty_cache()
            self.gan.train()  # train mode
            etime = time.time()
            for i, (coords, real_img, metos_data) in enumerate(self.trainloader):
                if i > (self.opts.train.early_break_epoch or 1e9):
                    break
                stime = time.time()
                if i == 0 and self.verbose > 0:
                    print("\n\nLoading time: {:.3f}".format(stime - etime))

                shape = metos_data.shape

                self.input_tensor = self.get_noise_tensor(shape)
                self.input_tensor[:, : self.opts.model.Cin, :, :] = metos_data
                self.input_tensor = self.input_tensor.to(device)

                real_img = real_img.to(device)
                generated_img = self.g(self.input_tensor)

                real_prob = self.d(real_img)
                fake_prob = self.d(generated_img.detach())

                real_target = torch.ones(real_prob.shape, device=device)
                fake_target = torch.zeros(fake_prob.shape, device=device)

                d_optimizer.zero_grad()
                d_loss = 0.5 * (
                    MSE(fake_prob, fake_target) + MSE(real_prob, real_target)
                )
                d_loss.backward(retain_graph=True)
                d_optimizer.step()

                g_optimizer.zero_grad()
                fake_prob = self.d(generated_img)
                L1_loss = L1(generated_img, real_img)
                gan_loss = MSE(fake_prob, real_target)

                g_loss = lambda_gan * gan_loss + lambda_L1 * L1_loss
                g_loss.backward()
                g_optimizer.step()
                if self.exp:
                    self.exp.log_metrics(
                        {
                            "train/g_loss": g_loss.item(),
                            "train/d_loss": d_loss.item(),
                            "train/L1_loss": L1_loss.item(),
                        }
                    )
                t = time.time()
                times.append(t - stime)
                times = times[-100:]
                if self.verbose > 0:
                    ep_str = "epoch:{}/{} step {}/{} d_loss:{:0.4f} l1:{:0.4f} g_loss:{:0.4f} | "
                    ep_str += "t/step {:.1f} | t/ep {:.1f} | t {:.1f}"
                    print(
                        ep_str.format(
                            epoch + 1,
                            n_epochs,
                            i + 1,
                            len(self.trainloader),
                            d_loss.item(),
                            L1_loss.item(),
                            g_loss.item(),
                            np.mean(times),
                            t - etime,
                            t - start_time,
                        ),
                        end="\r",
                    )

            # ------------
            # END OF EPOCH
            # ------------

            # output sample images
            input_tensor = self.get_noise_tensor(shape)
            input_tensor[:, : self.opts.model.Cin, :, :] = metos_data
            input_tensor = input_tensor.to(device)

            generated_img = self.g(input_tensor)

            for i in range(input_tensor.shape[0]):
                # concatenate verticaly 3 metos, generated clouds, ground truth clouds
                tmp_tensor = input_tensor[i, 22:25].clone().detach()
                tmp_tensor -= tmp_tensor.min()
                tmp_tensor /= tmp_tensor.max()
                imgs = torch.cat((tmp_tensor, generated_img[i], real_img[i]), 1)
                imgs_cpu = imgs.cpu().detach().numpy()
                imgs_cpu = np.swapaxes(imgs_cpu, 0, 2)
                np.save(str(self.imgdir / f"imgs_{epoch}_{i}.npy"), imgs_cpu)
                if self.exp:
                    try:
                        self.exp.log_image(imgs_cpu, name=f"imgs_{epoch}_{i}")
                    except Exception as e:
                        print(f"\n{e}\n")

            self.save(epoch)


if __name__ == "__main__":

    scratch = os.environ.get("SCRATCH") or os.path.join(
        os.environ.get("HOME"), "scratch"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--message",
        type=str,
        default="",
        help="Add a message to the commet experiment",
    )
    parser.add_argument(
        "-f",
        "--offline",
        default=False,
        action="store_true",
        help="use an offline or standard experiment",
    )
    parser.add_argument(
        "-c",
        "--conf_name",
        type=str,
        default="defaults",
        help="name of conf file in config/ | may ommit the .json extension",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="where the run's data should be stored ; used to resume",
    )
    parser.add_argument("-n", "--no_exp", default=False, action="store_true")
    opts = parser.parse_args()

    conf_path = opts.conf_name
    output_path = Path(opts.output_dir)

    if not output_path.exists():
        output_path.mkdir()

    if not Path(conf_path).exists():
        conf_name = conf_path
        if not conf_name.endswith(".json"):
            conf_name += ".json"
        conf_path = Path("config") / conf_name
        assert conf_path.exists()

    params = merge_defaults({"model": {}, "train": {}}, conf_path)

    data_path = params.train.datapath.split("/")
    for i, d in enumerate(data_path):
        if "$" in d:
            data_path[i] = os.environ.get(d.replace("$", ""))
    params.train.datapath = "/".join(data_path)

    assert Path(params.train.datapath).exists()
    assert (Path(params.train.datapath) / "imgs").exists()
    assert (Path(params.train.datapath) / "metos").exists()
    # print("Make sure you are using proxychains so that comet has internet access")

    scratch = str(Path(scratch) / "comets")

    if opts.no_exp:
        exp = None
    else:
        if opts.offline:
            exp = OfflineExperiment(offline_directory=str(output_path))
        else:
            exp = Experiment()

    exp.log_parameter("__message", opts.message)

    trainer = gan_trainer(params, exp, output_path)

    result = trainer.run_trail()

    trainer.exp.end()
    if opts.offline and not opts.no_exp:
        subprocess.check_output(
            [
                "bash",
                "-c",
                "python -m comet_ml.scripts.upload {}".format(
                    str(Path(output_path).resolve() / (trainer.exp.id + ".zip"))
                ),
            ]
        )
