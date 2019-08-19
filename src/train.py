#!/usr/bin/env python
from comet_ml import OfflineExperiment
from datetime import datetime
from pathlib import Path
from src.data import EarthData
from src.gan import GAN
from torch import optim
from torch.utils import data

import json

import multiprocessing
import numpy as np
import os
import time
import torch
import torch.nn as nn

from tensorboardX import SummaryWriter
import multiprocessing


def merge_defaults(opts, defaults_path):
    result = json.load(open(defaults_path, "r"))
    for group in ["model", "train"]:
        for k, v in opts[group].items():
            result[group][k] = v

    return result


class gan_trainer:
    def __init__(self, opts, comet_exp=None, n_epochs=50):
        self.opts = opts
        self.trainset = EarthData(self.opts["train"]["datapath"], n_in_mem=50)
        self.trial_number = 0
        self.n_epochs = n_epochs
        self.start_time = datetime.now()

        timestamp = self.start_time.strftime("%Y_%m_%d_%H_%M_%S")
        self.timestamp = timestamp

        self.runname = "unet_gan_10level"
        self.runpath = Path("output") / self.runname / f"output_{timestamp}"
        self.results = []

        self.exp = comet_exp

    def make_directories(self):
        self.trialdir = self.runpath / f"trial_{self.trial_number}"
        self.logdir = self.trialdir / "log"
        self.imgdir = self.trialdir / "images"

        self.runpath.mkdir(parents=True, exist_ok=True)
        self.trialdir.mkdir(exist_ok=True)
        self.logdir.mkdir(exist_ok=True)
        self.imgdir.mkdir(exist_ok=True)

    def run_trail(self):
        if self.exp:
            self.exp.log_parameters(self.opts)

        # initialize objects
        self.make_directories()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.opts["train"]["batch_size"],
            shuffle=True,
            num_workers=min((multiprocessing.cpu_count() // 2, 10)),
        )

        self.gan = GAN(**self.opts["model"]).to(self.device)
        self.g = self.gan.g
        self.d = self.gan.d

        # train using "regress then GAN" approach
        val_loss = self.train(
            self.opts["train"]["n_epoch_regress"],
            self.opts["train"]["lr_d"],
            self.opts["train"]["lr_g1"],
            lambda_gan=0,
            lambda_L1=1,
        )
        return {"loss": val_loss, "opts": self.opts}

    def get_noise_tensor(self, shape):
        b, h, w = shape[0], shape[2], shape[3]
        Ctot = self.opts["model"]["Cin"] + self.opts["model"]["Cout"]
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

        print("-------------------------")
        print("--  Starting training  --")
        print("-------------------------")

        for epoch in range(n_epochs):
            torch.cuda.empty_cache()
            self.gan.train()  # train mode

            for i, (coords, real_img, metos_data) in enumerate(self.trainloader):

                shape = metos_data.shape

                input_tensor = self.get_noise_tensor(shape)
                input_tensor[:, : self.opts["model"]["Cin"], :, :] = metos_data
                input_tensor = input_tensor.to(device)

                real_img = real_img.to(device)
                generated_img = self.g(input_tensor)

                real_prob = self.d(real_img)
                fake_prob = self.d(generated_img)

                real_target = torch.ones(real_prob.shape, device=device)
                fake_target = torch.zeros(fake_prob.shape, device=device)

                d_optimizer.zero_grad()
                d_loss = 0.5 * (
                    MSE(fake_prob, fake_target) + MSE(real_prob, real_target)
                )
                d_loss.backward(retain_graph=True)
                d_optimizer.step()

                g_optimizer.zero_grad()
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

                print(
                    "trail:{} epoch:{}/{} iteration {}/{} train/d_loss:{:0.4f} train/L1_loss:{:0.4f} train/g_loss:{:0.4f}".format(
                        self.trial_number,
                        epoch + 1,
                        n_epochs,
                        i + 1,
                        len(self.trainloader),
                        d_loss.item(),
                        L1_loss.item(),
                        g_loss.item(),
                    ),
                    end="\r",
                )

            # ------------
            # END OF EPOCH
            # ------------

            # output sample images
            input_tensor = self.get_noise_tensor(shape)
            input_tensor[:, : self.opts["model"]["Cin"], :, :] = metos_data
            input_tensor = input_tensor.to(device)

            generated_img = self.g(input_tensor)

            # write out the model architechture
            imgs = torch.cat(
                (input_tensor[0, 22:25], generated_img[0, 0:3], real_img[0, 0:3]), 1
            )  # concatenate verticaly 3 metos, generated clouds, ground truth clouds

            for i in range(input_tensor.shape[0]):
                if i > 0:
                    imgs = torch.cat(
                        (
                            input_tensor[i, 22:25],
                            generated_img[i, 0:3],
                            real_img[i, 0:3],
                        ),
                        1,
                    )
                imgs_cpu = imgs.cpu().detach().numpy()
                imgs_cpu = np.swapaxes(imgs_cpu, 0, 2)
                if self.exp:
                    self.exp.log_image(imgs_cpu, str(self.imgdir / f"imgs{i}_{epoch}"))

        torch.save(self.gan.state_dict(), str(self.trialdir / "gan.pt"))


if __name__ == "__main__":
    scratch = os.environ.get("SCRATCH") or "~/scratch/"
    scratch = str(Path(scratch) / "comets")
    exp = OfflineExperiment(
        project_name="clouds", workspace="vict0rsch", offline_directory=scratch
    )

    params = merge_defaults({"model": {}, "train": {}}, "config/defaults.json")

    trainer = gan_trainer(params, exp)

    result = trainer.run_trail()

    trainer.exp.end()
    multiprocessing.check_output(
        [
            "bash",
            "-c",
            "python -m comet_ml.scripts.upload {}".format(
                str(Path(scratch).resolve() / (trainer.exp.id + ".zip"))
            ),
        ]
    )
