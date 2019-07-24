#!/usr/bin/env python
import os
import random
import time
from datetime import datetime
from pathlib import Path

from comet_ml import OfflineExperiment
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as distance
import torch
import torch.nn as nn
from hyperopt import STATUS_FAIL, STATUS_OK
from scipy import stats
from torch import optim
from torch.utils import data

from src.data import EarthData
from src.gan import GAN
from tensorboardX import SummaryWriter
import multiprocessing


class gan_trainer:
    def __init__(self, trainset, comet_exp=None, nepochs=50):

        self.trial_number = 0
        self.nepochs = nepochs
        self.start_time = datetime.now()

        timestamp = self.start_time.strftime("%Y_%m_%d_%H_%M_%S")
        self.timestamp = timestamp

        self.runname = "unet_gan_10level"
        self.runpath = Path("output") / self.runname / f"output_{timestamp}"
        self.results = []
        self.trainset = trainset

        self.exp = comet_exp

    def make_directories(self):
        self.trialdir = self.runpath / f"trial_{self.trial_number}"
        self.logdir = self.trialdir / "log"
        self.imgdir = self.trialdir / "images"

        self.runpath.mkdir(parents=True, exist_ok=True)
        self.trialdir.mkdir(exist_ok=True)
        self.logdir.mkdir(exist_ok=True)
        self.imgdir.mkdir(exist_ok=True)

    def run_trail(self, params):
        trial_start_time = time.time()
        self.trial_number += 1
        lr_d = params["lr_d"]
        lr_g1 = params["lr_g1"]
        lr_g2 = params["lr_g2"]
        lambda_gan_1 = params["lambda_gan_1"]
        lambda_L1_1 = params["lambda_L1_1"]
        lambda_gan_2 = params["lambda_gan_2"]
        lambda_L1_2 = params["lambda_L1_2"]
        self.batchsize = int(params["batch_size"])
        nepoch_regress = int(params["nepoch_regress"])
        nepoch_gan = int(params["nepoch_gan"])
        nblocks = int(params["nblocks"])
        nc = int(params["nchannels"])
        kernel_size = int(params["kernel_size"])
        dropout = params["dropout"]
        Cin = int(params["Cin"])
        Cout = 3
        Cnoise = 3
        self.Ctot = Cin + Cnoise
        self.Cin = Cin
        self.epoch = 0
        self.iteration = 0

        print(f"trial# {self.trial_number}: params={params}")
        if self.exp:
            self.exp.log_parameters(params)

        # initialize objects
        self.make_directories()
        self.writer = SummaryWriter(self.logdir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.trainset.Cin = self.Cin

        self.trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.batchsize,
            shuffle=True,
            num_workers=min((multiprocessing.cpu_count() // 2, 10)),
        )
        #        self.testloader = torch.utils.data.DataLoader(testset,  batch_size=128, shuffle=False, num_workers=8)

        self.gan = GAN(self.Ctot, Cout, nc, nblocks, kernel_size, dropout).to(
            self.device
        )
        self.g = self.gan.g
        self.d = self.gan.d

        # load previous stored weights
        # train using "regress then GAN" approach
        val_loss = self.train(nepoch_regress, lr_d, lr_g1, lambda_gan=0, lambda_L1=1)
        return {"loss": val_loss, "params": params, "status": STATUS_OK}

    def get_noise_tensor(self, shape):
        b, h, w = shape[0], shape[2], shape[3]
        input_tensor = torch.FloatTensor(b, self.Ctot, h, w)
        input_tensor.uniform_(-1, 1)
        return input_tensor

    def train(self, nepochs, lr_d=1e-2, lr_g=1e-2, lambda_gan=0.01, lambda_L1=1):
        # initialize trial
        d_optimizer = optim.Adam(self.d.parameters(), lr=lr_d)
        g_optimizer = optim.Adam(self.g.parameters(), lr=lr_g)

        L1 = nn.L1Loss()
        MSE = nn.MSELoss()
        device = self.device

        for epoch in range(nepochs):
            self.epoch += 1

            torch.cuda.empty_cache()
            self.gan.train()  # train mode
            num_batches = len(self.trainloader)

            for i, (coords, real_img, metos_data) in enumerate(self.trainloader):
                if epoch + i == 0:
                    print("Loaded!")
                    print(metos_data.shape, real_img.shape)
                self.iteration += 1

                shape = metos_data.shape

                input_tensor = self.get_noise_tensor(shape)
                input_tensor[:, : self.Cin, :, :] = metos_data
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
                self.writer.add_scalar("train/d_loss", d_loss.item(), self.iteration)

                g_optimizer.zero_grad()
                L1_loss = L1(generated_img, real_img)
                gan_loss = MSE(fake_prob, real_target)

                g_loss = lambda_gan * gan_loss + lambda_L1 * L1_loss
                g_loss.backward()
                g_optimizer.step()
                self.writer.add_scalar("train/g_loss", g_loss.item(), self.iteration)
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
                        nepochs,
                        i + 1,
                        num_batches,
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
            input_tensor[:, : self.Cin, :, :] = metos_data
            input_tensor = input_tensor.to(device)

            generated_img = self.g(input_tensor)

            # write out the model architechture
            if epoch == 0:
                # self.writer.add_graph(
                #     GAN(
                #         self.Ctot,
                #         3,
                #         params1["nchannels"],
                #         params1["nblocks"],
                #         int(params1["kernel_size"]),
                #         params1["dropout"],
                #     ).to(self.device),
                #     (input_tensor,),
                #     True,
                # )
                self.writer.add_graph(self.gan, (input_tensor,), True)
            imgs = torch.cat(
                (input_tensor[0, 22:25], generated_img[0, 0:3], real_img[0, 0:3]), 1
            )  # concatenate verticaly 3 metos, generated clouds, ground truth clouds
            self.writer.add_image("imgs", imgs, self.epoch, dataformats="CHW")
            # CHW = channel, height, width

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
                plt.imsave(
                    str(self.imgdir / f"imgs{i}_{self.epoch}"),
                    imgs_cpu,
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                )
                if self.exp:
                    self.exp.log_image(str(self.imgdir / f"imgs{i}_{self.epoch}"))

        torch.save(self.gan.state_dict(), str(self.trialdir / "gan.pt"))
        return 0


if __name__ == "__main__":
    scratch = os.environ.get("SCRATCH") or "~/scratch/comets"
    scratch = str(Path(scratch) / "comets")
    exp = OfflineExperiment(
        project_name="clouds", workspace="vict0rsch", offline_directory=scratch
    )

    datapath = "/home/vsch/scratch/clouds"
    trainset = EarthData(datapath, n_in_mem=50)

    trainer = gan_trainer(trainset, exp)

    params1 = {
        "nepoch_regress": 100,
        "nepoch_gan": 250,
        "optimizer": "adam",
        "lr_g1": 5e-4,
        "lr_d": 1e-4,
        "lr_g2": 1e-4,
        "lambda_gan_1": 1e-2,
        "lambda_gan_2": 3e-2,
        "lambda_L1_1": 1,
        "lambda_L1_2": 1,
        "batch_size": 32,
        "nblocks": 5,
        "nchannels": 16,
        "kernel_size": 3,
        "dropout": 0.75,
        "Cin": 42,
    }

    result = trainer.run_trail(params1)
    trainer.exp.end()
    multiprocessing.check_output([
        "bash",
        "-c", 
        "python -m comet_ml.scripts.upload {}".format(
            str(Path(scratch).resolve() / (trainer.exp.id + ".zip"))
        )
    ])

# * use pathlib
# * Cin, Cnoise, Cnoise etc <- Cin, Cnoise, Ctot etc to better read
#   + coherence accross files
# * K <- kernel_size
# * use black formatter everywhere
# * sort imports
# * remove get_wdist as it is not used
# * add get_noise_tensor function
# * rename variables to
#   fake_target, real_target, real_prob, fake_prob, generated_img, real_img
# * not sure: change add_graph to self.writer.add_graph(self.gan) -> current graph
#   instead of self.writer.add_graph(GAN(...)) -> new graph /!\/!\
# * data is loaded on the fly /!\/!\
# * inf and nan data values are set to 0. /!\/!\
