#!/usr/bin/env python
from hyperopt import STATUS_OK, STATUS_FAIL
from tensorboardX import SummaryWriter
from datetime import datetime
from torch import optim
import torch
from torch.utils import data
import time
import os
import random
import scipy.spatial.distance as distance
from scipy import stats


class gan_trainer:
    def __init__(self, nepochs=50):

        self.trial_number = 0
        self.nepochs = nepochs
        self.start_time = datetime.now()
        self.timestamp = self.start_time.strftime("%Y_%m_%d_%H_%M_%S")
        self.runname = "unet_gan_10level"
        self.runpath = os.path.join(
            "output/", self.runname, "output_{}".format(timestamp)
        )
        self.results = []

    def make_directories(self):
        self.trialdir = "{}/trial_{}".format(self.runpath, self.trial_number)
        self.logdir = "{}/log".format(self.trialdir)
        self.imgdir = "{}/images".format(self.trialdir)
        os.makedirs(self.trialdir, exist_ok=True)
        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.imgdir, exist_ok=True)

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
        K = int(params["kernel_size"])
        dropout = params["dropout"]
        cin = int(params["cin"])
        cout = 3
        cnoise = 3
        self.csum = cin + cnoise
        self.cin = cin
        self.epoch = 0
        self.iteration = 0

        print("trial# {}: params={}".format(self.trial_number, params))

        # initialize objects
        self.make_directories()
        self.writer = SummaryWriter(self.logdir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        trainset.Cin = self.cin

        self.trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batchsize, shuffle=True, num_workers=8
        )
        #        self.testloader = torch.utils.data.DataLoader(testset,  batch_size=128, shuffle=False, num_workers=8)

        self.gan = GAN(self.csum, cout, nc, nblocks, K, dropout).to(self.device)
        self.g = self.gan.g
        self.d = self.gan.d

        # load previous stored weights
        # train using "regress then GAN" approach
        val_loss = self.train(nepoch_regress, lr_d, lr_g1, lambda_gan=0, lambda_L1=1)
        return {"loss": val_loss, "params": params, "status": STATUS_OK}

    def get_wdist(self, B_real, B_fake):
        Bf = B_fake.cpu().detach().numpy()
        Br = B_real.cpu().detach().numpy()

        s = Br.shape
        npixels = s[1] * s[2] * s[3]

        B1 = Br.reshape((s[0], npixels))
        B2 = Bf.reshape((s[0], npixels))

        pdist1 = distance.pdist(B1, metric="cityblock") / npixels
        pdist2 = distance.pdist(B2, metric="cityblock") / npixels
        wass_dist = stats.wasserstein_distance(pdist1, pdist2)

        return wass_dist

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
            self.gan.train()
            num_batches = len(self.trainloader)

            for i, (A_imgs, B_real) in enumerate(self.trainloader):
                self.iteration += 1

                s = A_imgs.shape
                A = torch.FloatTensor(s[0], self.csum, s[2], s[3]).uniform_(-1, 1)
                A[:, : self.cin, :, :] = A_imgs
                A = A.to(device)

                B_real = B_real.to(device)
                B_fake = self.g(A)

                C_real = self.d(B_real)
                C_fake = self.d(B_fake)

                L_real = torch.ones(C_real.shape, device=device)
                L_fake = torch.zeros(C_fake.shape, device=device)

                d_optimizer.zero_grad()
                d_loss = 0.5 * (MSE(C_fake, L_fake) + MSE(C_real, L_real))
                d_loss.backward(retain_graph=True)
                d_optimizer.step()
                self.writer.add_scalar("train/d_loss", d_loss.item(), self.iteration)

                g_optimizer.zero_grad()
                L1_loss = L1(B_fake, B_real)
                gan_loss = MSE(C_fake, L_real)

                g_loss = lambda_gan * gan_loss + lambda_L1 * L1_loss
                g_loss.backward()
                g_optimizer.step()
                self.writer.add_scalar("train/g_loss", g_loss.item(), self.iteration)

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

            # output sample images
            s = A_imgs.shape
            A = torch.FloatTensor(s[0], self.csum, s[2], s[3]).uniform_(-1, 1)
            A[:, : self.cin, :, :] = A_imgs
            A = A.to(device)

            B_fake = self.g(A)
            # write out the model architechture
            if epoch == 0:
                self.writer.add_graph(
                    GAN(
                        self.csum,
                        3,
                        params1["nchannels"],
                        params1["nblocks"],
                        int(params1["kernel_size"]),
                        params1["dropout"],
                    ).to(self.device),
                    (A,),
                    True,
                )
            imgs = torch.cat((A[0, 22:25], B_fake[0, 0:3], B_real[0, 0:3]), 1)
            self.writer.add_image("imgs", imgs, self.epoch, dataformats="CHW")

            for i in range(1):
                imgs = torch.cat((A[i, 22:25], B_fake[i, 0:3], B_real[i, 0:3]), 1)
                imgs_cpu = imgs.cpu().detach().numpy()
                imgs_cpu = np.swapaxes(imgs_cpu, 0, 2)
                plt.imsave(
                    self.imgdir + "/imgs%d_%d" % (i, self.epoch),
                    imgs_cpu,
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                )

        torch.save(self.gan.state_dict(), self.trialdir + "/gan.pt")
        return 0


if __name__ == "__main__":
    datapath = "/data/"
    trainset = Dataset(datapath, "metos", "imgs")
    trainer = gan_trainer()

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
        "cin": 41,
    }

    result = trainer.run_trail(params1)
