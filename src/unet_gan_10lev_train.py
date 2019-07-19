#!/usr/bin/env python
import datetime
import os
import random
import time
from datetime import datetime
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as distance
import torch
from hyperopt import STATUS_FAIL, STATUS_OK
from scipy import stats
from torch import optim
from torch.utils import data

from model_unet_gan import *
from tensorboardX import SummaryWriter

datapath = "/data/"  # where training and test data is located
temp = "T"
rh = "RH"
u = "U"
v = "V"
ts = "TS"
r_chan = "Reflectance_680"
g_chan = "Reflectance_551"
b_chan = "Reflectance_443"
H, W = 256, 256  # num channels, height, width

# read out meto data given an npz file
def readin_epic_meto(fname):
    fh = np.load(fname)
    t = fh["T"][:]
    r_hum = fh["RH"][:]
    u_wnd = fh["U"][:]
    v_wnd = fh["V"][:]
    s_temp = np.expand_dims(fh["TS"][:], axis=0)
    fh.close()
    output = np.concatenate((t, r_hum, u_wnd, v_wnd, s_temp), axis=0)
    output[np.where(np.isnan(output))] = 0
    return output


def readin_epic_img(fname):
    fh = np.load(fname)
    r = np.expand_dims(fh[r_chan][:], axis=0)
    g = np.expand_dims(fh[g_chan][:], axis=0)
    b = np.expand_dims(fh[b_chan][:], axis=0)
    output = np.concatenate((r, g, b), axis=0)
    output[np.where(np.isnan(output))] = 0
    return output


class Dataset(data.Dataset):
    def __init__(self, root, x_field, y_field, start=0, stop=-1, Cout=3, Cin=41):

        self.Cin = Cin
        self.Cout = Cout
        self.x_files = []
        self.y_files = []
        prefix = "epic_1b_"

        x_path = "{}/{}".format(root, x_field)
        y_path = "{}/{}".format(root, y_field)

        files = sorted(glob(x_path + "/**.npz"))
        files = files[start:stop]
        self.x_files += files

        # get matching y's
        for i, file in enumerate(files):
            path, b = os.path.split(file)  # split off basename
            timestamp = b[b.find(prefix) : b.find("_Collocated_MERRA2.npz")]
            y_file = "{}/{}.npz".format(y_path, timestamp)
            self.y_files.append(y_file)

        self.N = len(self.x_files)

        self.x = torch.zeros([self.N, self.Cin, H, W], dtype=torch.float)
        self.y = torch.zeros([self.N, self.Cout, H, W], dtype=torch.float)

        print()

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        im_x = readin_epic_meto(self.x_files[index])
        im_y = readin_epic_img(self.y_files[index])

        x = torch.from_numpy(im_x).float()
        y = torch.from_numpy(im_y).float()
        return x, y


trainset = Dataset(datapath, "metos", "imgs")
# trainset = Dataset(datapath,[2014],fieldA,fieldB,start=0)
# testset  = Dataset(datapath,[2013],fieldA,fieldB,start=0,stop=1000)


class gan_trainer:
    def __init__(self, nepochs=50):

        self.trial_number = 0
        self.nepochs = nepochs
        self.start_time = datetime.now()
        self.timestamp = self.start_time.strftime("%Y_%m_%d_%H_%M_%S")
        self.runname = "unet_gan_10level"
        self.runpath = "output/" + self.runname + "/output_" + self.timestamp
        self.results = []
        # self.pt_file      = pt_file
        print(self.runpath)

    def make_directories(self):

        self.trialdir = "{}/trial_{}".format(self.runpath, self.trial_number)
        self.logdir = "{}/log".format(self.trialdir)
        self.imgdir = "{}/images".format(self.trialdir)
        print("output path:", self.trialdir)
        os.makedirs(self.trialdir, exist_ok=True)
        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.imgdir, exist_ok=True)

    def run_trail(self, params):

        # unpack parameters

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
        self.Cnoise = Cin + Cnoise
        self.Cin = Cin
        self.epoch = 0
        self.iteration = 0

        print()
        print("trial# {}: params={}".format(self.trial_number, params))

        self.make_directories()

        # initialize objects

        self.writer = SummaryWriter(self.logdir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        trainset.Cin = self.Cin
        # testset.Cin  = self.Cin

        self.trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batchsize, shuffle=True, num_workers=8
        )
        #        self.testloader = torch.utils.data.DataLoader(testset,  batch_size=128, shuffle=False, num_workers=8)

        self.gan = GAN(self.Cnoise, Cout, nc, nblocks, kernel_size, dropout).to(
            self.device
        )
        self.g = self.gan.g
        self.d = self.gan.d

        # load previous stored weights
        # self.gan.load_state_dict(torch.load(self.pt_file))

        # train using "regress then GAN" approach

        val_loss = self.train(nepoch_regress, lr_d, lr_g1, lambda_gan=0, lambda_L1=1)
        val_loss = self.train(nepoch_gan, lr_d, lr_g2, lambda_gan_1, lambda_L1_1)
        val_loss = self.train(nepoch_gan, lr_d, lr_g2, lambda_gan_2, lambda_L1_2)

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
                A = torch.FloatTensor(s[0], self.Cnoise, s[2], s[3]).uniform_(-1, 1)
                A[:, : self.Cin, :, :] = A_imgs
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

            print()

            # apply gan to validation data
            """
            torch.cuda.empty_cache()
            self.gan.eval()
            val_loss     = 0
            L1_loss_test = 0
            nval         = 0
            num_batches  = len(self.testloader)
            net_wdist = 0

            for i, (A_imgs, B_real) in enumerate(self.testloader):
                nval+=1

                s = A_imgs.shape
                A = torch.FloatTensor(s[0], self.Cnoise, s[2], s[3]).uniform_(-1,1)
                A[:,:self.Cin, :,:] = A_imgs
                A = A.to(device)

                B_real = B_real.to(device)
                B_fake = self.g(A)

                wdist = self.get_wdist(B_real,B_fake)
                net_wdist += wdist

                C_real = self.d(B_real)
                C_fake = self.d(B_fake)

                L_real = torch.ones (C_real.shape).to(device)
                L_fake = torch.zeros(C_real.shape).to(device)

                #d_loss_test = 0.5*(MSE(C_fake, L_fake) + MSE(C_real, L_real))
                L1_loss  = L1 (B_fake, B_real)
                gan_loss = MSE(C_fake, L_real)
                g_loss   = lambda_gan * gan_loss + lambda_L1 * L1_loss

                val_loss = val_loss + g_loss.item()
                L1_loss_test += L1_loss.item()

                print('trail:{} epoch:{}/{} test iteration {}/{} test/wdist:{:0.04f} test/L1_loss:{:0.4f} test/g_loss:{:0.4f}'.
                      format(self.trial_number,epoch+1,nepochs, i+1, num_batches, wdist, L1_loss.item(), g_loss.item()), end="\r")
            print()

            val_loss/=nval
            L1_loss_test/=nval
            wdist = net_wdist/nval

            self.writer.add_scalar('test/g_loss',val_loss, self.epoch)
            self.writer.add_scalar('test/L1_loss',L1_loss, self.epoch)
            self.writer.add_scalar('test/wdist',wdist, self.epoch)
            """
            # print('epoch:{} train/g_loss:{:0.4f} test/g_loss:{:0.4f}'.
            #      format(self.epoch, epoch_loss, val_loss))

            # output sample images
            s = A_imgs.shape
            A = torch.FloatTensor(s[0], self.Cnoise, s[2], s[3]).uniform_(-1, 1)
            A[:, : self.Cin, :, :] = A_imgs
            A = A.to(device)

            B_fake = self.g(A)
            # write out the model architechture
            if epoch == 0:
                self.writer.add_graph(
                    GAN(
                        self.Cnoise,
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
            print(imgs.shape)
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

        # self.results.append((val_loss, self.trial_number, nparams, params))
        # self.results.sort()

        # print("trial={}: loss={}".format(self.trial_number, val_loss))
        # nresults = len(self.results)
        # if(nresults>20): nresults=20
        # for i in range(nresults): print(self.results[i])
        return 0


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
    "Cin": 41,
}

result = trainer.run_trail(params1)
