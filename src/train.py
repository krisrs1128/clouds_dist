#!/usr/bin/env python
from comet_ml import Experiment, OfflineExperiment
import argparse
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from addict import Dict
from torch import optim
from torchvision import transforms

from src.data import EarthData
from src.gan import GAN
from src.preprocessing import Zoom, Rescale, RemoveNans, SquashChannels
from src.utils import merge_defaults, load_conf, sample_param
from src.optim import ExtraSGD, extragrad_step


def loss_hinge_dis(dis_fake, dis_real):
    # This version returns a single loss
    # from https://github.com/ajbrock/BigGAN-PyTorch/blob/master/losses.py
    loss = torch.mean(F.relu(1.0 - dis_real))
    loss += torch.mean(F.relu(1.0 + dis_fake))
    return loss


def loss_hinge_gen(dis_fake):
    # from https://github.com/ajbrock/BigGAN-PyTorch/blob/master/losses.py
    loss = -torch.mean(dis_fake)
    return loss


def weighted_mse_loss(input, target):
    # from https://discuss.pytorch.org/t/pixelwise-weights-for-mseloss/1254/2
    out = (input - target) ** 2
    weights = (input.sum(1) != 0).to(torch.float32)
    weights = weights.unsqueeze(1).expand_as(out) / weights.sum()
    out = out * weights
    loss = out.sum()
    return loss


class gan_trainer:
    def __init__(self, opts, comet_exp=None, output_dir=".", n_epochs=50, verbose=1):
        self.opts = opts
        self.losses = {
            "gan_loss": [],
            "matching_loss": [],
            "g_loss_total": [],
            "d_loss": [],
        }
        transfs = [Zoom()]

        if self.opts.data.squash_channels:
            transfs += [SquashChannels()]
            assert (
                self.opts.model.Cin == 8
            ), "using squash_channels, Cin should be 8 not {}".format(
                self.opts.model.Cin
            )

        if self.opts.data.preprocessed_data_path is None and self.opts.data.with_stats:
            transfs += [
                Rescale(
                    data_path=self.opts.data.path,
                    batch_size=self.opts.train.batch_size,
                    num_workers=self.opts.data.num_workers,
                    verbose=1,
                )
            ]
        transfs += [RemoveNans()]

        self.trainset = EarthData(
            self.opts.data.path,
            preprocessed_data_path=self.opts.data.preprocessed_data_path,
            load_limit=self.opts.data.load_limit or -1,
            transform=transforms.Compose(transfs),
        )

        self.trial_number = 0
        self.n_epochs = n_epochs
        self.start_time = datetime.now()
        self.verbose = verbose
        self.resumed = False

        timestamp = self.start_time.strftime("%Y_%m_%d_%H_%M_%S")
        self.timestamp = timestamp

        self.results = []

        self.exp = comet_exp
        self.output_dir = Path(output_dir)

        if self.verbose > 0:
            print("-------------------------")
            print("-----    Params     -----")
            print("-------------------------")
            for o, d in opts.items():
                print(o)
                for k, v in d.items():
                    print("{:<30}: {:<30}".format(str(k), str(v)))
            print()
        self.make_directories()
        self.debug = Dict()

    def resume(self, path=None, step_name="latest"):
        if path is None:
            file_path = self.ckptdir / f"state_{str(step_name)}.pt"
        else:
            file_path = Path(path) / f"state_{str(step_name)}.pt"

        assert (
            file_path.exists()
        ), "File {} does not exist (path: {}, step_name: {})".format(
            str(file_path), path, step_name
        )

        state = torch.load(str(file_path))
        self.gan.load_state_dict(state["state_dict"])
        self.g_optimizer.load_state_dict(state["g_optimizer"])
        self.d_optimizer.load_state_dict(state["d_optimizer"])
        self.total_steps = state["step"]
        self.resumed = True
        print("Loaded model from {}".format(str(file_path)))

    def save(self, step=0):
        state = {
            "step": step,
            "state_dict": self.gan.state_dict(),
            "d_optimizer": self.d_optimizer.state_dict(),
            "g_optimizer": self.g_optimizer.state_dict(),
        }
        torch.save(state, str(self.ckptdir / f"state_{step}.pt"))
        torch.save(state, str(self.ckptdir / f"state_latest.pt"))

    def make_directories(self):
        self.ckptdir = self.output_dir / "checkpoints"
        self.imgdir = self.output_dir / "images"
        self.offline_output_dir = self.output_dir / "output"

        self.ckptdir.mkdir(parents=True, exist_ok=True)
        self.imgdir.mkdir(exist_ok=True)
        self.offline_output_dir.mkdir(exist_ok=True)

    def setup(self):
        if self.exp:
            self.exp.log_parameters(self.opts.train)
            self.exp.log_parameters(self.opts.model)

        # initialize objects
        self.make_directories()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.opts.train.batch_size,
            shuffle=True,
            num_workers=self.opts.data.get("num_workers", 3),
        )

        # calculate the bottleneck dimension
        if self.trainset.metos_shape[-1] % 2 ** self.opts.model.n_blocks != 0:
            raise ValueError(
                "Data shape ({}) and n_blocks ({}) are not compatible".format(
                    self.trainset.metos_shape[-1], self.opts.model.n_blocks
                )
            )
        btdim = self.trainset.metos_shape[-1] // 2 ** self.opts.model.n_blocks
        self.gan = GAN(**self.opts.model, bottleneck_dim=btdim, device=self.device).to(
            self.device
        )
        self.g = self.gan.g
        self.d = self.gan.d

        self.d_optimizer = ExtraSGD(self.d.parameters(), lr=self.opts.train.lr_d)
        self.g_optimizer = ExtraSGD(self.g.parameters(), lr=self.opts.train.lr_g)

    def run_trial(self):
        self.train(
            self.opts.train.n_epochs,
            self.opts.train.lambda_gan,
            self.opts.train.lambda_L,
            self.opts.train.num_D_accumulations,
            self.opts.train.matching_loss,
        )

    def get_noise_tensor(self, shape):
        b, h, w = shape[0], shape[2], shape[3]
        Ctot = self.opts.model.Cin + self.opts.model.Cnoise
        input_tensor = torch.FloatTensor(b, Ctot, h, w)
        input_tensor.uniform_(-1, 1)
        return input_tensor

    def log_debug(self, var, name):
        self.debug[name].prev = self.debug[name].curr
        self.debug[name].curr = var

    def infer(self, batch, step, store_images, imgdir, exp):
        # output sample images
        shape = batch["metos"].shape

        real_img = batch["real_imgs"].to(self.device)

        input_tensor = self.get_noise_tensor(shape)
        input_tensor[:, : self.opts.model.Cin, :, :] = batch["metos"]
        input_tensor = input_tensor.to(self.device)

        generated_img = self.g(input_tensor)

        for i in range(input_tensor.shape[0]):
            # concatenate verticaly:
            # [3 metos, generated clouds, ground truth clouds]
            tmp_tensor = input_tensor[i, 22:25].clone().detach()
            tmp_tensor -= tmp_tensor.min()
            tmp_tensor /= tmp_tensor.max()
            imgs = torch.cat((tmp_tensor, generated_img[i], real_img[i]), 1)
            imgs_cpu = imgs.cpu().detach().numpy()
            imgs_cpu = np.swapaxes(imgs_cpu, 0, 2)
            if store_images:
                np.save(str(imgdir / f"imgs_{step}_{i}.npy"), imgs_cpu)
            if exp:
                try:
                    exp.log_image(imgs_cpu, name=f"imgs_{step}_{i}")
                except Exception as e:
                    print(f"\n{e}\n")

    def should_save(self, steps):
        return not self.opts.train.save_every_steps or (
            steps and steps % self.opts.train.save_every_steps == 0
        )

    def should_infer(self, steps):
        return not self.opts.train.infer_every_steps or (
            steps and steps % self.opts.train.infer_every_steps == 0
        )

    def plot_losses(self, losses):
        plt.figure()
        for loss in losses:
            plt.plot(np.arange(len(losses[loss])), losses[loss], label=loss)
            plt.legend()
            plt.xlabel("steps")
            plt.ylabel("losses")
            plt.savefig(str(self.offline_output_dir / "losses.png"))

    def train(
        self, n_epochs, lambda_gan=0.01, lambda_L=1, num_D_accumulations=1, loss="l1"
    ):
        # -------------------------------
        # ----- Set Up Optimization -----
        # -------------------------------

        matching_loss = (
            nn.L1Loss()
            if loss == "l1"
            else weighted_mse_loss
            if loss == "weighted"
            else nn.MSELoss()
        )
        MSE = nn.MSELoss()
        device = self.device
        if self.verbose > 0:
            print("-----------------------------")
            print("----- Starting training -----")
            print("-----------------------------")
        self.times = []
        start_time = time.time()
        self.total_steps = 0
        t = 0
        for epoch in range(n_epochs):
            # -------------------------
            # ----- Prepare Epoch -----
            # -------------------------
            torch.cuda.empty_cache()
            self.gan.train()  # train mode
            etime = time.time()
            for i, batch in enumerate(self.trainloader):
                # --------------------------------
                # ----- Start Step Procedure -----
                # --------------------------------
                if i > (self.opts.train.early_break_epoch or 1e9):
                    break
                self.batch = batch
                stime = time.time()

                if i == 0 and self.verbose > -1:
                    print("\n\nLoading time: {:.3f}".format(stime - etime))

                shape = batch["metos"].shape

                for acc in range(num_D_accumulations):
                    # ---------------------------------------------
                    # ----- Accumulate Discriminator Gradient -----
                    # ---------------------------------------------
                    self.input_tensor = self.get_noise_tensor(shape)
                    self.input_tensor[:, : self.opts.model.Cin, :, :] = batch["metos"]
                    self.input_tensor = self.input_tensor.to(device)

                    real_img = batch["real_imgs"].to(device)
                    generated_img = self.g(self.input_tensor)

                    real_prob = self.d(real_img)
                    fake_prob = self.d(generated_img.detach())

                    real_target = torch.ones(real_prob.shape, device=device)
                    fake_target = torch.zeros(fake_prob.shape, device=device)

                    # ----------------------------------
                    # ----- Backprop Discriminator -----
                    # ----------------------------------
                    self.d_optimizer.zero_grad()
                    d_loss = loss_hinge_dis(fake_prob, real_prob) / float(
                        num_D_accumulations
                    )
                    extragrad_step(self.d_optimizer, self.d, i)

                # ----------------------------
                # ----- Generator Update -----
                # ----------------------------
                self.g_optimizer.zero_grad()
                fake_prob = self.d(generated_img)
                loss = matching_loss(real_img, generated_img)
                gan_loss = loss_hinge_gen(fake_prob)

                g_loss_total = lambda_gan * gan_loss + lambda_L * loss
                extragrad_step(self.g_optimizer, self.g, i)

                # -------------------
                # ----- Logging -----
                # -------------------

                if self.exp:
                    self.exp.log_metrics(
                        {
                            "g/losss/total": g_loss_total.item(),
                            "g/loss/disc": gan_loss.item(),
                            "g/loss/matching": loss.item(),
                            "d/loss": d_loss.item(),
                            "track_gen/min": generated_img.min(),
                            "track_gen/max": generated_img.max(),
                            "track_gen/mean": generated_img.mean(),
                            "track_gen/std": generated_img.std(),
                        }
                    )

                if self.should_infer(self.total_steps):
                    print("\nINFERING\n")
                    self.infer(
                        batch,
                        self.total_steps,
                        self.opts.train.store_images,
                        self.imgdir,
                        self.exp,
                    )

                if self.should_save(self.total_steps):
                    print("\nSAVING\n")
                    self.save(self.total_steps)

                t = time.time()
                self.times.append(t - stime)
                self.times = self.times[-100:]
                self.total_steps += 1

                if (
                    self.total_steps % opts.train.offline_losses_steps == 0
                    and self.exp is None
                ):
                    self.losses["gan_loss"].append(gan_loss.item())
                    self.losses["matching_loss"].append(loss.item())
                    self.losses["g_loss_total"].append(g_loss_total.item())
                    self.losses["d_loss"].append(d_loss.item())
                    self.plot_losses(self.losses)

                if self.total_steps % 10 == 0 and self.verbose > 0:
                    ep_str = "epoch:{}/{} step {}/{} d_loss:{:0.4f} l:{:0.4f} gan_loss:{:0.4f} "
                    ep_str += (
                        "g_loss_total:{:0.4f} | t/step {:.1f} | t/ep {:.1f} | t {:.1f}"
                    )
                    print(
                        ep_str.format(
                            epoch + 1,
                            n_epochs,
                            i + 1,
                            len(self.trainloader),
                            d_loss.item(),
                            loss.item(),
                            gan_loss.item(),
                            g_loss_total.item(),
                            np.mean(self.times),
                            t - etime,
                            t - start_time,
                        ),
                        end="\r",
                    )
            print("\nEnd of Epoch\n")
            # ------------------------
            # ----- END OF EPOCH -----
            # ------------------------


if __name__ == "__main__":

    scratch = os.environ.get("SCRATCH") or os.path.join(
        os.environ.get("HOME"), "scratch"
    )

    # -------------------------
    # ----- Set Up Parser -----
    # -------------------------

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
        help="name of conf file in config/ | may ommit the .yaml extension",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="where the run's data should be stored ; used to resume",
    )
    parser.add_argument(
        "-r",
        "--resume",
        action="store_true",
        help="Resumes the model (and opts) that are stored as state_latest.pt in output_dir",
    )
    parser.add_argument("-n", "--no_exp", default=False, action="store_true")
    parsed_opts = parser.parse_args()

    # ---------------------------
    # ----- Set output path -----
    # ---------------------------

    output_path = Path(parsed_opts.output_dir).resolve()

    if not output_path.exists():
        output_path.mkdir()

    # ----------------------------------
    # ----- Get Configuration File -----
    # ----------------------------------

    conf_path = parsed_opts.conf_name
    if not Path(conf_path).exists():
        conf_name = conf_path
        if not conf_name.endswith(".yaml"):
            conf_name += ".yaml"
        conf_path = Path(__file__).parent.parent / "shared" / conf_name
        assert conf_path.exists()

    # --------------------
    # ----- Get Opts -----
    # --------------------

    opts = merge_defaults({"model": {}, "train": {}, "data": {}}, conf_path)

    data_path = opts.data.path.split("/")
    for i, d in enumerate(data_path):
        if "$" in d:
            data_path[i] = os.environ.get(d.replace("$", ""))
    opts.data.path = "/".join(data_path)

    # ----------------------------------
    # ----- Check Data Directories -----
    # ----------------------------------

    print("Loading data from ", str(opts.data.path))
    assert Path(opts.data.path).exists()
    assert (Path(opts.data.path) / "imgs").exists()
    assert (Path(opts.data.path) / "metos").exists()
    # print("Make sure you are using proxychains so that comet has internet access")

    # ------------------------------
    # ----- Configure comet.ml -----
    # ------------------------------

    if parsed_opts.no_exp:
        exp = None
    else:
        if parsed_opts.offline:
            exp = OfflineExperiment(offline_directory=str(output_path))
        else:
            exp = Experiment()
        exp.log_parameter("__message", parsed_opts.message)

    # --------------------------
    # -----   Initialize   -----
    # --------------------------

    trainer = gan_trainer(opts, exp, output_path)
    trainer.setup()
    # ----------------------
    # -----   Resume   -----
    # ----------------------
    if parsed_opts.resume:
        trainer.resume()

    # ---------------------
    # -----   Train   -----
    # ---------------------
    trainer.run_trial()

    # --------------------------------
    # ----- End Comet Experiment -----
    # --------------------------------

    if not parsed_opts.no_exp:
        trainer.exp.end()

    if parsed_opts.offline and not parsed_opts.no_exp:
        subprocess.check_output(
            [
                "bash",
                "-c",
                "python -m comet_ml.scripts.upload {}".format(
                    str(Path(output_path).resolve() / (trainer.exp.id + ".zip"))
                ),
            ]
        )
