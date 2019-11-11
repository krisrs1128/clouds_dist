#!/usr/bin/env python
import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from addict import Dict

from src.data import get_loader, get_transforms
from src.gan import GAN
from src.optim import get_optimizers
from src.stats import get_stats
from src.utils import (
    check_data_dirs,
    get_opts,
    loss_hinge_dis,
    loss_hinge_gen,
    to_0_1,
    weighted_mse_loss,
)


class gan_trainer:
    def __init__(self, opts, exp=None, output_dir=".", n_epochs=50, verbose=1):
        self.opts = opts
        self.losses = {
            "gan_loss": [],
            "matching_loss": [],
            "g_loss_total": [],
            "d_loss": [],
        }
        self.trial_number = 0
        self.n_epochs = n_epochs
        self.start_time = datetime.now()
        self.verbose = verbose
        self.resumed = False
        self.timestamp = self.start_time.strftime("%Y_%m_%d_%H_%M_%S")
        self.results = []
        self.exp = exp
        self.output_dir = Path(output_dir)
        self.debug = Dict()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stats = None
        self.shape = None

        if self.verbose > 0:
            print("-------------------------")
            print("-----    Params     -----")
            print("-------------------------")
            for o, d in opts.items():
                print(o)
                for k, v in d.items():
                    print("{:<30}: {:<30}".format(str(k), str(v)))
            print()

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
        # initialize objects
        self.make_directories()

        self.transforms = get_transforms(self.opts)
        self.stats = get_stats(self.opts, self.device, self.transforms)
        self.trainloader, transforms_string = get_loader(
            opts, self.transforms, self.stats
        )
        self.trainset = self.trainloader.dataset

        # calculate the bottleneck dimension
        if self.trainset.metos_shape[-1] % 2 ** self.opts.model.n_blocks != 0:
            raise ValueError(
                "Data shape ({}) and n_blocks ({}) are not compatible".format(
                    self.trainset.metos_shape[-1], self.opts.model.n_blocks
                )
            )
        if not self.opts.model.bottleneck_dim:
            self.opts.model.bottleneck_dim = (
                self.trainset.metos_shape[-1] // 2 ** self.opts.model.n_blocks
            )

        self.gan = GAN(**self.opts.model, device=self.device).to(self.device)
        if self.opts.train.checkpoint:
            state = torch.load(Path(self.opts.train.checkpoint))
            self.gan.load_state_dict(state["state_dict"])

        self.g = self.gan.g
        self.d = self.gan.d

        if self.exp:
            wandb.watch((self.g, self.d))
            wandb.config.update(
                {
                    "transforms": transforms_string,
                    "d_num_trainable_params": sum(
                        p.numel() for p in self.d.parameters() if p.requires_grad
                    ),
                    "g_num_trainable_params": sum(
                        p.numel() for p in self.g.parameters() if p.requires_grad
                    ),
                }
            )

        self.g_optimizer, self.d_optimizer = get_optimizers(self.g, self.d, self.opts)

        if self.exp:
            wandb.config.update(
                {
                    "transforms": transforms_string,
                    "d_num_trainable_params": sum(
                        p.numel() for p in self.d.parameters() if p.requires_grad
                    ),
                    "g_num_trainable_params": sum(
                        p.numel() for p in self.g.parameters() if p.requires_grad
                    ),
                }
            )

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
        noise_tensor = torch.FloatTensor(b, Ctot, h, w)
        noise_tensor.uniform_(-1, 1)  # TODO is that legit?
        return noise_tensor

    def log_debug(self, var, name):
        self.debug[name].prev = self.debug[name].curr
        self.debug[name].curr = var

    def infer(self, batch, step, store_images, imgdir, exp):
        real_img = batch["real_imgs"].to(self.device)

        input_tensor = self.get_noisy_input_tensor(batch)

        generated_img = self.g(input_tensor)

        wandb_images = []

        for i in range(input_tensor.shape[0]):
            # concatenate verticaly:
            # [3 metos, generated clouds, ground truth clouds]
            imgs = torch.cat(
                (
                    to_0_1(input_tensor[i, 22:25]),
                    to_0_1(generated_img[i]),
                    to_0_1(real_img[i]),
                ),
                1,
            )
            imgs_cpu = imgs.cpu().clone().detach().numpy()
            imgs_cpu = np.swapaxes(imgs_cpu, 0, 2)
            if store_images:
                plt.imsave(str(imgdir / f"imgs_{step}_{i}.png"), imgs_cpu)
            if exp:
                wandb_images.append(wandb.Image(imgs_cpu, caption=f"imgs_{step}_{i}"))
        if exp:
            try:
                wandb.log({"inference_images": wandb_images}, step=step)
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
        # TODO move out to utils
        plt.figure()
        for loss in losses:
            plt.plot(np.arange(len(losses[loss])), losses[loss], label=loss)
            plt.legend()
            plt.xlabel("steps")
            plt.ylabel("losses")
            plt.savefig(str(self.offline_output_dir / "losses.png"))

    def get_noisy_input_tensor(self, batch):
        input_tensor = self.get_noise_tensor(batch["metos"].shape)
        input_tensor[:, : self.opts.model.Cin, :, :] = batch["metos"]
        return input_tensor.to(self.device)

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
                stime = time.time()
                self.batch = batch

                if i == 0 and self.verbose > -1:
                    print("\n\nLoading time: {:.3f}".format(stime - etime))

                if self.shape is None:
                    self.shape = batch["metos"].shape

                generated_img = None
                real_img = batch["real_imgs"].to(device)
                d_loss = 0
                for acc in range(num_D_accumulations):
                    # ---------------------------------------------
                    # ----- Accumulate Discriminator Gradient -----
                    # ---------------------------------------------
                    self.input_tensor = self.get_noisy_input_tensor(batch)
                    generated_img = self.g(self.input_tensor)
                    if not self.opts.model.multi_disc:
                        real_prob = self.d(real_img)
                        fake_prob = self.d(generated_img.detach())

                        # ----------------------------------
                        # ----- Backprop Discriminator -----
                        # ----------------------------------
                        self.d_optimizer.zero_grad()
                        d_loss += loss_hinge_dis(fake_prob, real_prob) / float(
                            num_D_accumulations
                        )
                    else:
                        d_loss += (
                            self.d.compute_loss(real_img, 1)
                            + self.d.compute_loss(generated_img.detach(), 0)
                        ) / float(num_D_accumulations)

                d_loss.backward()
                if (
                    "extra" not in self.opts.train.optimizer
                    or (self.total_steps + 1) % 2 == 0
                ):
                    self.d_optimizer.step()
                else:
                    self.d_optimizer.extrapolation()

                # ----------------------------
                # ----- Generator Update -----
                # ----------------------------
                self.g_optimizer.zero_grad()
                if generated_img is None:
                    self.input_tensor = self.get_noisy_input_tensor(batch)
                    generated_img = self.g(self.input_tensor)
                loss = matching_loss(real_img, generated_img)

                if num_D_accumulations > 0:
                    if not self.opts.model.multi_disc:
                        fake_prob = self.d(generated_img)
                        gan_loss = loss_hinge_gen(fake_prob)
                    else:
                        gan_loss = self.d.compute_loss(generated_img, 1)
                else:
                    gan_loss = torch.Tensor([-1])
                    d_loss = torch.Tensor([-1])

                g_loss_total = lambda_gan * gan_loss + lambda_L * loss
                g_loss_total.backward()
                if (
                    "extra" not in self.opts.train.optimizer
                    or (self.total_steps + 1) % 2 == 0
                ):
                    self.g_optimizer.step()
                else:
                    self.g_optimizer.extrapolation()

                self.total_steps += 1

                # -------------------
                # ----- Logging -----
                # -------------------

                if self.exp:
                    wandb.log(
                        {
                            "g/losss/total": g_loss_total.item(),
                            "g/loss/disc": gan_loss.item(),
                            "g/loss/matching": loss.item(),
                            "d/loss": d_loss.item(),
                        },
                        step=self.total_steps,
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

                if (
                    self.total_steps % opts.train.offline_losses_steps == 0
                    and self.exp is None
                ):  # TODO create self.should_plot_losses()
                    self.losses["gan_loss"].append(gan_loss.item())
                    self.losses["matching_loss"].append(loss.item())
                    self.losses["g_loss_total"].append(g_loss_total.item())
                    self.losses["d_loss"].append(d_loss.item())
                    self.plot_losses(self.losses)

                if self.total_steps % 10 == 0 and self.verbose > 0:
                    ep_str = "epoch:{}/{} step {}/{} ({})"
                    ep_str += " d_loss:{:0.4f} l:{:0.4f} gan_loss:{:0.4f} "
                    ep_str += (
                        "g_loss_total:{:0.4f} | t/step {:.1f} | t/ep {:.1f} | t {:.1f}"
                    )
                    print(
                        ep_str.format(
                            epoch + 1,
                            n_epochs,
                            i + 1,
                            len(self.trainloader),
                            self.total_steps,
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
        help="Resumes the model and opts (state_latest.pt in output_dir)",
    )
    parser.add_argument(
        "-x",
        "--existing_exp_id",
        type=str,
        help="if resuming, the existing exp id to continue",
    )
    parser.add_argument("-n", "--no_exp", default=False, action="store_true")
    parsed_opts = parser.parse_args()

    # ---------------------------
    # ----- Set output path -----
    # ---------------------------

    output_path = Path(parsed_opts.output_dir).resolve()

    if not output_path.exists():
        output_path.mkdir()

    # --------------------
    # ----- Get Opts -----
    # --------------------

    opts = get_opts(parsed_opts.conf_name)

    # ----------------------------------
    # ----- Check Data Directories -----
    # ----------------------------------

    opts = check_data_dirs(opts)

    # -----------------------------
    # -----  Configure wandb  -----
    # -----------------------------

    if parsed_opts.no_exp:
        exp = None
    else:
        exp, init_opts = True, {"dir": str(output_path)}
        if parsed_opts.offline:
            os.environ["WANDB_MODE"] = "dryrun"
        else:
            if parsed_opts.resume and parsed_opts.existing_exp_id:
                init_opts["resume"] = parsed_opts.existing_exp_id
        wandb.init(**init_opts)
        wandb.config.update(opts.to_dict())
        wandb.config.update({"__message": parsed_opts.message})
        if "WANDB_RUN_ID" in os.environ:
            with open(output_path / "run_id.txt", "w") as f:
                f.write(os.environ["WANDB_RUN_ID"])
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
