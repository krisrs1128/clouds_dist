#!/usr/bin/env python
import argparse
import os
import time
from pathlib import Path
# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from addict import Dict
from src.data import get_loader, get_transforms
from src.gan import GAN
from src.optim import get_optimizers
from src.stats import get_stats
import src.utils as utils

torch.manual_seed(0)


class gan_trainer:
    def __init__(self, opts, exp=None, output_dir=".", n_epochs=50, verbose=1):
        self.opts = opts
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.resumed = False
        self.results = []
        self.exp = exp
        self.output_dir = Path(output_dir)
        self.debug = Dict()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stats = None

        if self.verbose > 0:
            print("-------------------------")
            print("-----    Params     -----")
            print("-------------------------")
            for o, d in opts.items():
                print(o)
                for k, v in d.items():
                    print("{:<30}: {:<30}".format(str(k), str(v)))
            print()

    def resume(self, path=None, step_name="latest", init_keys=["*"]):
        if path is None:
            file_path = self.ckptdir / f"state_{str(step_name)}.pt"
        else:
            file_path = Path(path) / f"state_{str(step_name)}.pt"

        assert (
            file_path.exists()
        ), "File {} does not exist (path: {}, step_name: {})".format(
            str(file_path), path, step_name
        )

        chkpt = torch.load(str(file_path))
        state = self.gan.state_dict()
        partial_state = utils.subset_keys(chkpt["state_dict"], init_keys)
        state.update(partial_state)
        self.gan.load_state_dict(state)

        self.g_optimizer.load_state_dict(chkpt["g_optimizer"])
        self.d_optimizer.load_state_dict(chkpt["d_optimizer"])
        self.total_steps = chkpt["step"]
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
        self.stats = get_stats(self.opts, self.transforms)
        self.train_loader, self.val_loader, transforms_string = get_loader(
            opts, self.transforms, self.stats
        )
        self.trainset = self.train_loader.dataset

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
                    "val_samples": len(self.val_loader.dataset),
                    "train_samples": len(self.train_loader.dataset),
                }
            )

        self.g_optimizer, self.d_optimizer = get_optimizers(self.g, self.d, self.opts)

        if self.opts.train.init_chkpt_dir:
            chkpt_path = Path(self.opts.train.init_chkpt_dir)
            self.resume(
                chkpt_path, self.opts.train.init_chkpt_step, self.opts.train.init_keys
            )

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

    def infer_(self, batch, nb_of_inferences):
        real_img = batch["real_imgs"].to(self.device)
        generated_img = None
        for i in range(nb_of_inferences):
            input_tensor = self.get_noisy_input_tensor(batch)
            gen = self.g(input_tensor)
            if generated_img is None:
                generated_img = gen
            else:
                generated_img = torch.cat([generated_img, gen], dim=-1)

        return input_tensor, real_img, generated_img

    def infer(
        self, batch, store_images, imgdir, exp, step, nb_images, nb_of_inferences
    ):
        input_tensor, real_img, generated_img = self.infer_(batch, nb_of_inferences)
        imgs = utils.cpu_images(input_tensor, real_img, generated_img)
        utils.record_images(imgs, store_images, exp, imgdir, step, nb_images)
        return generated_img

    def should_save(self, steps):
        return not self.opts.train.save_every_steps or (
            steps and steps % self.opts.train.save_every_steps == 0
        )

    def should_infer(self, steps):
        return not self.opts.val.infer_every_steps or (
            steps and steps % self.opts.val.infer_every_steps == 0
        )

    def get_noisy_input_tensor(self, batch):
        input_tensor = self.get_noise_tensor(batch["metos"].shape)
        input_tensor[:, : self.opts.model.Cin, :, :] = batch["metos"]
        return input_tensor.to(self.device)


    def get_discriminator_input(self, batch, step="disc"):
        input_real = torch.cat((batch["real_imgs"].to(self.device), batch["metos"].to(self.device)), dim=1) \
            if self.opts.model.conditional_disc else batch["real_imgs"].to(self.device)
        self.input_tensor = self.get_noisy_input_tensor(batch)
        generated_img = self.g(self.input_tensor).detach() if step == "disc" else  self.g(self.input_tensor)
        input_fake = torch.cat((generated_img, batch["metos"].to(self.device)), dim=1) \
            if self.opts.model.conditional_disc else generated_img
        return {"real": input_real, "fake": input_fake, "generated_img": generated_img}



    def discriminator_step(self, batch, i):
        d_loss = 0
        self.d_optimizer.zero_grad()
        nd_acc = self.opts.train.num_D_accumulations
        for acc in range(nd_acc):
            # ---------------------------------------------
            # ----- Accumulate Discriminator Gradient -----
            # ---------------------------------------------
            disc_input = self.get_discriminator_input(batch, "disc")
            if not self.opts.model.multi_disc:
                real_prob = self.d(disc_input["real"])
                fake_prob = self.d(disc_input["fake"])

                d_loss += utils.loss_hinge_dis(fake_prob, real_prob) / float(nd_acc)
            else:
                d_loss += (
                    self.d.compute_loss(disc_input["real"], 1)
                    + self.d.compute_loss(disc_input["fake"], 0)
                ) / float(nd_acc)

        # ----------------------------------
        # ----- Backprop Discriminator -----
        # ----------------------------------
        d_loss.backward()
        self.d_optimizer = utils.optim_step(
            self.d_optimizer, self.opts.train.optimizer, self.total_steps, i
        )
        return disc_input["generated_img"], d_loss

    def generator_step(self, batch, generated_img, i, matching_loss):
        # ----------------------------
        # ----- Generator Update -----
        # ----------------------------
        self.g_optimizer.zero_grad()

        disc_input = self.get_discriminator_input(batch, "generator")
        if not self.opts.model.multi_disc:
            fake_prob = self.d(disc_input['fake'])
            gan_loss = utils.loss_hinge_gen(fake_prob)
        else:
            gan_loss = self.d.compute_loss(disc_input['fake'], 1)

        loss = matching_loss(batch["real_imgs"].to(self.device), generated_img)
        g_loss_total = (
            self.opts.train.lambda_gan * gan_loss + self.opts.train.lambda_L * loss
        )

        g_loss_total.backward()
        self.g_optimizer = utils.optim_step(
            self.g_optimizer, self.opts.train.optimizer, self.total_steps, i
        )
        return g_loss_total, gan_loss, loss

    def log_step(
        self, batch, i, epoch, stime, etime, d_loss, g_loss_total, gan_loss, loss
    ):
        if self.exp:
            wandb.log(
                {
                    "g/loss/total": g_loss_total.item(),
                    "g/loss/disc": gan_loss.item(),
                    "g/loss/matching": loss.item(),
                    "d/loss": d_loss.item(),
                },
                step=self.total_steps,
            )

        if self.should_infer(self.total_steps):
            print("\nINFERRING\n")
            if self.opts.model.Cnoise != 0:
                self.g.eval()
            elif self.opts.model.dropout == 0:
                print("Warning: no Cnoise, no dropout : deterministic inference")
            nb_images = 0
            self.val_distances = []
            with torch.no_grad():
                for i, batch in enumerate(self.val_loader):
                    # batch x channels x height x (nb_of_inferences * width)
                    generated_imgs = self.infer(
                        batch,
                        self.opts.val.store_images,
                        self.imgdir,
                        self.exp,
                        self.total_steps,
                        nb_images,
                        self.opts.val.nb_of_inferences,
                    )
                    for gen_im in generated_imgs:
                        self.val_distances += utils.all_distances(
                            torch.split(
                                gen_im,
                                gen_im.shape[-1] // self.opts.val.nb_of_inferences,
                                -1,
                            )
                        )
                self.val_distances = [d.item() for d in self.val_distances]
                iqd = np.quantile(self.val_distances, (0.25, 0.75))
                iqd = iqd[1] - iqd[0]
                nb_images += len(batch)
                mean = np.mean(self.val_distances)
                std = np.std(self.val_distances)
                if self.exp:
                    wandb.log(
                        {
                            "val_sample_dist_iqd": iqd,
                            "val_sample_dist_mean": mean,
                            "val_sample_dist_std": std,
                        },
                        step=self.total_steps,
                    )

            self.g.train()

        if self.should_save(self.total_steps):
            print("\nSAVING\n")
            self.save(self.total_steps)

        t = time.time()
        self.times.append(t - stime)
        self.times = self.times[-100:]

        if self.total_steps % 10 == 0 and self.verbose > 0:
            ep_str = "epoch:{}/{} step {}/{} ({})"
            ep_str += " d_loss:{:0.4f} l:{:0.4f} gan_loss:{:0.4f} "
            ep_str += "g_loss_total:{:0.4f} | t/step {:.1f} | t/ep {:.1f} | t {:.1f}"
            print(
                ep_str.format(
                    epoch + 1,
                    self.opts.train.n_epochs,
                    i + 1,
                    len(self.train_loader),
                    self.total_steps,
                    d_loss.item(),
                    loss.item(),
                    gan_loss.item(),
                    g_loss_total.item(),
                    np.mean(self.times),
                    t - etime,
                    t - stime,
                ),
                end="\r",
            )

    def train(
        self, n_epochs, lambda_gan=0.01, lambda_L=1, num_D_accumulations=1, loss="l1"
    ):
        # -------------------------------
        # ----- Set Up Optimization -----
        # -------------------------------
        matching_loss = (
            nn.L1Loss()
            if loss == "l1"
            else utils.weighted_mse_loss
            if loss == "weighted"
            else nn.MSELoss()
        )
        if self.verbose > 0:
            print("-----------------------------")
            print("----- Starting training -----")
            print("-----------------------------")
        self.times = []
        self.start_time = time.time()
        self.total_steps = 0
        for epoch in range(n_epochs):
            # -------------------------
            # ----- Prepare Epoch -----
            # -------------------------
            torch.cuda.empty_cache()
            self.gan.train()  # train mode
            etime = time.time()
            for i, batch in enumerate(self.train_loader):
                # --------------------------------
                # ----- Prepare Step Procedure -----
                # --------------------------------
                if i > (self.opts.train.early_break_epoch or 1e9):
                    break
                stime = time.time()

                if i == 0 and self.verbose > -1:
                    print("\n\nLoading time: {:.3f}".format(stime - etime))

                # --------------------------------
                # ----- Take Gradient Steps -----
                # --------------------------------

                generated_img, d_loss = self.discriminator_step(batch, i)
                g_loss_total, gan_loss, loss = self.generator_step(
                    batch, generated_img, i, matching_loss
                )

                self.total_steps += 1

                # -------------------
                # ----- Logging -----
                # -------------------
                self.log_step(
                    batch, i, epoch, stime, etime, d_loss, g_loss_total, gan_loss, loss
                )


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
        help="name of conf file in config/ | may omit the .yaml extension",
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
    parser.add_argument(
        "-n",
        "--no_exp",
        default=False,
        action="store_true",
        help="Don't start an experiment for this run",
    )
    parsed_opts = parser.parse_args()

    # ---------------------------
    # ----- Set output path -----
    # ---------------------------

    output_path = Path(parsed_opts.output_dir).resolve()

    if not output_path.exists():
        output_path.mkdir()

    utils.write_hash(output_path)

    # --------------------
    # ----- Get Opts -----
    # --------------------

    opts = utils.get_opts(parsed_opts.conf_name)

    # ----------------------------------
    # ----- Check Data Directories -----
    # ----------------------------------

    opts = utils.check_data_dirs(opts)

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
