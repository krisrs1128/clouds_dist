import argparse
from pathlib import Path
import numpy as np
import torch
from dataloaders import CustomLoader
from models.default import Discriminator, Generator
from options import BaseOptions
from torch.autograd import Variable
from torchvision.utils import save_image
from comet_ml import Experiment


cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def set_dirs(opt):
    s = Path(opt.source).resolve()
    assert s.exists()

    d = Path(opt.dest).resolve()
    if not d.exists():
        d.mkdir(parents=True)


def sample_image(n_row, batches_done, comet_exp=None):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(
        gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True
    )
    if comet_exp is not None:
        comet_exp.log_image("images/%d.png" % batches_done)


if __name__ == "__main__":

    # Add the following code anywhere in your machine learning file
    comex = Experiment(project_name="clouds", workspace="vict0rsch")

    parser = argparse.ArgumentParser()
    opt = BaseOptions(parser).parse()

    # Loss functions
    adversarial_loss = torch.nn.MSELoss()

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    dataloader = CustomLoader(opt)
    # Optimizers
    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
    )

    # ----------
    #  Training
    # ----------

    for epoch in range(opt.n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):

            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(
                FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim)))
            )
            gen_labels = Variable(
                LongTensor(np.random.randint(0, opt.n_classes, batch_size))
            )

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)
            comex.log_metric("g_loss", g_loss)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)
            comex.log_metric("d_real_loss", d_real_loss)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)
            comex.log_metric("d_fake_loss", d_fake_loss)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            comex.log_metric("d_loss", d_loss)

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    d_loss.item(),
                    g_loss.item(),
                )
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                sample_image(n_row=10, batches_done=batches_done, comet_exp=comex)
