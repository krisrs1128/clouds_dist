import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from src.data import get_transforms, get_loader
from src.stats import get_stats
from src.gan import GAN
from src.utils import to_0_1, get_opts


class FFT:
    def __init__(self, opts):
        self.opts = opts
        self.transforms = get_transforms(opts)
        self.stats = get_stats(opts, self.transforms)
        self.train_loader, self.val_loader, self.transforms_string = get_loader(
            opts, self.transforms, self.stats
        )
        self.trainset = self.train_loader.dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_checkpoint(self):
        self.gan = GAN(**self.opts.model, device=self.device).to(self.device)
        state = self.gan.state_dict()
        chkpt = torch.load(str(args.check_point), map_location=self.device)
        state.update(chkpt["state_dict"])
        self.gan.load_state_dict(state)
        return

    def get_noisy_input_tensor(self, batch):
        input_tensor = self.get_noise_tensor(batch["metos"].shape)
        input_tensor[:, : self.opts.model.Cin, :, :] = batch["metos"]
        return input_tensor.to(self.device)

    def get_noise_tensor(self, shape):
        b, h, w = shape[0], shape[2], shape[3]
        Ctot = self.opts.model.Cin + self.opts.model.Cnoise
        noise_tensor = torch.FloatTensor(b, Ctot, h, w)
        noise_tensor.uniform_(-1, 1)  # TODO is that legit?
        return noise_tensor

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


    def infer(self, num_inferences):
        """
        :return:
        num_inferences pair of real and generated images
        """
        self.load_checkpoint()
        inffered_imgs = []
        with torch.no_grad():
            for i, batch in enumerate(self.train_loader):
                noise_input = self.get_noisy_input_tensor(batch)
                generated_imgs = self.gan.g(noise_input)
                for i in range(generated_imgs.shape[0]):
                    generated_img = to_0_1(generated_imgs[i])
                    generated_img = generated_img.permute((1, 2, 0)).cpu().clone().detach().numpy()
                    real_img = to_0_1(batch['real_imgs'][i])
                    real_img = real_img.permute((1, 2, 0)).cpu().clone().detach().numpy()
                    margin = 58
                    inffered_imgs.append({'real': real_img[margin: -margin, margin: -margin, :], 'fake': generated_img[margin: -margin, margin: -margin, :]})
                if (i + 1) * batch.shape[0] >= num_inferences:
                    break
        return inffered_imgs



    def fft(self, img):
        img = self.rgb2gray(img)
        img = torch.from_numpy(img).to(torch.float32)
        img = torch.cat([img[:, :, None], torch.zeros_like(img[:, :, None])], dim=-1).to(torch.float32)
        fft = torch.fft(img, 2)
        fft_r = fft[:, :, 0]
        fft_i = fft[:, :, 1]
        ffn =torch.sqrt(fft_r ** 2 + fft_i ** 2)
        c = 255.0 / torch.log(1 + ffn.max())
        ffn = c * torch.log(1 + ffn).numpy()
        return ffn


parser = argparse.ArgumentParser()

parser.add_argument( "-c",
    "--config_file",
    type=str,
    default=".",
    help="The experiment yaml file")

parser.add_argument( "-ch",
    "--check_point",
    type=str,
    default=".",
    help="the checkpoint contains the trained model state")

args = parser.parse_args()
config = args.config_file
opts = get_opts(config)
#############################
fft = FFT(opts)
inferred_images = fft.infer()
#############################
for image in inferred_images:
    plt.subplot(2, 3, 1)
    plt.imshow(image['real'])
    plt.title('real image')

    plt.subplot(2, 3, 2)
    fft_r = fft.fft(image['real'])
    plt.imshow(fft_r,  cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    plt.title('DFT of real image')

    plt.subplot(2, 3, 3)
    sns.distplot(fft_r.flatten())
    plt.title('Histogram of the DFT of the real image')

    plt.subplot(2, 3, 4)
    fft_f = fft.fft(image['fake'])
    plt.imshow(image['fake'])
    plt.title('generated image')

    plt.subplot(2, 3,  5)
    plt.imshow(fft_f, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    plt.title('DFT of generated image')

    plt.subplot(2, 3,  6)
    sns.distplot(fft_f.flatten())
    plt.title('Histogram of the DFT of the generated image')

    fft_d = np.linalg.norm(fft_f - fft_r)/fft_f.size
    print("distance between the two DFTs = ", fft_d)
    plt.show()








