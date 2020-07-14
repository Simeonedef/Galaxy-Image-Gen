import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import utils as vutils
from torch import nn
from tqdm import tqdm
from matplotlib.colors import Normalize
from PIL import Image
batch_size = 8

class FullGalaxyGan(nn.Module):
    """
    A generative model that generates entire galaxy images
    """

    def __init__(self, image_width=1000, image_height=1000, device="cpu", weights_file="../pretrained_weights/full_galaxy_gan_64", normalize=True):
        super(FullGalaxyGan, self).__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.device = torch.device(device)
        self.latent_dim = 100
        self.normalize = normalize
        self.batch_size = batch_size
        self.gen_base_filters = 64
        self.device = device
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( self.latent_dim, self.gen_base_filters * 8, 4, 1, 0, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(self.gen_base_filters * 8),
            # state size. (self.gen_base_filters*8) x 4 x 4
            nn.ConvTranspose2d(self.gen_base_filters * 8, self.gen_base_filters * 4, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(self.gen_base_filters * 4),
            # state size. (self.gen_base_filters*4) x 8 x 8
            nn.ConvTranspose2d( self.gen_base_filters * 4, self.gen_base_filters * 2, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(self.gen_base_filters * 2),
            # state size. (self.gen_base_filters*2) x 16 x 16
            nn.ConvTranspose2d( self.gen_base_filters * 2, self.gen_base_filters, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(self.gen_base_filters),
            # state size. (self.gen_base_filters) x 32 x 32
            nn.ConvTranspose2d( self.gen_base_filters, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        self.load_weights(weights_file)

    def forward(self, input):
        return self.main(input)

    def load_weights(self, weights_file):
        print("Loading Full Galaxy model...")
        self.load_state_dict(torch.load(weights_file, map_location=self.device))
        self.eval()
        self.to(self.device)
        print("Full Galaxy model loaded!")

    def generate(self, n_images):
        norm = Normalize()
        output = []
        print("Generating positions:")
        for i in tqdm(range(0, n_images, self.batch_size)):
            size = min(self.batch_size, n_images - i)
            noise = torch.randn(size, self.latent_dim, 1, 1, device=self.device)
            with torch.no_grad():
                images = self(noise).detach().cpu().numpy()
            if self.normalize:
                images = norm(images).data
            else:
                images = (images + 1) / 2
            images = (images * 255).astype(np.uint8)
            images = np.asarray([np.asarray(Image.fromarray(images[i].squeeze()).resize((self.image_height, self.image_width))) for i in range(images.shape[0])])
            images = images / 255.
            for j in range(0, size):
                output.append(images[j])
        return output



if __name__ == "__main__":
    model = FullGalaxyGan()
    imgs = model.generate(9)
    grid = vutils.make_grid(torch.Tensor(imgs).reshape(9, 1, 1000, 1000), padding=10, pad_value=1, normalize=False,
                            range=(0, 255), nrow=3)
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.show()