# Copyright 2019 Stanislav Pidhorskyi
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, zsize, channels=4, imsize=84):
        super().__init__()

        self.zsize = zsize
        self.d_max = 3136

        self.fc1 = nn.Linear(self.d_max, self.zsize)
        # self.fc2 = nn.Linear(self.d_max, self.zsize)
        self.d1 = nn.Linear(self.zsize, self.d_max)

        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
    
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, channels, kernel_size=8, stride=4, padding=0),
            # nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2)
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        h1 = self.fc1(x)
        # h2 = self.fc2(x)
        return h1  # , h2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, x):
        x = self.d1(x)
        x = x.view(x.shape[0], 64, 7, 7)
        # x = batch_norm(x)
        x = F.leaky_relu(x, 0.2)
        x = self.decoder(x)
        return x
    
    @torch.no_grad()
    def get_latent(self, x):
        # mu, logvar = self.encode(x)
        # mu = mu.squeeze()
        # logvar = logvar.squeeze()
        # z = self.reparameterize(mu, logvar)
        z = self.encode(x)
        return z

    def forward(self, x):
        # mu, logvar = self.encode(x)
        # mu = mu.squeeze()
        # logvar = logvar.squeeze()
        # z = self.reparameterize(mu, logvar)
        mu = torch.tensor(0.0)
        logvar = torch.tensor(1.0)
        z = self.encode(x)
        return self.decode(z), mu, logvar

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
