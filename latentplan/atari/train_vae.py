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

from __future__ import print_function
import torch.utils.data
from torch import optim
from torchvision.utils import save_image
from vae import VAE
import numpy as np
import time
import random
import os
import wandb


def loss_function(recon_x, x, mu, logvar):
    BCE = torch.mean((recon_x - x)**2)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1))
    return BCE, KLD * 0.1


def get_batch(data, b_size, device, process=True):
    if process:
        data = data.astype(np.float32)
        data /= 255  # normalize images
        random.shuffle(data)

    n_batches = len(data) // b_size
    for i in range(n_batches):
        batch = data[i*b_size : (i+1)*b_size]
        yield torch.from_numpy(batch).to(device)


def get_dataset():
    DEBUG = True
    if DEBUG:
        import pickle
        with open('/home/nikita/Projects/RL/decision-transformer/atari/dqn_replay/Breakout/atari_debug.pickle', 'rb') as f:
            dataset = pickle.load(f)
    else:
        from atari_dataset import create_atari_dataset
        num_buffers=50
        num_steps=500000
        game='Breakout'
        trajectories_per_buffer=10
        data_dir_prefix='/home/nikitad/projects/def-martin4/nikitad/decision-transformer/atari/dqn_replay/'
        dataset = create_atari_dataset(num_buffers, num_steps, game, data_dir_prefix, trajectories_per_buffer)

    dataset = dataset['observations']
    return np.array(dataset)
    

def main(args):
    im_size = args.im_size
    channels = args.channels
    device = args.device
    n_epoch = args.n_epoch
    batch_size = args.batch_size
    z_size = args.z_size
    lr = args.lr

    wandb.init(
        project="VAE",
        name="VAE" + '-' + str(random.randint(int(1e4), int(1e5))), 
        config=args, 
        mode='offline',
    )

    vae = VAE(zsize=z_size)
    vae.weight_init(mean=0, std=0.02)
    vae.to(device)

    vae_optimizer = optim.Adam(vae.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
 
    sample1 = torch.randn(im_size, z_size).to(device)

    data_train = get_dataset()
    print("Train set size:", len(data_train))

    for epoch in range(n_epoch):
        vae.train()

        batches = get_batch(data_train, batch_size, device)

        rec_loss = 0
        kl_loss = 0

        epoch_start_time = time.time()

        if (epoch + 1) % 8 == 0:
            vae_optimizer.param_groups[0]['lr'] /= 4
            print("learning rate change!")

        i = 0
        for x in batches:
            vae.train()
            vae.zero_grad()
            rec, mu, logvar = vae(x)

            loss_re, loss_kl = loss_function(rec, x, mu, logvar)
            (loss_re + loss_kl).backward()
            vae_optimizer.step()
            rec_loss += loss_re.item()
            kl_loss += loss_kl.item()

            #############################################

            os.makedirs('./vae_checkpoints', exist_ok=True)
            os.makedirs('./vae_logs/results_rec', exist_ok=True)
            os.makedirs('./vae_logs/results_gen', exist_ok=True)

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time

            # report losses and save samples each 60 iterations
            m = 60
            i += 1
            if i % m == 0:
                rec_loss /= m
                kl_loss /= m
                print('\n[%d/%d] - ptime: %.2f, rec loss: %.9f, KL loss: %.9f' % (
                    (epoch + 1), n_epoch, per_epoch_ptime, rec_loss, kl_loss))
                wandb.log({'rec_loss': rec_loss, 'kl_loss': kl_loss})
                rec_loss = 0
                kl_loss = 0
                with torch.no_grad():
                    vae.eval()
                    x_rec, _, _ = vae(x)
                    resultsample = torch.cat([x, x_rec]) * 0.5 + 0.5
                    resultsample = resultsample.cpu()
                    save_image(resultsample.view(-1, channels, im_size, im_size),
                               './vae_logs/results_rec/sample_' + str(epoch) + "_" + str(i) + '.png')
                    x_rec = vae.decode(sample1)
                    resultsample = x_rec * 0.5 + 0.5
                    resultsample = resultsample.cpu()
                    save_image(resultsample.view(-1, channels, im_size, im_size),
                               './vae_logs/results_gen/sample_' + str(epoch) + "_" + str(i) + '.png')

        if (epoch+1) % 10 == 0:            
            torch.save(vae.state_dict(), f"./vae_checkpoints/VAEmodel_{epoch+1}.pkl")

    print("Training finish!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--im_size", type=int, default=84)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--n_epoch", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--z_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.0005)    
    args = parser.parse_args()

    main(args)