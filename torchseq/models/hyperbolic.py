import torch
import torch.nn as nn

from torchseq.models.kl_divergence import get_kl

# import sys
# sys.path.insert(0, sys.path[-1] + '/torchseq/models')


# import torchseq.models.pvae as pvae

# from pvae.models.vae import VAE
# from pvae.distributions import RiemannianNormal, WrappedNormal
# from pvae.models.architectures import EncMob, DecMob, DecBernouilliWrapper
# from pvae.manifolds import PoincareBall
import torch.distributions as dist


class HyperbolicBottleneck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        hidden_dim = 768
        data_size = [hidden_dim]
        latent_dim = 10
        prior_iso = False

        c = nn.Parameter(1.0 * torch.ones(1), requires_grad=False)
        manifold = PoincareBall(latent_dim, c)

        self.pvae = VAE(
            WrappedNormal,  # prior distribution
            WrappedNormal,  # posterior distribution
            dist.RelaxedBernoulli,  # likelihood distribution
            EncMob(manifold, data_size, None, 0, hidden_dim, prior_iso),
            DecBernouilliWrapper(DecMob(manifold, data_size, None, 0, hidden_dim)),
            data_size,
        )

    # manifold, data_size, non_lin, num_hidden_layers, hidden_dim, prior_iso
    # manifold, data_size, non_lin, num_hidden_layers, hidden_dim

    def forward(self, encoding, memory, global_step):

        print(encoding.shape)

        z_x, px_z, zs = self.pvae(encoding.squeeze(1))

        pz = self.pvae.pz(*self.pvae.pz_params)
        kld = dist.kl_divergence(qz_x, pz).unsqueeze(0).sum(-1)

        print(kld.shape)
        print(px_z.shape)
        exit()

        if "loss" not in memory:
            memory["loss"] = 0
        memory["loss"] += hyper_loss

        return px_z, memory
