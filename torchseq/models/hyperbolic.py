from typing import Tuple
import torch
import torch.nn as nn

from torchseq.models.hyperbolic_utils import RiemannianNormal, WrappedNormal, PoincareBall, GeodesicLayer
from torchseq.utils.logging import Logger
from math import sqrt


class HyperbolicBottleneck(nn.Module):
    curvature: float
    embedding_dim: int
    latent_dim: int
    kl_weight: float
    prior_distribution: str
    posterior_distribution: str

    def __init__(
        self,
        embedding_dim,
        latent_dim=None,
        curvature=1.0,
        kl_weight=1.0,
        prior_distribution="wrapped_normal",
        posterior_distribution="wrapped_normal",
    ):
        super().__init__()

        self.curvature = curvature
        self.embedding_dim = embedding_dim
        self.latent_dim = embedding_dim if latent_dim is None else latent_dim
        self.kl_weight = kl_weight
        self.prior_distribution = prior_distribution
        self.posterior_distribution = posterior_distribution

        self.latent_manifold = PoincareBall(dim=self.latent_dim, c=self.curvature)

        if self.prior_distribution == "riemannian_normal":
            self.prior = RiemannianNormal
        else:
            self.prior = WrappedNormal

        if self.posterior_distribution == "riemannian_normal":
            self.posterior = RiemannianNormal
        else:
            self.posterior = WrappedNormal

        self.mu_proj = nn.Linear(self.embedding_dim, self.latent_dim, bias=True)
        self.log_var_proj = nn.Linear(self.embedding_dim, 1, bias=True)

        self._pz_mu = nn.Parameter(torch.zeros(1, self.latent_dim), requires_grad=False)
        self._pz_logvar = nn.Parameter(torch.zeros(1, 1), requires_grad=False)

        self.out_proj = GeodesicLayer(self.latent_dim, self.embedding_dim, self.latent_manifold)

    def forward(self, x: torch.Tensor, global_step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.squeeze(1)  # Input is bsz x 1 x dim

        mu_preproj, log_var = self.mu_proj(x), torch.log(nn.functional.softplus(self.log_var_proj(x)) + 1e-5)

        mu = self.latent_manifold.expmap0(mu_preproj)

        # If we're in eval mode, make this deterministic
        scale = log_var.exp() if self.training else torch.full_like(log_var, 1e-15)

        qz_x = self.posterior(loc=mu, scale=scale, manifold=self.latent_manifold)
        z = qz_x.rsample(torch.Size([]))

        pz = self.prior(loc=self._pz_mu, scale=self._pz_logvar.exp(), manifold=self.latent_manifold)

        KLD = torch.nn.functional.relu(qz_x.log_prob(z) - pz.log_prob(z)).sum(-1) / self.latent_dim

        dev_str = "train" if self.training else "dev"
        Logger().log_scalar(f"hyperbolic_{dev_str}/kl", KLD.mean(), global_step)

        z = self.out_proj(z)

        return z.unsqueeze(1), KLD * self.kl_weight
