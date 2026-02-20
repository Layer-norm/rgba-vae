import torch
import torch.nn as nn
import torch.nn.functional as F

def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> List[torch.Tensor]:
    # recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    recon_loss = F.l1_loss(recon_x, x, reduction='sum') / x.shape[0]

    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.shape[0]

    return recon_loss + kl_divergence, recon_loss, kl_divergence

def adversarial_loss(
        real_validity: torch.Tensor,
        fake_validity: torch.Tensor
    ) -> torch.Tensor:


    adversarial_loss = F.binary_cross_entropy(real_validity, torch.ones_like(real_validity)) + \
               F.binary_cross_entropy(fake_validity, torch.zeros_like(fake_validity))

    return adversarial_loss

def feature_matching_loss(real_features: torch.Tensor, fake_features: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(real_features, fake_features)


def vision_alignment_loss(vison_features: torch.Tensor) -> torch.Tensor:
    target = torch.ones_like(vison_features)
    alignment_loss = F.mse_loss(vison_features, target)

    return alignment_loss