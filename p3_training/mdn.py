"""
P3 MDN — Mixture Density Network
==================================
Manual implementation using basic tensor ops only.
No torch.distributions — avoids MPS compatibility gaps.

The MDN models the probability distribution over the next
stroke point (Δx, Δy) as a mixture of 2D Gaussians.
"""
import torch
import torch.nn as nn
import math

LOG2PI = math.log(2 * math.pi)


def mdn_loss(
    pi:     torch.Tensor,   # [B, T, K]  mixture weights (post-softmax)
    mu_x:   torch.Tensor,   # [B, T, K]  mean x
    mu_y:   torch.Tensor,   # [B, T, K]  mean y
    sigma_x:torch.Tensor,   # [B, T, K]  std x (positive)
    sigma_y:torch.Tensor,   # [B, T, K]  std y (positive)
    rho:    torch.Tensor,   # [B, T, K]  correlation (-1, 1)
    target_x: torch.Tensor, # [B, T]     actual Δx
    target_y: torch.Tensor, # [B, T]     actual Δy
    mask:     torch.Tensor, # [B, T]     1=valid, 0=padding
) -> torch.Tensor:
    """
    Negative log-likelihood of (target_x, target_y) under the MDN mixture.
    Averaged over valid (non-padded) positions.
    """
    # Expand targets to match mixture dim: [B, T, 1]
    tx = target_x.unsqueeze(-1)
    ty = target_y.unsqueeze(-1)

    # Normalised deviations
    dx = (tx - mu_x) / (sigma_x + 1e-6)
    dy = (ty - mu_y) / (sigma_y + 1e-6)

    # Bivariate Gaussian exponent
    # z = dx² + dy² - 2ρ·dx·dy
    one_minus_rho2 = 1.0 - rho ** 2 + 1e-6
    z = dx**2 + dy**2 - 2.0 * rho * dx * dy

    # Log of Gaussian density for each component
    # log N(x,y | μ,σ,ρ) = -0.5*(z/(1-ρ²)) - log(2π·σx·σy·√(1-ρ²))
    log_gaussian = (
        -0.5 * z / one_minus_rho2
        - LOG2PI
        - torch.log(sigma_x + 1e-6)
        - torch.log(sigma_y + 1e-6)
        - 0.5 * torch.log(one_minus_rho2)
    )

    # Log mixture: log Σ π_k · N_k = logsumexp(log π_k + log N_k)
    log_pi   = torch.log(pi + 1e-6)
    log_mix  = torch.logsumexp(log_pi + log_gaussian, dim=-1)  # [B, T]

    # Negative log-likelihood, masked
    nll = -log_mix
    nll = nll * mask

    # Average over valid positions
    n_valid = mask.sum().clamp(min=1)
    return nll.sum() / n_valid


def mdn_sample(
    pi:      torch.Tensor,  # [K]
    mu_x:    torch.Tensor,  # [K]
    mu_y:    torch.Tensor,  # [K]
    sigma_x: torch.Tensor,  # [K]
    sigma_y: torch.Tensor,  # [K]
    rho:     torch.Tensor,  # [K]
    temperature: float = 1.0,
) -> tuple[float, float]:
    """
    Sample (Δx, Δy) from the mixture at inference time.
    Temperature > 1 = more random, < 1 = more conservative.
    """
    # Apply temperature to mixture weights
    pi_temp = pi ** (1.0 / temperature)
    pi_temp = pi_temp / pi_temp.sum()

    # Sample component
    k = torch.multinomial(pi_temp, 1).item()

    mx = mu_x[k].item()
    my = mu_y[k].item()
    sx = sigma_x[k].item() * temperature
    sy = sigma_y[k].item() * temperature
    r  = rho[k].item()

    # Sample from bivariate Gaussian using Cholesky decomposition
    # z1, z2 ~ N(0,1)
    z1 = torch.randn(1).item()
    z2 = torch.randn(1).item()

    # Cholesky: x = μx + σx·z1
    #           y = μy + σy·(ρ·z1 + √(1-ρ²)·z2)
    dx = mx + sx * z1
    dy = my + sy * (r * z1 + math.sqrt(max(1.0 - r**2, 1e-6)) * z2)

    return dx, dy
