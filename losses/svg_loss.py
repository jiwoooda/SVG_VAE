import torch
import torch.nn.functional as F


def command_loss(
    cmd_logits: torch.Tensor,
    target_cmd: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cross-entropy loss for SVG command prediction.

    Args:
        cmd_logits: Predicted command logits (batch, seq_len, 4)
        target_cmd: Target command indices (batch, seq_len)

    Returns:
        Command classification loss
    """
    return F.cross_entropy(cmd_logits.view(-1, 4), target_cmd.view(-1))


def mdn_loss(
    pi_logits: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    target_args: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Mixture Density Network negative log-likelihood loss.

    Args:
        pi_logits: Mixture weight logits (batch, seq_len, num_mixture)
        mu: Coordinate means (batch * seq_len, num_mixture, 2)
        log_sigma: Coordinate log stds (batch * seq_len, num_mixture, 2)
        target_args: Target coordinates (batch, seq_len, 2)

    Returns:
        MDN negative log-likelihood loss
    """
    # Get mixture weights
    pi = F.softmax(pi_logits.view(-1, pi_logits.size(-1)), dim=-1)

    # Compute Gaussian log probabilities
    sigma = torch.exp(log_sigma)
    target_flat = target_args.view(-1, 1, 2)  # (batch * seq_len, 1, 2)

    # Normal distribution log probability
    normal = torch.distributions.Normal(mu, sigma)
    log_prob = normal.log_prob(target_flat).sum(-1)  # Sum over 2D coordinates

    # Weighted log probability with mixture weights
    weighted_log_prob = log_prob + torch.log(pi + 1e-8)

    # Log-sum-exp for mixture
    mdn_nll = -torch.logsumexp(weighted_log_prob, dim=-1).mean()

    return mdn_nll


def svg_decoder_loss(
    cmd_logits: torch.Tensor,
    pi_logits: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    target_cmd: torch.Tensor,
    target_args: torch.Tensor,
    cmd_weight: float = 1.0,
    mdn_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute full SVG decoder loss: command CE + MDN NLL.

    Args:
        cmd_logits: Predicted command logits
        pi_logits: Mixture weight logits
        mu: Coordinate means
        log_sigma: Coordinate log stds
        target_cmd: Target command indices
        target_args: Target coordinates
        cmd_weight: Weight for command loss
        mdn_weight: Weight for MDN loss

    Returns:
        total_loss: Combined loss
        cmd_loss: Command loss component
        mdn_loss_val: MDN loss component
    """
    cmd_loss_val = command_loss(cmd_logits, target_cmd)
    mdn_loss_val = mdn_loss(pi_logits, mu, log_sigma, target_args)

    total_loss = cmd_weight * cmd_loss_val + mdn_weight * mdn_loss_val

    return total_loss, cmd_loss_val, mdn_loss_val
