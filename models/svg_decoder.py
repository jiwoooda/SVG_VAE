import torch
import torch.nn as nn
import torch.nn.functional as F


class SVGDecoder(nn.Module):
    """SVG Decoder with 4-stacked LSTM and Mixture Density Network (MDN)."""

    def __init__(
        self,
        latent_dim: int = 64,
        num_classes: int = 62,
        hidden_dim: int = 512,
        num_mixture: int = 4,
        dropout: float = 0.7,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_mixture = num_mixture

        # 4 stacked LSTM layers
        self.lstm = nn.LSTM(
            input_size=latent_dim + num_classes + 2,  # z + class_onehot + prev_token
            hidden_size=hidden_dim,
            num_layers=4,
            dropout=dropout,
            batch_first=True,
        )

        # Command prediction head (moveTo, cubicBezier, lineTo, EOS)
        self.cmd_head = nn.Linear(hidden_dim, 4)

        # MDN heads for coordinate arguments
        self.pi_head = nn.Linear(hidden_dim, num_mixture)       # Mixture weights
        self.mu_head = nn.Linear(hidden_dim, num_mixture * 2)   # 2D coordinate means
        self.sigma_head = nn.Linear(hidden_dim, num_mixture * 2)  # 2D coordinate stds

    def forward(
        self,
        z: torch.Tensor,
        class_label: torch.Tensor,
        prev_tokens: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
        """
        Decode latent vector to SVG command sequence.

        Args:
            z: Latent vector (batch, seq_len, latent_dim)
            class_label: One-hot class label (batch, seq_len, num_classes)
            prev_tokens: Previous token coordinates (batch, seq_len, 2)
            hidden: Optional LSTM hidden state

        Returns:
            cmd_logits: Command type logits (batch, seq_len, 4)
            pi_logits: Mixture weight logits (batch, seq_len, num_mixture)
            mu: Coordinate means (batch * seq_len, num_mixture, 2)
            log_sigma: Coordinate log stds (batch * seq_len, num_mixture, 2)
            hidden: Updated LSTM hidden state
        """
        # Concatenate inputs
        x = torch.cat([z, class_label, prev_tokens], dim=-1)

        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)

        # Command type logits
        cmd_logits = self.cmd_head(lstm_out)

        # MDN parameters
        pi_logits = self.pi_head(lstm_out)
        mu = self.mu_head(lstm_out).view(-1, self.num_mixture, 2)
        log_sigma = self.sigma_head(lstm_out).view(-1, self.num_mixture, 2)

        return cmd_logits, pi_logits, mu, log_sigma, hidden

    def sample(
        self,
        z: torch.Tensor,
        class_label: torch.Tensor,
        max_seq_len: int = 100,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressively sample SVG commands.

        Args:
            z: Latent vector (batch, latent_dim)
            class_label: One-hot class label (batch, num_classes)
            max_seq_len: Maximum sequence length
            temperature: Sampling temperature

        Returns:
            commands: Sampled command types (batch, seq_len)
            coordinates: Sampled coordinates (batch, seq_len, 2)
        """
        batch_size = z.size(0)
        device = z.device

        commands = []
        coordinates = []
        hidden = None

        # Start token (zeros)
        prev_token = torch.zeros(batch_size, 1, 2, device=device)

        for _ in range(max_seq_len):
            z_expanded = z.unsqueeze(1)
            class_expanded = class_label.unsqueeze(1)

            cmd_logits, pi_logits, mu, log_sigma, hidden = self.forward(
                z_expanded, class_expanded, prev_token, hidden
            )

            # Sample command
            cmd_probs = F.softmax(cmd_logits.squeeze(1) / temperature, dim=-1)
            cmd = torch.multinomial(cmd_probs, 1)
            commands.append(cmd)

            # Sample from mixture
            pi = F.softmax(pi_logits.squeeze(1) / temperature, dim=-1)
            mixture_idx = torch.multinomial(pi, 1).squeeze(-1)

            # Get selected mixture parameters
            batch_indices = torch.arange(batch_size, device=device)
            selected_mu = mu[batch_indices, mixture_idx]  # (batch, 2)
            selected_sigma = torch.exp(log_sigma[batch_indices, mixture_idx])

            # Sample coordinates
            coord = selected_mu + selected_sigma * torch.randn_like(selected_mu)
            coordinates.append(coord.unsqueeze(1))

            prev_token = coord.unsqueeze(1)

            # Check for EOS (command index 3)
            if (cmd == 3).all():
                break

        commands = torch.cat(commands, dim=1)
        coordinates = torch.cat(coordinates, dim=1)

        return commands, coordinates
