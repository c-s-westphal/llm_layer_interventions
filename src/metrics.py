"""Metrics for evaluating intervention effects."""

import logging
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


class InterventionMetrics:
    """Compute metrics comparing clean and edited model outputs."""

    def __init__(self, logger: logging.Logger = None):
        """Initialize metrics calculator.

        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger("sae_interventions")

    def compute_all_metrics(
        self,
        clean_logits: torch.Tensor,
        edited_logits: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: torch.Tensor = None,
        live_mask: torch.Tensor = None
    ) -> Dict[str, float]:
        """Compute all metrics comparing clean and edited outputs.

        Args:
            clean_logits: Logits from clean model [batch, seq, vocab]
            edited_logits: Logits from edited model [batch, seq, vocab]
            targets: Target token IDs [batch, seq]
            attention_mask: Mask [batch, seq] where 1=real token, 0=padding
            live_mask: Mask [batch, seq] where feature was "live"

        Returns:
            Dictionary with metric values
        """
        metrics = {}

        # Overall metrics
        metrics["d_loss"] = self.compute_delta_loss(
            clean_logits, edited_logits, targets, attention_mask
        )
        metrics["mean_logit_delta"] = self.compute_mean_logit_delta(
            clean_logits, edited_logits, attention_mask
        )
        metrics["kl_avg"] = self.compute_kl_divergence(
            clean_logits, edited_logits, attention_mask
        )

        # Slice metrics if live_mask provided
        if live_mask is not None:
            # Firing slice (where feature was live)
            firing_metrics = self._compute_slice_metrics(
                clean_logits, edited_logits, targets,
                attention_mask, live_mask, slice_name="firing"
            )
            metrics.update(firing_metrics)

            # Control slice (where feature was not live)
            if attention_mask is not None:
                control_mask = attention_mask & (~live_mask)
            else:
                control_mask = ~live_mask

            control_metrics = self._compute_slice_metrics(
                clean_logits, edited_logits, targets,
                attention_mask, control_mask, slice_name="control"
            )
            metrics.update(control_metrics)

        return metrics

    def compute_delta_loss(
        self,
        clean_logits: torch.Tensor,
        edited_logits: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> float:
        """Compute change in cross-entropy loss (edited - clean).

        Args:
            clean_logits: Clean logits [batch, seq, vocab]
            edited_logits: Edited logits [batch, seq, vocab]
            targets: Target tokens [batch, seq]
            attention_mask: Mask [batch, seq]

        Returns:
            Delta loss (positive means edited performs worse)
        """
        # Shift logits and targets for next-token prediction
        # Logits: [batch, seq-1, vocab], Targets: [batch, seq-1]
        clean_logits_shifted = clean_logits[:, :-1, :].contiguous()
        edited_logits_shifted = edited_logits[:, :-1, :].contiguous()
        targets_shifted = targets[:, 1:].contiguous()

        if attention_mask is not None:
            mask_shifted = attention_mask[:, 1:].contiguous()
        else:
            mask_shifted = None

        # Compute losses
        clean_loss = self._compute_ce_loss(
            clean_logits_shifted, targets_shifted, mask_shifted
        )
        edited_loss = self._compute_ce_loss(
            edited_logits_shifted, targets_shifted, mask_shifted
        )

        return (edited_loss - clean_loss).item()

    def compute_mean_logit_delta(
        self,
        clean_logits: torch.Tensor,
        edited_logits: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> float:
        """Compute mean logit change at argmax positions.

        For each position, compare:
        - clean_logit[argmax_clean] vs edited_logit[argmax_clean]

        Args:
            clean_logits: Clean logits [batch, seq, vocab]
            edited_logits: Edited logits [batch, seq, vocab]
            attention_mask: Mask [batch, seq]

        Returns:
            Mean logit delta
        """
        # Get argmax from clean logits
        clean_argmax = clean_logits.argmax(dim=-1)  # [batch, seq]

        # Gather logits at these positions
        clean_logit_at_max = torch.gather(
            clean_logits, dim=-1, index=clean_argmax.unsqueeze(-1)
        ).squeeze(-1)  # [batch, seq]

        edited_logit_at_max = torch.gather(
            edited_logits, dim=-1, index=clean_argmax.unsqueeze(-1)
        ).squeeze(-1)  # [batch, seq]

        # Compute delta
        logit_delta = edited_logit_at_max - clean_logit_at_max

        # Apply mask if provided
        if attention_mask is not None:
            mask = attention_mask.bool()
            logit_delta = logit_delta[mask]

        return logit_delta.mean().item()

    def compute_kl_divergence(
        self,
        clean_logits: torch.Tensor,
        edited_logits: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> float:
        """Compute KL divergence KL(p_clean || p_edited).

        Args:
            clean_logits: Clean logits [batch, seq, vocab]
            edited_logits: Edited logits [batch, seq, vocab]
            attention_mask: Mask [batch, seq]

        Returns:
            Average KL divergence
        """
        # Convert to probabilities
        p_clean = F.softmax(clean_logits, dim=-1)
        log_p_edited = F.log_softmax(edited_logits, dim=-1)

        # KL divergence: sum_i p_clean[i] * log(p_clean[i] / p_edited[i])
        kl = F.kl_div(log_p_edited, p_clean, reduction='none').sum(dim=-1)  # [batch, seq]

        # Apply mask if provided
        if attention_mask is not None:
            mask = attention_mask.bool()
            kl = kl[mask]

        return kl.mean().item()

    def compute_token_wise_kl(
        self,
        clean_logits: torch.Tensor,
        edited_logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute token-wise KL divergence (for snippet extraction).

        Args:
            clean_logits: Clean logits [batch, seq, vocab]
            edited_logits: Edited logits [batch, seq, vocab]

        Returns:
            KL values [batch, seq]
        """
        p_clean = F.softmax(clean_logits, dim=-1)
        log_p_edited = F.log_softmax(edited_logits, dim=-1)
        kl = F.kl_div(log_p_edited, p_clean, reduction='none').sum(dim=-1)
        return kl

    def _compute_slice_metrics(
        self,
        clean_logits: torch.Tensor,
        edited_logits: torch.Tensor,
        targets: torch.Tensor,
        attention_mask: torch.Tensor,
        slice_mask: torch.Tensor,
        slice_name: str
    ) -> Dict[str, float]:
        """Compute metrics for a specific slice of positions.

        Args:
            clean_logits: Clean logits [batch, seq, vocab]
            edited_logits: Edited logits [batch, seq, vocab]
            targets: Target tokens [batch, seq]
            attention_mask: Attention mask [batch, seq]
            slice_mask: Slice mask [batch, seq]
            slice_name: Name for this slice (e.g., "firing", "control")

        Returns:
            Dictionary with slice-specific metrics
        """
        # Combine attention mask and slice mask
        if attention_mask is not None:
            combined_mask = attention_mask & slice_mask
        else:
            combined_mask = slice_mask

        # Check if there are any valid positions in this slice
        if combined_mask.sum() == 0:
            return {
                f"d_loss_{slice_name}": 0.0,
                f"kl_{slice_name}": 0.0,
                f"num_positions_{slice_name}": 0
            }

        # Shift for next-token prediction
        clean_logits_shifted = clean_logits[:, :-1, :].contiguous()
        edited_logits_shifted = edited_logits[:, :-1, :].contiguous()
        targets_shifted = targets[:, 1:].contiguous()
        combined_mask_shifted = combined_mask[:, 1:].contiguous()

        # Delta loss
        clean_loss = self._compute_ce_loss(
            clean_logits_shifted, targets_shifted, combined_mask_shifted
        )
        edited_loss = self._compute_ce_loss(
            edited_logits_shifted, targets_shifted, combined_mask_shifted
        )
        d_loss = (edited_loss - clean_loss).item()

        # KL divergence (use unshifted mask)
        kl = self.compute_kl_divergence(
            clean_logits, edited_logits, combined_mask
        )

        return {
            f"d_loss_{slice_name}": d_loss,
            f"kl_{slice_name}": kl,
            f"num_positions_{slice_name}": combined_mask.sum().item()
        }

    def _compute_ce_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute cross-entropy loss.

        Args:
            logits: Logits [batch, seq, vocab]
            targets: Target tokens [batch, seq]
            mask: Mask [batch, seq]

        Returns:
            Scalar loss
        """
        # Flatten
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)

        # Compute loss per position
        loss_per_pos = F.cross_entropy(
            logits_flat, targets_flat, reduction='none'
        ).reshape(targets.shape)

        # Apply mask if provided
        if mask is not None:
            mask = mask.bool()
            loss_per_pos = loss_per_pos[mask]

        return loss_per_pos.mean()
