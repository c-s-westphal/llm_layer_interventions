"""Intervention logic for SAE feature manipulation."""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from sae_lens import SAE
from tqdm import tqdm
from transformer_lens import HookedTransformer


class FeatureIntervention:
    """Manage feature interventions with calibration and alpha scaling."""

    def __init__(
        self,
        model: HookedTransformer,
        saes: Dict[int, SAE],
        hook: str = "resid_pre",
        live_percentile: float = 90.0,
        logger: logging.Logger = None
    ):
        """Initialize intervention manager.

        Args:
            model: HookedTransformer model
            saes: Dictionary mapping layer index to SAE
            hook: Hook type ("resid_pre" or "resid_post")
            live_percentile: Percentile threshold for "live" features (default: 90)
            logger: Logger instance
        """
        self.model = model
        self.saes = saes
        self.hook = hook
        self.live_percentile = live_percentile
        self.logger = logger or logging.getLogger("sae_interventions")

        # Store calibrated thresholds: {(layer, feature_id): threshold}
        self.thresholds = {}

    def calibrate_thresholds(
        self,
        calibration_data: List[torch.Tensor],
        selected_features: Dict[int, List[int]],
        batch_size: int = 16
    ) -> Dict[Tuple[int, int], float]:
        """Calibrate per-feature thresholds using calibration data.

        Args:
            calibration_data: List of token ID tensors
            selected_features: Dict mapping layer -> list of feature IDs
            batch_size: Batch size for processing

        Returns:
            Dictionary mapping (layer, feature_id) to threshold value
        """
        self.logger.info(
            f"Calibrating thresholds on {len(calibration_data)} passages "
            f"(P{self.live_percentile} over non-zero activations)..."
        )

        # Collect activations per (layer, feature)
        activations_dict = {}

        for layer in selected_features.keys():
            for feature_id in selected_features[layer]:
                activations_dict[(layer, feature_id)] = []

        # Process calibration data in batches
        num_batches = (len(calibration_data) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc="Calibrating"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(calibration_data))
            batch_tokens = calibration_data[batch_start:batch_end]

            # Pad and collate
            from src.data import collate_batch
            batch_dict = collate_batch(batch_tokens, device=self.model.cfg.device)

            # Run model and get activations for all layers
            with torch.no_grad():
                hook_names = [f"blocks.{layer}.hook_{self.hook}" for layer in selected_features.keys()]
                _, cache = self.model.run_with_cache(
                    batch_dict["input_ids"],
                    names_filter=hook_names
                )

                # Extract feature activations
                for layer in selected_features.keys():
                    hook_name = f"blocks.{layer}.hook_{self.hook}"
                    acts = cache[hook_name]  # [batch, seq, d_model]

                    # Encode with SAE
                    sae = self.saes[layer]
                    sae_acts = sae.encode(acts)  # [batch, seq, d_sae]

                    # Extract specific features
                    for feature_id in selected_features[layer]:
                        feature_acts = sae_acts[:, :, feature_id]  # [batch, seq]

                        # Flatten and collect non-zero activations
                        flat_acts = feature_acts.flatten().cpu().numpy()
                        non_zero = flat_acts[flat_acts > 0]  # Exclude exact zeros

                        if len(non_zero) > 0:
                            activations_dict[(layer, feature_id)].append(non_zero)

        # Compute thresholds
        for (layer, feature_id), act_list in activations_dict.items():
            if len(act_list) > 0:
                all_acts = np.concatenate(act_list)
                if len(all_acts) > 0:
                    threshold = np.percentile(all_acts, self.live_percentile)
                    self.thresholds[(layer, feature_id)] = threshold
                else:
                    self.logger.warning(
                        f"No activations for layer {layer}, feature {feature_id}. "
                        "Using threshold 0.0"
                    )
                    self.thresholds[(layer, feature_id)] = 0.0
            else:
                self.logger.warning(
                    f"No calibration data for layer {layer}, feature {feature_id}. "
                    "Using threshold 0.0"
                )
                self.thresholds[(layer, feature_id)] = 0.0

        self.logger.info(f"Calibrated {len(self.thresholds)} thresholds")

        # Log some example thresholds
        for (layer, feature_id), threshold in list(self.thresholds.items())[:5]:
            self.logger.info(
                f"  Layer {layer}, Feature {feature_id}: threshold={threshold:.4f}"
            )

        return self.thresholds

    def create_intervention_hook(
        self,
        layer: int,
        feature_id: int,
        alpha: float = 2.0
    ):
        """Create a hook function that intervenes on a specific feature.

        Args:
            layer: Layer index
            feature_id: Feature ID to intervene on
            alpha: Scaling factor for intervention

        Returns:
            Hook function that can be used with model.add_hook()
        """
        sae = self.saes[layer]
        threshold = self.thresholds.get((layer, feature_id), 0.0)

        def hook_fn(activations, hook):
            """Hook function that modifies activations.

            Args:
                activations: Tensor of shape [batch, seq, d_model]
                hook: Hook object

            Returns:
                Modified activations
            """
            # Encode to SAE latent space
            sae_acts = sae.encode(activations)  # [batch, seq, d_sae]

            # Apply intervention to the feature (regardless of magnitude)
            # Note: This applies SAE reconstruction to all positions, but the
            # intervention effect should dominate over reconstruction error
            sae_acts[:, :, feature_id] *= alpha

            # Decode back to activation space
            modified_acts = sae.decode(sae_acts)

            return modified_acts

        return hook_fn

    def get_live_mask(
        self,
        activations: torch.Tensor,
        layer: int,
        feature_id: int
    ) -> torch.Tensor:
        """Get mask of positions where feature is "live".

        Args:
            activations: Original activations [batch, seq, d_model]
            layer: Layer index
            feature_id: Feature ID

        Returns:
            Boolean mask [batch, seq]
        """
        sae = self.saes[layer]
        threshold = self.thresholds.get((layer, feature_id), 0.0)

        with torch.no_grad():
            sae_acts = sae.encode(activations)
            feature_acts = sae_acts[:, :, feature_id]
            live_mask = feature_acts >= threshold

        return live_mask

    def compute_feature_statistics(
        self,
        activations: torch.Tensor,
        layer: int,
        feature_id: int,
        attention_mask: torch.Tensor = None
    ) -> Dict[str, float]:
        """Compute statistics for a feature on given activations.

        Args:
            activations: Activations [batch, seq, d_model]
            layer: Layer index
            feature_id: Feature ID
            attention_mask: Mask [batch, seq] where 1=real token, 0=padding

        Returns:
            Dictionary with statistics
        """
        sae = self.saes[layer]

        with torch.no_grad():
            sae_acts = sae.encode(activations)
            feature_acts = sae_acts[:, :, feature_id]  # [batch, seq]

            # Apply attention mask if provided
            if attention_mask is not None:
                mask = attention_mask.bool()
                valid_acts = feature_acts[mask]
            else:
                valid_acts = feature_acts.flatten()

            # Compute statistics
            mean_act = valid_acts.mean().item()
            max_act = valid_acts.max().item()
            std_act = valid_acts.std().item()

            # Firing rate
            threshold = self.thresholds.get((layer, feature_id), 0.0)
            firing_rate = (valid_acts >= threshold).float().mean().item()

            return {
                "mean_act": mean_act,
                "max_act": max_act,
                "std_act": std_act,
                "firing_rate": firing_rate
            }
