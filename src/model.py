"""Model and SAE loading utilities."""

import logging
from typing import Dict, List

import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer


class ModelLoader:
    """Load GPT-2 model and SAEs."""

    def __init__(
        self,
        model_name: str = "gpt2-small",
        sae_release: str = "gpt2-small-res-jb",
        hook: str = "resid_post",
        layers: List[int] = None,
        device: torch.device = None,
        logger: logging.Logger = None
    ):
        """Initialize model loader.

        Args:
            model_name: Name of model to load
            sae_release: SAE release identifier
            hook: Hook point type ("resid_post" or "resid_pre")
            layers: List of layer indices to load SAEs for
            device: Device to load models on
            logger: Logger instance
        """
        self.model_name = model_name
        self.sae_release = sae_release
        self.hook = hook
        self.layers = layers or list(range(12))
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger or logging.getLogger("sae_interventions")

        self.model = None
        self.saes = {}

    def load_model(self) -> HookedTransformer:
        """Load base language model."""
        self.logger.info(f"Loading model: {self.model_name}")

        self.model = HookedTransformer.from_pretrained(
            self.model_name,
            device=self.device
        )

        self.logger.info(
            f"Model loaded: {self.model_name} "
            f"(n_layers={self.model.cfg.n_layers}, d_model={self.model.cfg.d_model})"
        )

        return self.model

    def load_saes(self) -> Dict[int, SAE]:
        """Load SAEs for all specified layers.

        Returns:
            Dictionary mapping layer index to SAE
        """
        self.logger.info(
            f"Loading SAEs from release: {self.sae_release} "
            f"(hook={self.hook}, layers={self.layers})"
        )

        for layer in self.layers:
            try:
                # Construct hook point string
                # SAE-Lens expects format like "blocks.{L}.hook_resid_pre"
                hook_point = f"blocks.{layer}.hook_{self.hook}"

                self.logger.info(f"Loading SAE for layer {layer} (hook_point={hook_point})")

                sae = SAE.from_pretrained(
                    release=self.sae_release,
                    sae_id=hook_point,
                    device=str(self.device)
                )

                self.saes[layer] = sae

                self.logger.info(
                    f"  Layer {layer} SAE loaded: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}"
                )

            except Exception as e:
                self.logger.error(f"Failed to load SAE for layer {layer}: {e}")
                # Continue with other layers
                continue

        if not self.saes:
            raise RuntimeError("Failed to load any SAEs")

        self.logger.info(f"Successfully loaded {len(self.saes)} SAEs")

        return self.saes

    def get_hook_name(self, layer: int) -> str:
        """Get TransformerLens hook name for a layer.

        Args:
            layer: Layer index

        Returns:
            Hook name like "blocks.{layer}.hook_resid_post"
        """
        return f"blocks.{layer}.hook_{self.hook}"

    def get_sae_hook_point(self, layer: int) -> str:
        """Get SAE-Lens hook point for a layer.

        Args:
            layer: Layer index

        Returns:
            Hook point like "blocks.{layer}.hook_resid_pre"
        """
        return f"blocks.{layer}.hook_{self.hook}"


def test_sae_reconstruction(
    model: HookedTransformer,
    sae: SAE,
    layer: int,
    hook: str,
    sample_text: str = "The quick brown fox jumps over the lazy dog.",
    logger: logging.Logger = None
) -> Dict[str, float]:
    """Test SAE reconstruction quality on sample text.

    Args:
        model: HookedTransformer model
        sae: SAE instance
        layer: Layer index
        hook: Hook type ("resid_post" or "resid_pre")
        sample_text: Text to test on
        logger: Logger instance

    Returns:
        Dictionary with reconstruction metrics
    """
    logger = logger or logging.getLogger("sae_interventions")

    # Tokenize
    tokens = model.to_tokens(sample_text)

    # Run model and get activations
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=[f"blocks.{layer}.hook_{hook}"])

        # Get original activations
        hook_name = f"blocks.{layer}.hook_{hook}"
        original_acts = cache[hook_name]

        # Encode and decode with SAE
        sae_acts = sae.encode(original_acts)
        reconstructed_acts = sae.decode(sae_acts)

        # Compute reconstruction error
        mse = torch.mean((original_acts - reconstructed_acts) ** 2).item()
        rel_error = (mse / torch.mean(original_acts ** 2).item()) ** 0.5

        # Feature statistics
        num_active = (sae_acts > 0).sum().item()
        total_features = sae_acts.numel()
        sparsity = 1.0 - (num_active / total_features)

        metrics = {
            "mse": mse,
            "relative_error": rel_error,
            "sparsity": sparsity,
            "num_active_features": num_active,
            "total_features": total_features
        }

        logger.info(
            f"Layer {layer} SAE reconstruction test: "
            f"MSE={mse:.6f}, RelErr={rel_error:.4f}, Sparsity={sparsity:.4f}"
        )

        return metrics
