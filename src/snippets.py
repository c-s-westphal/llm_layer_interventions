"""Snippet extraction for qualitative analysis."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from src.utils import safe_decode_text


class SnippetExtractor:
    """Extract text snippets around high/low KL positions."""

    def __init__(
        self,
        tokenizer: Any,
        window_size: int = 20,
        num_examples: int = 5,
        logger: logging.Logger = None
    ):
        """Initialize snippet extractor.

        Args:
            tokenizer: Tokenizer from model
            window_size: Number of tokens before/after target position
            num_examples: Number of max/min examples to extract
            logger: Logger instance
        """
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.num_examples = num_examples
        self.logger = logger or logging.getLogger("sae_interventions")

    def extract_snippets(
        self,
        token_ids_batch: torch.Tensor,
        kl_values: torch.Tensor,
        live_mask: torch.Tensor,
        layer: int,
        feature_id: int,
        alpha: float,
        text_sources: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Extract max and min KL snippets where feature was live.

        Args:
            token_ids_batch: Token IDs [batch, seq]
            kl_values: Token-wise KL divergences [batch, seq]
            live_mask: Mask where feature was live [batch, seq]
            layer: Layer index
            feature_id: Feature ID
            alpha: Alpha value used
            text_sources: Optional list of source texts

        Returns:
            List of snippet dictionaries
        """
        snippets = []

        # Get positions where feature was live
        live_positions = torch.where(live_mask)

        if len(live_positions[0]) == 0:
            self.logger.warning(
                f"No live positions for layer {layer}, feature {feature_id}"
            )
            return snippets

        # Get KL values at live positions
        live_kl = kl_values[live_positions]

        # Find top-k max and min
        num_live = len(live_kl)
        k = min(self.num_examples, num_live)

        if k == 0:
            return snippets

        # Get indices for max KL
        _, max_indices = torch.topk(live_kl, k=k, largest=True)

        # Get indices for min KL
        _, min_indices = torch.topk(live_kl, k=k, largest=False)

        # Extract snippets for max KL
        for idx in max_indices:
            batch_idx = live_positions[0][idx].item()
            token_idx = live_positions[1][idx].item()
            kl_val = live_kl[idx].item()

            snippet = self._extract_snippet(
                token_ids_batch[batch_idx],
                token_idx,
                kl_val,
                layer,
                feature_id,
                alpha,
                snippet_type="max_kl",
                source_text=text_sources[batch_idx] if text_sources else None
            )
            snippets.append(snippet)

        # Extract snippets for min KL
        for idx in min_indices:
            batch_idx = live_positions[0][idx].item()
            token_idx = live_positions[1][idx].item()
            kl_val = live_kl[idx].item()

            snippet = self._extract_snippet(
                token_ids_batch[batch_idx],
                token_idx,
                kl_val,
                layer,
                feature_id,
                alpha,
                snippet_type="min_kl",
                source_text=text_sources[batch_idx] if text_sources else None
            )
            snippets.append(snippet)

        return snippets

    def _extract_snippet(
        self,
        token_ids: torch.Tensor,
        token_idx: int,
        kl_value: float,
        layer: int,
        feature_id: int,
        alpha: float,
        snippet_type: str,
        source_text: str = None
    ) -> Dict[str, Any]:
        """Extract a single snippet around a token position.

        Args:
            token_ids: Token IDs [seq]
            token_idx: Index of target token
            kl_value: KL value at this position
            layer: Layer index
            feature_id: Feature ID
            alpha: Alpha value
            snippet_type: "max_kl" or "min_kl"
            source_text: Optional source text

        Returns:
            Snippet dictionary
        """
        # Determine window bounds
        start_idx = max(0, token_idx - self.window_size)
        end_idx = min(len(token_ids), token_idx + self.window_size + 1)

        # Extract window
        window_token_ids = token_ids[start_idx:end_idx].cpu().tolist()

        # Decode text
        text_window = safe_decode_text(window_token_ids, self.tokenizer)

        # Get the target token text
        target_token_id = token_ids[token_idx].item()
        target_token_text = safe_decode_text([target_token_id], self.tokenizer)

        return {
            "layer": layer,
            "feature_id": feature_id,
            "alpha": alpha,
            "kl_value": kl_value,
            "snippet_type": snippet_type,
            "token_index": token_idx,
            "window_start": start_idx,
            "window_end": end_idx,
            "text_window": text_window,
            "target_token": target_token_text,
            "source_text": source_text[:200] if source_text else None  # First 200 chars
        }

    def save_snippets(
        self,
        snippets: List[Dict[str, Any]],
        output_path: str
    ) -> None:
        """Save snippets to JSONL file.

        Args:
            snippets: List of snippet dictionaries
            output_path: Path to output file
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for snippet in snippets:
                f.write(json.dumps(snippet) + '\n')

        self.logger.info(f"Saved {len(snippets)} snippets to {output_path}")

    def load_snippets(self, input_path: str) -> List[Dict[str, Any]]:
        """Load snippets from JSONL file.

        Args:
            input_path: Path to input file

        Returns:
            List of snippet dictionaries
        """
        snippets = []

        with open(input_path, 'r') as f:
            for line in f:
                snippets.append(json.loads(line))

        self.logger.info(f"Loaded {len(snippets)} snippets from {input_path}")

        return snippets

    def format_snippet_for_display(self, snippet: Dict[str, Any]) -> str:
        """Format snippet for display in report.

        Args:
            snippet: Snippet dictionary

        Returns:
            Formatted string
        """
        lines = [
            f"**Layer {snippet['layer']}, Feature {snippet['feature_id']}, α={snippet['alpha']:.1f}**",
            f"Type: {snippet['snippet_type']}, KL: {snippet['kl_value']:.4f}",
            f"Target token: `{snippet['target_token']}`",
            f"",
            f"```",
            f"{snippet['text_window']}",
            f"```",
            ""
        ]

        return '\n'.join(lines)
