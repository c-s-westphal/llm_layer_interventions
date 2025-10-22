"""Data loading utilities for corpus preparation."""

import logging
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm


class CorpusLoader:
    """Load and tokenize text corpus for SAE experiments."""

    def __init__(
        self,
        corpus_name: str = "wikitext2",
        max_passages: int = 200,
        max_len: int = 256,
        tokenizer: any = None,
        logger: logging.Logger = None
    ):
        """Initialize corpus loader.

        Args:
            corpus_name: Name of corpus to load (currently only "wikitext2")
            max_passages: Maximum number of passages to use
            max_len: Maximum token length per passage
            tokenizer: Tokenizer from transformer_lens model
            logger: Logger instance
        """
        self.corpus_name = corpus_name
        self.max_passages = max_passages
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.logger = logger or logging.getLogger("sae_interventions")

    def load_and_tokenize(self) -> Tuple[List[torch.Tensor], List[str]]:
        """Load corpus and tokenize.

        Returns:
            Tuple of (token_ids_list, text_list)
        """
        self.logger.info(f"Loading {self.corpus_name} corpus...")

        if self.corpus_name.lower() == "wikitext2":
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        else:
            raise ValueError(f"Unsupported corpus: {self.corpus_name}")

        # Filter out empty texts
        texts = [
            text for text in dataset["text"]
            if text.strip() and len(text.strip()) > 50  # Skip very short passages
        ]

        self.logger.info(f"Found {len(texts)} non-empty passages")

        # Limit number of passages
        texts = texts[:self.max_passages]

        # Tokenize
        self.logger.info(f"Tokenizing {len(texts)} passages...")
        token_ids_list = []
        valid_texts = []

        for text in tqdm(texts, desc="Tokenizing"):
            # Tokenize
            tokens = self.tokenizer.encode(text)

            # Convert to tensor and truncate
            if isinstance(tokens, list):
                tokens = torch.tensor(tokens)

            # Truncate to max_len
            if len(tokens) > self.max_len:
                tokens = tokens[:self.max_len]

            # Skip if too short (need at least 2 tokens for next-token prediction)
            if len(tokens) < 2:
                continue

            token_ids_list.append(tokens)
            valid_texts.append(text)

        self.logger.info(
            f"Prepared {len(token_ids_list)} passages "
            f"(avg length: {sum(len(t) for t in token_ids_list) / len(token_ids_list):.1f} tokens)"
        )

        return token_ids_list, valid_texts

    def create_calibration_split(
        self,
        token_ids_list: List[torch.Tensor],
        calibration_passages: int = 50
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Split corpus into calibration and experiment sets.

        Args:
            token_ids_list: List of tokenized passages
            calibration_passages: Number of passages for calibration

        Returns:
            Tuple of (calibration_list, experiment_list)
        """
        calibration_size = min(calibration_passages, len(token_ids_list) // 4)
        calibration_list = token_ids_list[:calibration_size]
        experiment_list = token_ids_list[calibration_size:]

        self.logger.info(
            f"Split corpus: {len(calibration_list)} calibration, "
            f"{len(experiment_list)} experiment passages"
        )

        return calibration_list, experiment_list


def collate_batch(
    token_ids_list: List[torch.Tensor],
    pad_token_id: int = 50256,  # GPT-2 EOS token as padding
    device: torch.device = None
) -> Dict[str, torch.Tensor]:
    """Collate list of token sequences into padded batch.

    Args:
        token_ids_list: List of token ID tensors
        pad_token_id: Token ID to use for padding
        device: Device to place tensors on

    Returns:
        Dictionary with 'input_ids' and 'attention_mask'
    """
    if len(token_ids_list) == 0:
        raise ValueError("Cannot collate empty list")

    # Find max length in batch
    max_len = max(len(tokens) for tokens in token_ids_list)

    # Pad sequences
    padded_ids = []
    attention_masks = []

    for tokens in token_ids_list:
        # Create attention mask (1 for real tokens, 0 for padding)
        mask = torch.ones(len(tokens), dtype=torch.long)

        # Pad if necessary
        if len(tokens) < max_len:
            padding_len = max_len - len(tokens)
            tokens = torch.cat([
                tokens,
                torch.full((padding_len,), pad_token_id, dtype=tokens.dtype)
            ])
            mask = torch.cat([
                mask,
                torch.zeros(padding_len, dtype=torch.long)
            ])

        padded_ids.append(tokens)
        attention_masks.append(mask)

    # Stack into batch
    batch_ids = torch.stack(padded_ids)
    batch_masks = torch.stack(attention_masks)

    if device is not None:
        batch_ids = batch_ids.to(device)
        batch_masks = batch_masks.to(device)

    return {
        "input_ids": batch_ids,
        "attention_mask": batch_masks
    }
