"""Utility functions for logging, seeding, batching, and error handling."""

import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Set up logging to console and optionally to file."""
    logger = logging.getLogger("sae_interventions")
    logger.setLevel(level)
    logger.handlers = []  # Clear any existing handlers

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "auto") -> torch.device:
    """Get torch device based on availability."""
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    return device


def get_gpu_memory_info() -> Dict[str, float]:
    """Get current GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return {"total": 0.0, "allocated": 0.0, "free": 0.0}

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    free = total - reserved

    return {
        "total": total,
        "allocated": allocated,
        "reserved": reserved,
        "free": free
    }


def log_environment_info(logger: logging.Logger) -> Dict[str, Any]:
    """Log environment and library versions."""
    import platform
    import torch
    import transformer_lens

    try:
        import sae_lens
        sae_lens_version = sae_lens.__version__
    except (ImportError, AttributeError):
        sae_lens_version = "unknown"

    env_info = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformer_lens_version": transformer_lens.__version__,
        "sae_lens_version": sae_lens_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }

    if torch.cuda.is_available():
        mem_info = get_gpu_memory_info()
        env_info["gpu_memory_total_gb"] = mem_info["total"]

    logger.info("Environment information:")
    for key, value in env_info.items():
        logger.info(f"  {key}: {value}")

    return env_info


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save configuration to YAML file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def parse_layer_spec(layer_spec: str) -> List[int]:
    """Parse layer specification string.

    Args:
        layer_spec: String like "0-11" or "0,3,6,9"

    Returns:
        List of layer indices
    """
    if '-' in layer_spec:
        # Range specification like "0-11"
        start, end = map(int, layer_spec.split('-'))
        return list(range(start, end + 1))
    else:
        # Comma-separated like "0,3,6,9"
        return [int(x.strip()) for x in layer_spec.split(',')]


class OOMHandler:
    """Handle out-of-memory errors by reducing batch size."""

    def __init__(self, initial_batch_size: int, min_batch_size: int = 1, logger: Optional[logging.Logger] = None):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.logger = logger or logging.getLogger("sae_interventions")
        self.oom_count = 0

    def handle_oom(self) -> bool:
        """Handle OOM by reducing batch size.

        Returns:
            True if batch size was reduced, False if cannot reduce further
        """
        self.oom_count += 1
        new_batch_size = max(self.min_batch_size, self.current_batch_size // 2)

        if new_batch_size < self.current_batch_size:
            self.logger.warning(
                f"OOM error (#{self.oom_count}). Reducing batch size: "
                f"{self.current_batch_size} -> {new_batch_size}"
            )
            self.current_batch_size = new_batch_size

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return True
        else:
            self.logger.error(
                f"Cannot reduce batch size further (already at {self.current_batch_size}). "
                "Consider using a smaller model or reducing sequence length."
            )
            return False

    def get_batch_size(self) -> int:
        """Get current batch size."""
        return self.current_batch_size


def create_batches(items: List[Any], batch_size: int) -> List[List[Any]]:
    """Split items into batches."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def safe_decode_text(token_ids: List[int], tokenizer: Any, errors: str = 'ignore') -> str:
    """Safely decode token IDs to text, handling UTF-8 errors."""
    try:
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        # Try to encode back to check for errors
        text.encode('utf-8', errors=errors)
        return text
    except (UnicodeDecodeError, UnicodeEncodeError):
        # Fallback: decode with error handling
        return tokenizer.decode(token_ids, skip_special_tokens=True, errors=errors)
