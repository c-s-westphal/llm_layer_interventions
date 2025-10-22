"""Feature selection from Neuronpedia or CSV."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests


class FeatureSelector:
    """Select interpretable features from Neuronpedia or CSV."""

    def __init__(
        self,
        feature_source: str = "csv",
        layers: List[int] = None,
        top_k: int = 5,
        min_conf: float = 0.0,
        neuronpedia_token: Optional[str] = None,
        model_id: str = "gpt2-small",
        sae_release: str = "res-jb",
        logger: logging.Logger = None
    ):
        """Initialize feature selector.

        Args:
            feature_source: "csv", "neuronpedia_api", or "static"
            layers: List of layer indices
            top_k: Number of features to select per layer
            min_conf: Minimum label confidence (0.0 allows <0.7)
            neuronpedia_token: API token for Neuronpedia
            model_id: Model identifier for Neuronpedia
            sae_release: SAE release name
            logger: Logger instance
        """
        self.feature_source = feature_source
        self.layers = layers or list(range(12))
        self.top_k = top_k
        self.min_conf = min_conf
        self.neuronpedia_token = neuronpedia_token
        self.model_id = model_id
        self.sae_release = sae_release
        self.logger = logger or logging.getLogger("sae_interventions")

    def select_features(
        self,
        csv_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Select features based on configured source.

        Args:
            csv_path: Path to CSV file (required if feature_source="csv")

        Returns:
            DataFrame with columns: layer, feature_id, label, label_confidence
        """
        self.logger.info(f"Selecting features via {self.feature_source}...")

        if self.feature_source == "csv":
            if csv_path is None:
                raise ValueError("csv_path required when feature_source='csv'")
            features_df = self._load_from_csv(csv_path)

        elif self.feature_source == "neuronpedia_api":
            if self.neuronpedia_token is None:
                self.logger.warning(
                    "No Neuronpedia API token provided. "
                    "Falling back to CSV method."
                )
                if csv_path is None:
                    raise ValueError("Need either API token or csv_path")
                features_df = self._load_from_csv(csv_path)
            else:
                features_df = self._fetch_from_api()

        elif self.feature_source == "static":
            features_df = self._load_static_features()

        else:
            raise ValueError(f"Unknown feature_source: {self.feature_source}")

        # Validate and filter
        features_df = self._validate_and_filter(features_df)

        self.logger.info(
            f"Selected {len(features_df)} features across {len(features_df['layer'].unique())} layers"
        )

        return features_df

    def _load_from_csv(self, csv_path: str) -> pd.DataFrame:
        """Load features from CSV file."""
        self.logger.info(f"Loading features from {csv_path}")

        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Validate columns
        required_cols = {"layer", "feature_id", "label", "label_confidence"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"CSV must contain columns: {required_cols}. "
                f"Found: {set(df.columns)}"
            )

        return df

    def _fetch_from_api(self) -> pd.DataFrame:
        """Fetch features from Neuronpedia API.

        Note: This is a placeholder implementation since the API doesn't
        directly support "list by confidence". In practice, you would need
        to scrape or use an undocumented endpoint.
        """
        self.logger.info("Fetching features from Neuronpedia API...")

        features = []
        base_url = "https://www.neuronpedia.org/api/feature"

        headers = {}
        if self.neuronpedia_token:
            headers["x-api-key"] = self.neuronpedia_token

        # NOTE: This is a simplified approach. The actual API may not support
        # listing all features. You may need to:
        # 1. Use the search endpoint with sample text
        # 2. Scrape the web interface
        # 3. Pre-generate a CSV from the Neuronpedia dashboard

        for layer in self.layers:
            # Construct SAE ID (format varies by release)
            sae_id = f"{layer}-{self.sae_release}"

            # Try to fetch feature metadata
            # This is pseudo-code - actual endpoint may differ
            try:
                url = f"{base_url}/{self.model_id}/{sae_id}/list"
                response = requests.get(url, headers=headers, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    # Parse response and extract top_k by confidence
                    # Exact format depends on API response structure
                    layer_features = self._parse_api_response(data, layer)
                    features.extend(layer_features)
                else:
                    self.logger.warning(
                        f"Failed to fetch features for layer {layer}: "
                        f"Status {response.status_code}"
                    )
            except Exception as e:
                self.logger.error(f"Error fetching layer {layer}: {e}")

        if not features:
            raise RuntimeError(
                "Failed to fetch any features from API. "
                "Consider using CSV fallback."
            )

        return pd.DataFrame(features)

    def _parse_api_response(self, data: dict, layer: int) -> List[Dict]:
        """Parse API response to extract feature info.

        This is a placeholder - actual parsing depends on API structure.
        """
        # Example structure (adjust based on actual API):
        features = []

        if "features" in data:
            for feature in data["features"]:
                features.append({
                    "layer": layer,
                    "feature_id": feature.get("index", feature.get("id")),
                    "label": feature.get("label", "unknown"),
                    "label_confidence": feature.get("confidence", 0.0)
                })

        # Sort by confidence and take top_k
        features = sorted(features, key=lambda x: x["label_confidence"], reverse=True)
        return features[:self.top_k]

    def _load_static_features(self) -> pd.DataFrame:
        """Load features from static configuration (hardcoded).

        This is useful for testing with known features.
        """
        # Example static features
        features = []
        for layer in self.layers:
            for i in range(self.top_k):
                features.append({
                    "layer": layer,
                    "feature_id": i * 1000,  # Placeholder IDs
                    "label": f"layer{layer}_feature{i}",
                    "label_confidence": 0.75 - (i * 0.05)
                })

        return pd.DataFrame(features)

    def _validate_and_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and filter features based on criteria."""
        # Filter by layers
        df = df[df["layer"].isin(self.layers)].copy()

        # Filter by minimum confidence
        if self.min_conf > 0:
            df = df[df["label_confidence"] >= self.min_conf]

        # Sort by confidence within each layer
        df = df.sort_values(["layer", "label_confidence"], ascending=[True, False])

        # Take top_k per layer
        df = df.groupby("layer").head(self.top_k).reset_index(drop=True)

        # Warn if any layer has fewer than top_k features
        layer_counts = df["layer"].value_counts()
        for layer in self.layers:
            count = layer_counts.get(layer, 0)
            if count < self.top_k:
                self.logger.warning(
                    f"Layer {layer} has only {count}/{self.top_k} features "
                    f"(min_conf={self.min_conf})"
                )

        return df

    def save_selected_features(self, df: pd.DataFrame, output_path: str) -> None:
        """Save selected features to CSV."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved selected features to {output_path}")
