"""Main experiment runner."""

import argparse
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from tqdm import tqdm

from src.data import CorpusLoader, collate_batch
from src.features import FeatureSelector
from src.intervene import FeatureIntervention
from src.metrics import InterventionMetrics
from src.model import ModelLoader
from src.plots import PlotGenerator
from src.report import ReportGenerator
from src.snippets import SnippetExtractor
from src.utils import (
    OOMHandler,
    create_batches,
    get_device,
    load_config,
    log_environment_info,
    parse_layer_spec,
    save_config,
    set_seed,
    setup_logging,
)


def main():
    """Main experiment execution."""
    # Parse arguments
    args = parse_args()

    # Load config
    config = load_config(args.config) if args.config else {}

    # Override config with CLI args
    update_config_from_args(config, args)

    # Setup
    output_dir = Path(config.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (output_dir / "csv").mkdir(exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    (output_dir / "snippets").mkdir(exist_ok=True)

    logger = setup_logging(
        log_file=str(output_dir / "experiment.log")
    )

    logger.info("=" * 80)
    logger.info("Starting SAE Intervention Experiment")
    logger.info("=" * 80)

    # Set seed
    seed = config.get("seed", 1234)
    set_seed(seed)
    logger.info(f"Set random seed: {seed}")

    # Get device
    device = get_device(config.get("device", "auto"))
    logger.info(f"Using device: {device}")

    # Log environment
    env_info = log_environment_info(logger)

    # Save config
    save_config(config, output_dir / "config.yaml")

    # Parse layers
    layers = parse_layer_spec(config.get("layers", "0-11"))
    logger.info(f"Processing layers: {layers}")

    # Step 1: Load model and SAEs
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Loading model and SAEs")
    logger.info("=" * 80)

    model_loader = ModelLoader(
        model_name=config.get("model_name", "gpt2-small"),
        sae_release=config.get("sae_release", "gpt2-small-res-jb"),
        hook=config.get("hook", "resid_post"),
        layers=layers,
        device=device,
        logger=logger
    )

    model = model_loader.load_model()
    saes = model_loader.load_saes()

    # Step 2: Load corpus
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Loading corpus")
    logger.info("=" * 80)

    corpus_loader = CorpusLoader(
        corpus_name=config.get("corpus_name", "wikitext2"),
        max_passages=config.get("max_passages", 200),
        max_len=config.get("max_len", 256),
        tokenizer=model.tokenizer,
        logger=logger
    )

    token_ids_list, text_list = corpus_loader.load_and_tokenize()

    # Split into calibration and experiment
    calibration_list, experiment_list = corpus_loader.create_calibration_split(
        token_ids_list,
        calibration_passages=config.get("calibration_passages", 50)
    )

    # Step 3: Select features
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Selecting features")
    logger.info("=" * 80)

    feature_selector = FeatureSelector(
        feature_source=config.get("feature_source", "csv"),
        layers=layers,
        top_k=config.get("top_k", 5),
        min_conf=config.get("min_conf", 0.0),
        neuronpedia_token=config.get("neuronpedia_token"),
        logger=logger
    )

    # Try to load features CSV
    csv_path = "data/neuronpedia_features.csv"
    if not Path(csv_path).exists():
        logger.warning(f"Features CSV not found at {csv_path}, trying template...")
        csv_path = "data/neuronpedia_features_template.csv"

    features_df = feature_selector.select_features(csv_path=csv_path)

    # Save selected features
    feature_selector.save_selected_features(
        features_df,
        output_dir / "selected_features.csv"
    )

    # Convert to dict: layer -> [feature_ids]
    selected_features = {}
    for layer in layers:
        layer_features = features_df[features_df["layer"] == layer]["feature_id"].tolist()
        if layer_features:
            selected_features[layer] = layer_features

    logger.info(f"Selected {sum(len(v) for v in selected_features.values())} features")

    # Step 4: Calibration
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Calibrating feature thresholds")
    logger.info("=" * 80)

    intervention_manager = FeatureIntervention(
        model=model,
        saes=saes,
        hook=config.get("hook", "resid_pre"),
        live_percentile=config.get("live_percentile", 90),
        logger=logger
    )

    oom_handler = OOMHandler(
        initial_batch_size=config.get("batch_size", 16),
        logger=logger
    )

    intervention_manager.calibrate_thresholds(
        calibration_data=calibration_list,
        selected_features=selected_features,
        batch_size=oom_handler.get_batch_size()
    )

    # Step 5: Run interventions
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: Running interventions")
    logger.info("=" * 80)

    alpha_sweep = config.get("alpha_sweep", [config.get("alpha", 2.0)])
    if not isinstance(alpha_sweep, list):
        alpha_sweep = [alpha_sweep]

    logger.info(f"Alpha values: {alpha_sweep}")

    results = run_interventions(
        model=model,
        saes=saes,
        intervention_manager=intervention_manager,
        experiment_data=experiment_list,
        text_list=text_list[len(calibration_list):],
        selected_features=selected_features,
        features_df=features_df,
        alpha_sweep=alpha_sweep,
        oom_handler=oom_handler,
        config=config,
        logger=logger
    )

    # Step 6: Generate outputs
    logger.info("\n" + "=" * 80)
    logger.info("Step 6: Generating outputs")
    logger.info("=" * 80)

    # Save CSVs
    per_feature_df = pd.DataFrame(results["per_feature"])
    per_feature_df.to_csv(output_dir / "csv" / "per_feature_metrics.csv", index=False)
    logger.info(f"Saved per-feature metrics: {len(per_feature_df)} rows")

    # Aggregate by layer
    per_layer_df = aggregate_by_layer(per_feature_df)
    per_layer_df.to_csv(output_dir / "csv" / "per_layer_summary.csv", index=False)
    logger.info(f"Saved per-layer summary: {len(per_layer_df)} rows")

    # Save snippets
    snippet_extractor = SnippetExtractor(
        tokenizer=model.tokenizer,
        window_size=config.get("snippet_window", 20),
        num_examples=config.get("num_snippet_examples", 5),
        logger=logger
    )

    all_snippets = results.get("snippets", [])
    snippet_extractor.save_snippets(
        all_snippets,
        output_dir / "snippets" / "examples.jsonl"
    )

    # Generate plots
    plot_generator = PlotGenerator(
        output_dir=output_dir / "plots",
        logger=logger
    )

    default_alpha = config.get("alpha", 2.0)
    plot_paths = plot_generator.create_summary_plots(
        per_feature_df,
        alpha_value=default_alpha
    )

    # Generate report
    report_generator = ReportGenerator(
        output_path=output_dir / "report.md",
        logger=logger
    )

    report_generator.generate_full_report(
        config=config,
        env_info=env_info,
        features_df=features_df,
        per_feature_df=per_feature_df,
        per_layer_df=per_layer_df,
        plot_paths=plot_paths,
        snippets=all_snippets,
        alpha_value=default_alpha
    )

    logger.info("\n" + "=" * 80)
    logger.info("Experiment complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)


def run_interventions(
    model,
    saes,
    intervention_manager,
    experiment_data,
    text_list,
    selected_features,
    features_df,
    alpha_sweep,
    oom_handler,
    config,
    logger
):
    """Run interventions for all features and alpha values."""
    metrics_calculator = InterventionMetrics(logger=logger)
    snippet_extractor = SnippetExtractor(
        tokenizer=model.tokenizer,
        window_size=config.get("snippet_window", 20),
        num_examples=config.get("num_snippet_examples", 5),
        logger=logger
    )

    results = {
        "per_feature": [],
        "snippets": []
    }

    # Get feature metadata
    feature_metadata = {}
    for _, row in features_df.iterrows():
        feature_metadata[(row["layer"], row["feature_id"])] = {
            "label": row["label"],
            "label_conf": row["label_confidence"]
        }

    # Iterate over all (layer, feature, alpha) combinations
    total_experiments = sum(len(feats) for feats in selected_features.values()) * len(alpha_sweep)
    pbar = tqdm(total=total_experiments, desc="Running interventions")

    for layer in sorted(selected_features.keys()):
        for feature_id in selected_features[layer]:
            for alpha in alpha_sweep:
                try:
                    result = run_single_intervention(
                        model=model,
                        saes=saes,
                        intervention_manager=intervention_manager,
                        experiment_data=experiment_data,
                        text_list=text_list,
                        layer=layer,
                        feature_id=feature_id,
                        alpha=alpha,
                        metrics_calculator=metrics_calculator,
                        snippet_extractor=snippet_extractor,
                        oom_handler=oom_handler,
                        logger=logger
                    )

                    # Add metadata
                    metadata = feature_metadata.get((layer, feature_id), {})
                    result.update(metadata)

                    results["per_feature"].append(result)

                    # Extract snippets (only for default alpha)
                    if alpha == config.get("alpha", 2.0) and result.get("snippets"):
                        results["snippets"].extend(result["snippets"])

                except Exception as e:
                    logger.error(
                        f"Failed intervention L{layer}F{feature_id}α{alpha}: {e}"
                    )
                    # Record failure
                    results["per_feature"].append({
                        "layer": layer,
                        "feature_id": feature_id,
                        "alpha": alpha,
                        "error": str(e)
                    })

                pbar.update(1)

    pbar.close()

    return results


def run_single_intervention(
    model,
    saes,
    intervention_manager,
    experiment_data,
    text_list,
    layer,
    feature_id,
    alpha,
    metrics_calculator,
    snippet_extractor,
    oom_handler,
    logger
):
    """Run intervention for a single (layer, feature, alpha) combination."""
    batch_size = oom_handler.get_batch_size()
    batches = create_batches(experiment_data, batch_size)

    # Accumulate metrics across batches
    all_metrics = []
    all_stats = []
    all_snippets = []

    hook_name = f"blocks.{layer}.hook_{intervention_manager.hook}"

    for batch_idx, batch_tokens in enumerate(batches):
        # Collate batch
        batch_dict = collate_batch(batch_tokens, device=model.cfg.device)

        # Run clean forward pass
        with torch.no_grad():
            clean_logits = model(batch_dict["input_ids"])

        # Run intervention
        hook_fn = intervention_manager.create_intervention_hook(layer, feature_id, alpha)

        with torch.no_grad():
            with model.hooks([(hook_name, hook_fn)]):
                edited_logits = model(batch_dict["input_ids"])

        # Get live mask (need to recompute activations)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                batch_dict["input_ids"],
                names_filter=[hook_name]
            )
            live_mask = intervention_manager.get_live_mask(
                cache[hook_name],
                layer,
                feature_id
            )

            # Compute statistics
            stats = intervention_manager.compute_feature_statistics(
                cache[hook_name],
                layer,
                feature_id,
                batch_dict["attention_mask"]
            )
            all_stats.append(stats)

        # Compute metrics
        metrics = metrics_calculator.compute_all_metrics(
            clean_logits=clean_logits,
            edited_logits=edited_logits,
            targets=batch_dict["input_ids"],
            attention_mask=batch_dict["attention_mask"],
            live_mask=live_mask
        )
        all_metrics.append(metrics)

        # Extract snippets (only first few batches)
        if batch_idx < 3 and alpha == 2.0:  # Limit snippet extraction
            kl_token_wise = metrics_calculator.compute_token_wise_kl(
                clean_logits, edited_logits
            )

            snippets = snippet_extractor.extract_snippets(
                token_ids_batch=batch_dict["input_ids"],
                kl_values=kl_token_wise,
                live_mask=live_mask,
                layer=layer,
                feature_id=feature_id,
                alpha=alpha,
                text_sources=text_list[batch_idx * oom_handler.get_batch_size():(batch_idx + 1) * oom_handler.get_batch_size()] if text_list else None
            )
            all_snippets.extend(snippets)

    # Average metrics across batches
    avg_metrics = average_metrics(all_metrics)
    avg_stats = average_stats(all_stats)

    # Combine into result
    result = {
        "layer": layer,
        "feature_id": feature_id,
        "alpha": alpha,
        **avg_metrics,
        **avg_stats,
        "snippets": all_snippets
    }

    return result


def average_metrics(metrics_list: List[Dict]) -> Dict:
    """Average metrics across batches."""
    if not metrics_list:
        return {}

    keys = metrics_list[0].keys()
    avg = {}

    for key in keys:
        values = [m[key] for m in metrics_list if key in m]
        if values:
            avg[key] = sum(values) / len(values)

    return avg


def average_stats(stats_list: List[Dict]) -> Dict:
    """Average statistics across batches."""
    return average_metrics(stats_list)


def aggregate_by_layer(per_feature_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-feature metrics by layer."""
    # Group by layer and alpha
    grouped = per_feature_df.groupby(["layer", "alpha"])

    # Compute mean/std for each metric
    agg_dict = {
        "d_loss": ["mean", "std"],
        "mean_logit_delta": ["mean", "std"],
        "kl_avg": ["mean", "std"],
        "firing_rate": ["mean", "std"],
        "mean_act": ["mean", "std"],
        "max_act": ["max"]
    }

    # Only include metrics that exist
    agg_dict = {k: v for k, v in agg_dict.items() if k in per_feature_df.columns}

    summary = grouped.agg(agg_dict).reset_index()

    # Flatten column names
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]

    return summary


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SAE Intervention Experiment"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file"
    )

    # Model
    parser.add_argument("--model", dest="model_name", type=str, help="Model name")
    parser.add_argument("--sae-release", dest="sae_release", type=str, help="SAE release")
    parser.add_argument("--hook", type=str, help="Hook point type")
    parser.add_argument("--layers", type=str, help="Layer specification")

    # Intervention
    parser.add_argument("--alpha", type=float, help="Alpha value")
    parser.add_argument("--alpha-sweep", dest="alpha_sweep", type=str, help="Alpha sweep (comma-separated)")

    # Corpus
    parser.add_argument("--corpus", dest="corpus_name", type=str, help="Corpus name")
    parser.add_argument("--max-passages", dest="max_passages", type=int, help="Max passages")
    parser.add_argument("--max-len", dest="max_len", type=int, help="Max token length")

    # Features
    parser.add_argument("--feature-source", dest="feature_source", type=str, help="Feature source")
    parser.add_argument("--neuronpedia-token", dest="neuronpedia_token", type=str, help="Neuronpedia API token")

    # Computation
    parser.add_argument("--device", type=str, help="Device")
    parser.add_argument("--batch-size", dest="batch_size", type=int, help="Batch size")
    parser.add_argument("--seed", type=int, help="Random seed")

    # Output
    parser.add_argument("--output-dir", dest="output_dir", type=str, help="Output directory")

    return parser.parse_args()


def update_config_from_args(config: Dict, args: argparse.Namespace) -> None:
    """Update config with CLI arguments."""
    for key, value in vars(args).items():
        if value is not None and key != "config":
            # Handle alpha_sweep special case
            if key == "alpha_sweep" and isinstance(value, str):
                config[key] = [float(x.strip()) for x in value.split(",")]
            else:
                config[key] = value


if __name__ == "__main__":
    main()
