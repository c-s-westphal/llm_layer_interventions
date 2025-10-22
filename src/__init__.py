"""SAE Intervention Pipeline.

A reproducible pipeline for performing feature interventions on GPT-2 Small
using Sparse Autoencoders (SAEs).
"""

__version__ = "0.1.0"

from src.data import CorpusLoader
from src.features import FeatureSelector
from src.intervene import FeatureIntervention
from src.metrics import InterventionMetrics
from src.model import ModelLoader
from src.plots import PlotGenerator
from src.report import ReportGenerator
from src.snippets import SnippetExtractor

__all__ = [
    "CorpusLoader",
    "FeatureSelector",
    "FeatureIntervention",
    "InterventionMetrics",
    "ModelLoader",
    "PlotGenerator",
    "ReportGenerator",
    "SnippetExtractor",
]
