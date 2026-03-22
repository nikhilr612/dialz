"""Enum types for steering methods and scoring methods."""

from enum import Enum


class Method(str, Enum):
    """Steering vector extraction strategy.

    Attributes:
        PCA: First principal component of positive-negative differences.
        PCA_CENTER: PCA on class-centered hidden states.
        UMAP: UMAP-weighted activation sum (experimental).
        MEAN_DIFF: Mean of positive-negative differences.
    """

    PCA = "pca"
    PCA_CENTER = "pca_center"
    UMAP = "umap"
    MEAN_DIFF = "mean_diff"


class ScoringMethod(str, Enum):
    """Aggregation strategy for token-level activation scores.

    Attributes:
        MEAN: Average dot product across all tokens.
        FINAL_TOKEN: Dot product of the final token only.
        MAX_TOKEN: Maximum dot product across tokens.
        MEDIAN_TOKEN: Median dot product across tokens.
    """

    MEAN = "mean"
    FINAL_TOKEN = "final_token"
    MAX_TOKEN = "max_token"
    MEDIAN_TOKEN = "median_token"
