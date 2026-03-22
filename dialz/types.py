from enum import Enum


class Method(str, Enum):
    PCA = "pca"
    PCA_CENTER = "pca_center"
    UMAP = "umap"
    MEAN_DIFF = "mean_diff"


class ScoringMethod(str, Enum):
    MEAN = "mean"
    FINAL_TOKEN = "final_token"
    MAX_TOKEN = "max_token"
    MEDIAN_TOKEN = "median_token"
