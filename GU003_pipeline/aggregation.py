import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis


def aggregate(cell_df, output_dir):

    patch_id = cell_df["patch_id"].iloc[0]

    patch_features = {}

    # ----------------------------------------------------
    # Basic aggregations
    # ----------------------------------------------------
    patch_features["patch_id"] = patch_id
    patch_features["cell_count"] = len(cell_df)

    # ----------------------------------------------------
    # Area statistics
    # ----------------------------------------------------
    patch_features["area_mean"] = cell_df["area"].mean()
    patch_features["area_median"] = cell_df["area"].median()
    patch_features["area_std"] = cell_df["area"].std()

    patch_features["area_skewness"] = skew(cell_df["area"])
    patch_features["area_kurtosis"] = kurtosis(cell_df["area"])

    # ----------------------------------------------------
    # Aspect ratio statistics
    # ----------------------------------------------------
    patch_features["aspect_ratio_mean"] = cell_df["aspect_ratio"].mean()
    patch_features["aspect_ratio_std"] = cell_df["aspect_ratio"].std()
    patch_features["aspect_ratio_skewness"] = skew(cell_df["aspect_ratio"])
    patch_features["aspect_ratio_kurtosis"] = kurtosis(cell_df["aspect_ratio"])

    # ----------------------------------------------------
    # Small / Large nuclei proportion
    # ----------------------------------------------------
    area_median = cell_df["area"].median()

    small_ratio = (cell_df["area"] < area_median).sum() / len(cell_df)
    large_ratio = (cell_df["area"] >= area_median).sum() / len(cell_df)

    patch_features["small_nuclei_ratio"] = small_ratio
    patch_features["large_nuclei_ratio"] = large_ratio

    # ----------------------------------------------------
    # Convert to DataFrame
    # ----------------------------------------------------
    patch_df = pd.DataFrame([patch_features])

    output_path = f"{output_dir}/{patch_id}_patch.csv"
    patch_df.to_csv(output_path, index=False)

    return patch_df
