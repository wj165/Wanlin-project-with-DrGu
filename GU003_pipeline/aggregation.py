import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis


def aggregate(cell_df, output_dir):

    patch_id = cell_df["patch_id"].iloc[0]

    patch_features = {}

    # ------------------------------
    # Basic info
    # ------------------------------
    patch_features["patch_id"] = patch_id
    patch_features["cell_count"] = len(cell_df)

    # ------------------------------
    # Small / Large nuclei ratio
    # ------------------------------
    area_median = cell_df["area"].median()

    patch_features["small_nuclei_ratio"] = (
        cell_df["area"] < area_median
    ).mean()

    patch_features["large_nuclei_ratio"] = (
        cell_df["area"] >= area_median
    ).mean()

    # ------------------------------
    # Auto aggregate ALL cell-level features
    # ------------------------------
    feature_cols = [
        col for col in cell_df.columns
        if col not in ["patch_id", "cell_id"]
    ]

    for col in feature_cols:
        values = cell_df[col].values

        patch_features[f"{col}_mean"] = np.mean(values)
        patch_features[f"{col}_std"] = np.std(values)
        patch_features[f"{col}_skewness"] = skew(values)
        patch_features[f"{col}_kurtosis"] = kurtosis(values)

    # ------------------------------
    # Convert to DataFrame
    # ------------------------------
    patch_df = pd.DataFrame([patch_features])

    output_path = f"{output_dir}/{patch_id}_patch.csv"
    patch_df.to_csv(output_path, index=False)

    return patch_df
