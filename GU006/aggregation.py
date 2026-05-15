
import os
import glob
import numpy as np
import pandas as pd

from umap import UMAP


class PatchFeatureAggregator:

    def __init__(
        self,
        cell_feature_dir,
        output_csv,
        n_components=10,
        random_state=42
    ):

        self.cell_feature_dir = cell_feature_dir

        self.output_csv = output_csv

        self.n_components = n_components

        self.random_state = random_state

    # =====================================================
    # load all cell feature csv
    # =====================================================

    def load_all_cells(self):

        csv_files = sorted(
            glob.glob(
                os.path.join(
                    self.cell_feature_dir,
                    "*.csv"
                )
            )
        )

        dfs = []

        for csv_path in csv_files:

            try:

                df = pd.read_csv(csv_path)

                dfs.append(df)

            except Exception as e:

                print(f"ERROR loading: {csv_path}")
                print(e)

        if len(dfs) == 0:

            raise ValueError(
                "No cell feature CSV found"
            )

        all_df = pd.concat(
            dfs,
            ignore_index=True
        )

        return all_df

    # =====================================================
    # UMAP aggregation
    # =====================================================

    def aggregate(self):

        df = self.load_all_cells()

        print(
            f"Loaded cells: {df.shape}"
        )

        # =================================================
        # metadata columns
        # =================================================

        metadata_cols = [
            "patch_id",
            "cell_id"
        ]

        # =================================================
        # feature columns
        # =================================================

        feature_cols = [
            c for c in df.columns
            if c not in metadata_cols
        ]

        print(
            f"Feature count: {len(feature_cols)}"
        )

        # =================================================
        # group by patch
        # =================================================

        grouped = df.groupby(
            "patch_id"
        )

        patch_records = []

        for patch_id, patch_df in grouped:

            try:

                X = patch_df[
                    feature_cols
                ].values

                # -----------------------------------------
                # skip tiny patches
                # -----------------------------------------

                if X.shape[0] < 2:
                    continue

                # -----------------------------------------
                # UMAP
                # -----------------------------------------

                reducer = UMAP(
                    n_components=self.n_components,
                    random_state=self.random_state
                )

                embedding = reducer.fit_transform(
                    X
                )

                # -----------------------------------------
                # aggregation
                # -----------------------------------------

                record = {}

                record["patch_id"] = patch_id

                record["cell_count"] = (
                    X.shape[0]
                )

                for dim in range(
                    self.n_components
                ):

                    vals = embedding[:, dim]

                    record[
                        f"umap_{dim+1}_mean"
                    ] = np.mean(vals)

                    record[
                        f"umap_{dim+1}_std"
                    ] = np.std(vals)

                    record[
                        f"umap_{dim+1}_min"
                    ] = np.min(vals)

                    record[
                        f"umap_{dim+1}_max"
                    ] = np.max(vals)

                patch_records.append(
                    record
                )

            except Exception as e:

                print(
                    f"ERROR patch: {patch_id}"
                )

                print(e)

        # =================================================
        # final dataframe
        # =================================================

        patch_df = pd.DataFrame(
            patch_records
        )

        print(
            f"Patch features: {patch_df.shape}"
        )

        # =================================================
        # save
        # =================================================

        patch_df.to_csv(
            self.output_csv,
            index=False
        )

        print(
            f"Saved: {self.output_csv}"
        )

        return patch_df
