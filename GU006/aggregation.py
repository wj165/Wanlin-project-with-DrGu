
# =========================================================
# GU006 STRICT PATCH-LEVEL UMAP PIPELINE
# Leakage-free version
# =========================================================

import os
import glob
import numpy as np
import pandas as pd

from umap import UMAP

from sklearn.model_selection import train_test_split


class StrictPatchUMAP:

    def __init__(

        self,

        clearcell_dir,

        papillary_dir,

        output_dir,

        n_components=10,

        n_neighbors=15,

        test_size=0.2,

        random_state=42
    ):

        self.clearcell_dir = clearcell_dir

        self.papillary_dir = papillary_dir

        self.output_dir = output_dir

        self.n_components = n_components

        self.n_neighbors = n_neighbors

        self.test_size = test_size

        self.random_state = random_state

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

    # =====================================================
    # load one group
    # =====================================================

    def load_group(

        self,

        folder,

        label,

        class_name
    ):

        csv_files = sorted(
            glob.glob(
                os.path.join(folder, "*.csv")
            )
        )

        dfs = []

        for csv_path in csv_files:

            try:

                df = pd.read_csv(csv_path)

                if len(df) == 0:
                    continue

                df["label"] = label

                df["class_name"] = class_name

                dfs.append(df)

            except Exception as e:

                print(f"ERROR loading {csv_path}")
                print(e)

        return pd.concat(
            dfs,
            ignore_index=True
        )

    # =====================================================
    # load all
    # =====================================================

    def load_all(self):

        clear_df = self.load_group(

            self.clearcell_dir,

            0,

            "Clearcell"
        )

        pap_df = self.load_group(

            self.papillary_dir,

            1,

            "Papillary"
        )

        df = pd.concat(

            [clear_df, pap_df],

            ignore_index=True
        )

        print(
            f"Loaded all cells: {df.shape}"
        )

        return df

    # =====================================================
    # patch split
    # =====================================================

    def split_patches(self, df):

        patch_df = df[[
            "patch_id",
            "label"
        ]].drop_duplicates()

        train_patch_ids, val_patch_ids = train_test_split(

            patch_df["patch_id"],

            test_size=self.test_size,

            random_state=self.random_state,

            stratify=patch_df["label"]
        )

        train_df = df[
            df["patch_id"].isin(train_patch_ids)
        ].copy()

        val_df = df[
            df["patch_id"].isin(val_patch_ids)
        ].copy()

        print(
            f"Train cells: {train_df.shape}"
        )

        print(
            f"Validation cells: {val_df.shape}"
        )

        return train_df, val_df

    # =====================================================
    # fit UMAP on train only
    # =====================================================

    def fit_umap(self, train_df):

        metadata_cols = [

            "patch_id",

            "cell_id",

            "label",

            "class_name"
        ]

        feature_cols = [

            c for c in train_df.columns

            if c not in metadata_cols
        ]

        reducer = UMAP(

            n_components=self.n_components,

            n_neighbors=self.n_neighbors,

            random_state=self.random_state
        )

        train_X = train_df[
            feature_cols
        ].values

        reducer.fit(train_X)

        print("UMAP fit complete")

        return reducer, feature_cols

    # =====================================================
    # transform
    # =====================================================

    def transform_cells(

        self,

        df,

        reducer,

        feature_cols
    ):

        X = df[
            feature_cols
        ].values

        embedding = reducer.transform(X)

        embed_cols = []

        for i in range(
            self.n_components
        ):

            col = f"umap_{i+1}"

            df[col] = embedding[:, i]

            embed_cols.append(col)

        return df, embed_cols

    # =====================================================
    # aggregate patch
    # =====================================================

    def aggregate_patches(

        self,

        df,

        embed_cols
    ):

        grouped = df.groupby("patch_id")

        records = []

        for patch_id, patch_df in grouped:

            record = {}

            record["patch_id"] = patch_id

            record["label"] = patch_df[
                "label"
            ].iloc[0]

            record["class_name"] = patch_df[
                "class_name"
            ].iloc[0]

            record["cell_count"] = len(
                patch_df
            )

            for col in embed_cols:

                vals = patch_df[col]

                record[f"{col}_mean"] = np.mean(vals)

                record[f"{col}_std"] = np.std(vals)

                record[f"{col}_min"] = np.min(vals)

                record[f"{col}_max"] = np.max(vals)

            records.append(record)

        return pd.DataFrame(records)

    # =====================================================
    # run
    # =====================================================

    def run(self):

        df = self.load_all()

        train_df, val_df = self.split_patches(df)

        reducer, feature_cols = self.fit_umap(
            train_df
        )

        train_df, embed_cols = self.transform_cells(

            train_df,

            reducer,

            feature_cols
        )

        val_df, _ = self.transform_cells(

            val_df,

            reducer,

            feature_cols
        )

        train_patch_df = self.aggregate_patches(

            train_df,

            embed_cols
        )

        val_patch_df = self.aggregate_patches(

            val_df,

            embed_cols
        )

        train_out = os.path.join(

            self.output_dir,

            "train_patch_features.csv"
        )

        val_out = os.path.join(

            self.output_dir,

            "val_patch_features.csv"
        )

        train_patch_df.to_csv(
            train_out,
            index=False
        )

        val_patch_df.to_csv(
            val_out,
            index=False
        )

        print("\nDONE")

        print(
            f"Train patch shape: "
            f"{train_patch_df.shape}"
        )

        print(
            f"Validation patch shape: "
            f"{val_patch_df.shape}"
        )

        return train_patch_df, val_patch_df
