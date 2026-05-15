
# =========================================================
# GU006 STRICT CASE-LEVEL UMAP PIPELINE
# LEAKAGE-FREE PATHOLOGY AI PIPELINE
# =========================================================

import os
import glob
import numpy as np
import pandas as pd

from umap import UMAP
from sklearn.model_selection import train_test_split


class StrictCaseUMAP:

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
    # load one class
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

                print(f"ERROR loading: {csv_path}")
                print(e)

        return pd.concat(
            dfs,
            ignore_index=True
        )

    # =====================================================
    # load all cells
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
    # STRICT CASE-LEVEL SPLIT
    # =====================================================

    def split_cases(self, df):

        df["case_id"] = (
            df["patch_id"]
            .str.split("_patch_")
            .str[0]
        )

        case_df = df[
            ["case_id", "label"]
        ].drop_duplicates()

        print(
            f"Total unique cases: {len(case_df)}"
        )

        train_cases, val_cases = train_test_split(

            case_df["case_id"],

            test_size=self.test_size,

            random_state=self.random_state,

            stratify=case_df["label"]
        )

        train_df = df[
            df["case_id"].isin(train_cases)
        ].copy()

        val_df = df[
            df["case_id"].isin(val_cases)
        ].copy()

        overlap = set(train_cases).intersection(
            set(val_cases)
        )

        print(
            f"Overlap cases: {len(overlap)}"
        )

        print(
            f"Train cases: {len(train_cases)}"
        )

        print(
            f"Validation cases: {len(val_cases)}"
        )

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

            "class_name",

            "case_id"
        ]

        feature_cols = [

            c for c in train_df.columns

            if c not in metadata_cols
        ]

        train_X = train_df[
            feature_cols
        ].values

        reducer = UMAP(

            n_components=self.n_components,

            n_neighbors=self.n_neighbors,

            random_state=self.random_state
        )

        reducer.fit(train_X)

        print(
            "UMAP fit complete"
        )

        return reducer, feature_cols

    # =====================================================
    # transform cells
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
    # aggregate patches
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

            record["case_id"] = patch_df[
                "case_id"
            ].iloc[0]

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
    # RUN FULL PIPELINE
    # =====================================================

    def run(self):

        df = self.load_all()

        train_df, val_df = self.split_cases(df)

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
