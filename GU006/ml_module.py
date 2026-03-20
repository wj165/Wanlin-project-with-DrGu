
import os
import shutil
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from autogluon.tabular import TabularPredictor


class GU003AutoML:

    def __init__(self, data_path, output_dir, model_name="autogluon_model"):

        self.data_path = data_path
        self.output_dir = output_dir
        self.model_dir = os.path.join(output_dir, model_name)

        os.makedirs(self.output_dir, exist_ok=True)

    # --------------------------------------------------
    # Load + keep ONLY 69 morphology features
    # --------------------------------------------------
    def load_and_prepare(self):

        df = pd.read_csv(self.data_path)

        # case-level split key
        df["case_id"] = df["patch_id"].apply(lambda x: x.split("_roi")[0])

        # keep only 69 features (Dr.Gu rule)
        feature_cols = [
            c for c in df.columns
            if c.endswith("_mean")
            or c.endswith("_std")
            or c.endswith("_skewness")
        ]

        df_ml = df[feature_cols + ["label", "case_id"]]

        return df_ml

    # --------------------------------------------------
    # Case-level split
    # --------------------------------------------------
    def group_split(self, df_ml, test_size=0.2, random_state=42):

        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state
        )

        train_idx, test_idx = next(
            gss.split(df_ml, df_ml["label"], groups=df_ml["case_id"])
        )

        train_df = df_ml.iloc[train_idx].drop(columns=["case_id"])
        test_df  = df_ml.iloc[test_idx].drop(columns=["case_id"])

        return train_df, test_df

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    def train(self, train_df):

        if os.path.exists(self.model_dir):
            shutil.rmtree(self.model_dir)

        predictor = TabularPredictor(
            label="label",
            eval_metric="roc_auc",
            path=self.model_dir
        ).fit(
            train_df,
            presets="best_quality",
            num_bag_folds=5
        )

        return predictor

    # --------------------------------------------------
    # Evaluate + save
    # --------------------------------------------------
    def evaluate_and_save(self, predictor, test_df):

        results = predictor.evaluate(test_df)
        leaderboard = predictor.leaderboard(test_df, silent=True)
        importance = predictor.feature_importance(test_df)

        pd.DataFrame([results]).to_csv(
            os.path.join(self.output_dir, "performance.csv"),
            index=False
        )

        leaderboard.to_csv(
            os.path.join(self.output_dir, "leaderboard.csv"),
            index=False
        )

        importance.to_csv(
            os.path.join(self.output_dir, "feature_importance.csv")
        )

        return results

    # --------------------------------------------------
    # Full run
    # --------------------------------------------------
    def run(self):

        df_ml = self.load_and_prepare()
        train_df, test_df = self.group_split(df_ml)
        predictor = self.train(train_df)
        results = self.evaluate_and_save(predictor, test_df)

        return results

