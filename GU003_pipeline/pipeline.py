import os
import pandas as pd

from io_utils import PatchLoader
from segmentation import CellSegmenter
from features import CellFeatureExtractor
from aggregation import PatchAggregator


class GU003Pipeline:

    def __init__(self, patch_dir, output_dir):
        self.patch_dir = patch_dir
        self.output_dir = output_dir

        self.loader = PatchLoader(patch_dir)
        self.segmenter = CellSegmenter()
        self.extractor = CellFeatureExtractor()
        self.aggregator = PatchAggregator()

    def run(self):

        patches = self.loader.load_all_patches()
        all_patch_rows = []

        for patch in patches:

            image = patch["image"]
            patch_id = patch["patch_id"]
            patient_id = patch["patient_id"]

            labels = self.segmenter.run(image)

            cell_df = self.extractor.compute_from_labels(
                labels,
                patch_id,
                patient_id
            )

            patch_row = self.aggregator.aggregate(cell_df)
            all_patch_rows.append(patch_row)

        final_df = pd.DataFrame(all_patch_rows)

        os.makedirs(self.output_dir, exist_ok=True)
        final_df.to_csv(
            os.path.join(self.output_dir, "patch_features.csv"),
            index=False
        )

        return final_df
