import os
import pandas as pd

from io_utils import PatchLoader
from segmentation import CellSegmenter
from features import CellFeatureExtractor
from aggregation import aggregate


class GU003PipelineFromImage:

    def __init__(
        self,
        patch_dir,
        output_dir,
        native_magnification=None,
        target_magnification=20,
        min_area=80
    ):

        self.patch_dir = patch_dir
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        self.loader = PatchLoader(patch_dir)

        self.segmenter = CellSegmenter(
            native_magnification=native_magnification,
            target_magnification=target_magnification,
            min_area=min_area
        )

    def run(self, max_patches=None):

        patches = self.loader.load_all_patches()

        if max_patches:
            patches = patches[:max_patches]

        all_rows = []

        for patch in patches:

            image = patch["image"]
            patch_id = patch["patch_id"]

            labels = self.segmenter.run(image)

            geojson_dir = os.path.join(self.output_dir, "geojson")

            geojson_path = self.segmenter.labels_to_geojson(
                labels,
                patch_id,
                geojson_dir
            )

            extractor = CellFeatureExtractor(
                geojson_path,
                patch_width=image.shape[1],
                patch_height=image.shape[0]
            )

            cell_df = extractor.compute_all_features()

            patch_df = aggregate(cell_df, self.output_dir)

            all_rows.append(patch_df)

        final_df = pd.concat(all_rows, ignore_index=True)

        final_path = os.path.join(self.output_dir, "patch_features.csv")
        final_df.to_csv(final_path, index=False)

        return final_df
