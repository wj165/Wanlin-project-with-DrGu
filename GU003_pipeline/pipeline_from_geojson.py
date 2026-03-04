import os
import pandas as pd

from features import CellFeatureExtractor
from aggregation import aggregate


class GU003PipelineFromGeoJSON:

    def __init__(self, geojson_dir, output_dir, patch_width, patch_height):

        self.geojson_dir = geojson_dir
        self.output_dir = output_dir
        self.patch_width = patch_width
        self.patch_height = patch_height

        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):

        files = sorted([
            f for f in os.listdir(self.geojson_dir)
            if f.endswith(".geojson")
        ])

        if len(files) == 0:
            raise ValueError("No geojson files found")

        all_rows = []

        for fname in files:

            path = os.path.join(self.geojson_dir, fname)

            extractor = CellFeatureExtractor(
                path,
                patch_width=self.patch_width,
                patch_height=self.patch_height
            )

            cell_df = extractor.compute_all_features()

            patch_df = aggregate(cell_df, self.output_dir)

            all_rows.append(patch_df)

        final_df = pd.concat(all_rows, ignore_index=True)

        final_path = os.path.join(self.output_dir, "patch_features.csv")
        final_df.to_csv(final_path, index=False)

        return final_df
