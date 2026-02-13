
import os
import pandas as pd

from io_utils import PatchLoader
from segmentation import CellSegmenter
from features import CellFeatureExtractor
from aggregation import aggregate


class GU003Pipeline:
    """
    Clean patch-first pipeline:

    PatchLoader
        ↓
    CellSegmenter (auto magnification scaling)
        ↓
    GeoJSON export
        ↓
    CellFeatureExtractor
        ↓
    Patch-level aggregation
        ↓
    Final CSV
    """

    def __init__(
        self,
        patch_dir,
        output_dir,
        native_magnification=None,   # e.g. 5 for 5x data
        target_magnification=20,
        min_area=80
    ):

        self.patch_dir = patch_dir
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        # --------------------------
        # Modules
        # --------------------------

        self.loader = PatchLoader(patch_dir)

        self.segmenter = CellSegmenter(
            native_magnification=native_magnification,
            target_magnification=target_magnification,
            min_area=min_area
        )

        self.extractor = CellFeatureExtractor()

    # --------------------------------------------------
    # Run full pipeline
    # --------------------------------------------------
    def run(self, max_patches=None):
        """
        max_patches: int or None
        Use for quick debugging (e.g. max_patches=2)
        """

        patches = self.loader.load_all_patches()

        if max_patches is not None:
            patches = patches[:max_patches]

        all_patch_rows = []

        for patch in patches:

            image = patch["image"]
            patch_id = patch["patch_id"]
            patient_id = patch["patient_id"]

            print(f"Processing {patch_id} ...")

            # ------------------------------------------
            # 1. Segmentation
            # ------------------------------------------
            labels = self.segmenter.run(image)

            # ------------------------------------------
            # 2. Save GeoJSON
            # ------------------------------------------
            geojson_dir = os.path.join(self.output_dir, "geojson")
            geojson_path = self.segmenter.labels_to_geojson(
                labels,
                patch_id,
                geojson_dir
            )

            # ------------------------------------------
            # 3. Cell-level features
            # ------------------------------------------
            cell_df = self.extractor.compute_from_labels(
                labels,
                patch_id,
                patient_id
            )

            # ------------------------------------------
            # 4. Patch-level aggregation
            # ------------------------------------------
            patch_df = aggregate(cell_df, self.output_dir)

            all_patch_rows.append(patch_df)

        # ------------------------------------------
        # Final merge
        # ------------------------------------------
        final_df = pd.concat(all_patch_rows, ignore_index=True)

        final_csv_path = os.path.join(self.output_dir, "patch_features.csv")
        final_df.to_csv(final_csv_path, index=False)

        print(f"\nFinal CSV saved to: {final_csv_path}")

        return final_df
