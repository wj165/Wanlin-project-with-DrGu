import os
from PIL import Image
import numpy as np


class PatchLoader:

    def __init__(self, patch_dir):
        self.patch_dir = patch_dir

        if not os.path.exists(patch_dir):
            raise ValueError(f"Patch directory not found: {patch_dir}")

    # --------------------------------------------------
    # Parse filename
    # Example:
    # DHMC_0195_roi_x_2304_y_6912.png
    # --------------------------------------------------
    def parse_ids(self, filename):

        base = os.path.basename(filename)
        name = base.replace(".png", "")

        parts = name.split("_")

        patient_id = parts[0] + "_" + parts[1]
        patch_id = name

        return patient_id, patch_id

    # --------------------------------------------------
    # Load single patch
    # --------------------------------------------------
    def load_patch(self, filepath):

        img = Image.open(filepath)
        img_np = np.array(img)

        patient_id, patch_id = self.parse_ids(filepath)

        pixel_size = self.get_pixel_size(img)

        return {
            "image": img_np,
            "patient_id": patient_id,
            "patch_id": patch_id,
            "pixel_size": pixel_size,
            "width": img_np.shape[1],
            "height": img_np.shape[0]
        }

    # --------------------------------------------------
    # Try reading pixel size from metadata
    # --------------------------------------------------
    def get_pixel_size(self, pil_image):

        info = pil_image.info

        if "dpi" in info:
            dpi = info["dpi"][0]
            micron_per_pixel = 25400 / dpi
            return micron_per_pixel

        # PNG usually has no pixel metadata
        return None

    # --------------------------------------------------
    # Load all PNGs in folder
    # --------------------------------------------------
    def load_all_patches(self):

        patches = []

        for file in os.listdir(self.patch_dir):
            if file.endswith(".png"):
                full_path = os.path.join(self.patch_dir, file)
                patches.append(self.load_patch(full_path))

        return patches
