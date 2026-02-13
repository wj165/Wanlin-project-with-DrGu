
import os
from PIL import Image
import numpy as np


class PatchLoader:
    """
    Clean I/O layer.

    Responsibilities:
    - Validate patch directory
    - Load PNG patches
    - Parse patient_id and patch_id
    - Return structured patch dictionary

    NO resizing
    NO magnification logic
    NO model inference
    """

    def __init__(self, patch_dir):
        self.patch_dir = patch_dir

        if not os.path.exists(patch_dir):
            raise ValueError(f"Patch directory not found: {patch_dir}")

    # --------------------------------------------------
    # Parse filename
    # Example:
    # DHMC_0011_roi__x_2048_y_3328.png
    # --------------------------------------------------
    def parse_ids(self, filepath):

        base = os.path.basename(filepath)
        name = base.replace(".png", "")

        parts = name.split("_")

        # DHMC_0011
        patient_id = parts[0] + "_" + parts[1]
        patch_id = name

        return patient_id, patch_id

    # --------------------------------------------------
    # Load single patch
    # --------------------------------------------------
    def load_patch(self, filepath):

        image = Image.open(filepath).convert("RGB")
        image = np.array(image)

        patient_id, patch_id = self.parse_ids(filepath)

        return {
            "image": image,
            "patient_id": patient_id,
            "patch_id": patch_id
        }

    # --------------------------------------------------
    # Load all patches in directory
    # --------------------------------------------------
    def load_all_patches(self):

        patch_list = []

        for fname in sorted(os.listdir(self.patch_dir)):
            if fname.lower().endswith(".png"):
                full_path = os.path.join(self.patch_dir, fname)
                patch = self.load_patch(full_path)
                patch_list.append(patch)

        if len(patch_list) == 0:
            raise ValueError("No PNG patches found in directory.")

        return patch_list
