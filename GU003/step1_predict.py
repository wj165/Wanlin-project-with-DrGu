import numpy as np
from tifffile import imread
from stardist.models import StarDist2D
import json

# -------- Settings --------
image_path = r"C:\Users\wj165\OneDrive\Desktop\QuPath\DHMC_0039_ROI.tif"
output_json = r"C:\Users\wj165\OneDrive\Desktop\QuPath\stardist_raw.json"

# -------- Load image --------
img = imread(image_path).astype(np.float32) / 255.0

# -------- Load model --------
model = StarDist2D.from_pretrained('2D_versatile_he')

# -------- Predict nuclei --------
labels, details = model.predict_instances(
    img,
    prob_thresh=0.55,
    nms_thresh=0.3
)

coords = details["coord"]     # shape (N, 2, 32)

print("Nuclei detected:", coords.shape[0])

# -------- Save raw polygons --------
raw = coords.tolist()

with open(output_json, "w") as f:
    json.dump({"coord": raw}, f)

print("Saved raw polygons to:", output_json)
