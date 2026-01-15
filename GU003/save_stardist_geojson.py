import numpy as np
from tifffile import imread
from stardist.models import StarDist2D
import geojson

# -------- Settings --------
image_path = r"C:\Users\wj165\OneDrive\Desktop\QuPath\DHMC_0039_ROI.tif"
output_path = r"C:\Users\wj165\OneDrive\Desktop\QuPath\stardist_output.geojson"

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

coords = details["coord"]   # shape = (N, 2, 32)
print(f"Nuclei detected: {coords.shape[0]}")

# -------- Convert to closed polygons --------
features = []

for i in range(coords.shape[0]):
    y = coords[i, 0, :]     # 32 y-points
    x = coords[i, 1, :]     # 32 x-points

    # Combine into (x, y) pairs
    poly = [(float(x[j]), float(y[j])) for j in range(len(x))]

    # CLOSE THE POLYGON by repeating first point
    poly.append(poly[0])

    features.append(
        geojson.Feature(
            geometry=geojson.Polygon([poly]),
            properties={"id": int(i)}
        )
    )

# -------- Save GeoJSON --------
fc = geojson.FeatureCollection(features)

with open(output_path, "w") as f:
    geojson.dump(fc, f)

print(f"Exported polygons: {len(features)}")
print(f"Saved to: {output_path}")
