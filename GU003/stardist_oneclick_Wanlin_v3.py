import json
import numpy as np
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from skimage import io
from shapely.geometry import Polygon

# --------------------------------------------------
# Load image
# --------------------------------------------------
img = io.imread("DHMC_0039_ROI.tif")
img_norm = normalize(img, 1, 99.8)

# --------------------------------------------------
# Run StarDist
# --------------------------------------------------
model = StarDist2D.from_pretrained("2D_versatile_he")
labels, details = model.predict_instances(img_norm)

coords_list = details["coord"]
print("Raw nucleus count:", len(coords_list))

# --------------------------------------------------
# Convert polygons (QuPath-compatible)
# --------------------------------------------------
features = []
valid_count = 0

for pts in coords_list:
    pts = np.array(pts)

    # skip invalid polygons
    if pts.shape[0] < 3:
        continue

    # ---- FORCE CLOSE THE POLYGON ----
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    # ---- Construct polygon manually, no shapely ----
    # QuPath requires: coordinates: [[[x,y], ...]]
    poly_coords = pts.tolist()

    # validate polygon has area
    if len(poly_coords) < 4:
        continue

    features.append({
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [poly_coords]   # must be one extra bracket
        }
    })

    valid_count += 1

# --------------------------------------------------
# Save GeoJSON
# --------------------------------------------------
geojson = {
    "type": "FeatureCollection",
    "features": features
}

with open("stardist_output_final.geojson", "w") as f:
    json.dump(geojson, f)

print("Valid polygons saved:", valid_count)
print("Saved to stardist_output_final.geojson")
