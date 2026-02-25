import json
import numpy as np
import pandas as pd
from shapely.geometry import shape
from shapely.geometry import Point
from shapely.ops import unary_union


class CellFeatureExtractor:

    def __init__(self, geojson_path, patch_width, patch_height):
        self.geojson_path = geojson_path
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.patch_center = Point(patch_width / 2, patch_height / 2)

        with open(geojson_path) as f:
            self.data = json.load(f)

    # ----------------------------------------------------
    # Main
    # ----------------------------------------------------
    
    
    def compute_all_features(self):

        records = []

        for i, feature in enumerate(self.data["features"]):

            try:
                geom = shape(feature["geometry"])
            except:
                continue

            # ---- Handle MultiPolygon ----
            if geom.geom_type == "MultiPolygon":
                if len(geom.geoms) == 0:
                    continue
                poly = max(geom.geoms, key=lambda p: p.area)
            elif geom.geom_type == "Polygon":
                poly = geom
            else:
                continue

            if poly.is_empty or not poly.is_valid:
                continue

            coords = list(poly.exterior.coords)
            if len(coords) < 4:
                continue

            props = feature.get("properties", {})
            record = {}

            record["patch_id"] = props.get("patch_id", "unknown")
            record["cell_id"] = i

            area = poly.area
            perimeter = poly.length

            record["area"] = area
            record["perimeter"] = perimeter
            record["equivalent_diameter"] = (4 * area / 3.1415926) ** 0.5

            # ---- PCA orientation ----
            coords_np = np.array(coords)
            coords_centered = coords_np - coords_np.mean(axis=0)
            cov = np.cov(coords_centered.T)

            try:
                eigvals, eigvecs = np.linalg.eig(cov)
                largest_index = np.argmax(eigvals)
                principal_vector = eigvecs[:, largest_index]
                orientation = np.arctan2(principal_vector[1], principal_vector[0])
            except:
                orientation = 0

            record["orientation"] = orientation

            records.append(record)

        return pd.DataFrame(records)
def save_csv(self, df, output_path):
        df.to_csv(output_path, index=False)
