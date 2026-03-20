
import json
import numpy as np
import pandas as pd
from shapely.geometry import shape, Point

class CellFeatureExtractor:

    def __init__(self, geojson_path, patch_width, patch_height):
        self.geojson_path = geojson_path
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.patch_center = Point(patch_width / 2, patch_height / 2)

        with open(geojson_path) as f:
            self.data = json.load(f)

    def compute_all_features(self):

        records = []

        for i, feature in enumerate(self.data["features"]):

            try:
                poly = shape(feature["geometry"])
                if not poly.is_valid or poly.area == 0:
                    continue
                if len(poly.exterior.coords) < 4:
                    continue
            except:
                continue

            props = feature.get("properties", {})
            record = {}

            record["patch_id"] = props.get("patch_id", "unknown")
            record["cell_id"] = i

            area = poly.area
            perimeter = poly.length

            record["area"] = area
            record["perimeter"] = perimeter
            record["equivalent_diameter"] = np.sqrt(4 * area / np.pi)

            minx, miny, maxx, maxy = poly.bounds
            width = maxx - minx
            height = maxy - miny

            major_axis = max(width, height)
            minor_axis = min(width, height)

            record["major_axis_length"] = major_axis
            record["minor_axis_length"] = minor_axis
            record["aspect_ratio"] = major_axis / (minor_axis + 1e-8)
            record["elongation"] = 1 - (minor_axis / (major_axis + 1e-8))

            record["circularity"] = (4 * np.pi * area) / (perimeter**2 + 1e-8)
            record["compactness"] = perimeter**2 / (area + 1e-8)
            record["solidity"] = area / (poly.convex_hull.area + 1e-8)
            record["extent"] = area / ((width * height) + 1e-8)
            record["eccentricity"] = np.sqrt(
                1 - (minor_axis**2 / (major_axis**2 + 1e-8))
            )

            # Stable SVD orientation
            coords = np.array(poly.exterior.coords)[:-1]
            coords_centered = coords - coords.mean(axis=0)
            try:
                _, _, vh = np.linalg.svd(coords_centered)
                principal_axis = vh[0]
                orientation = np.arctan2(principal_axis[1], principal_axis[0])
            except:
                orientation = 0

            record["orientation"] = orientation

            centroid = poly.centroid
            record["centroid_x"] = centroid.x
            record["centroid_y"] = centroid.y
            record["centroid_x_norm"] = centroid.x / self.patch_width
            record["centroid_y_norm"] = centroid.y / self.patch_height

            dist = centroid.distance(self.patch_center)
            max_dist = np.sqrt((self.patch_width/2)**2 + (self.patch_height/2)**2)
            record["distance_to_center_norm"] = dist / max_dist

            record["perimeter_area_ratio"] = perimeter / (area + 1e-8)

            hull = poly.convex_hull
            record["convex_hull_area"] = hull.area
            record["convex_hull_perimeter"] = hull.length
            record["convexity_ratio"] = hull.length / (perimeter + 1e-8)
            record["boundary_roughness"] = perimeter / (hull.length + 1e-8)

            records.append(record)

        return pd.DataFrame(records)
