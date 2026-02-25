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
                poly = shape(feature["geometry"])
            except:
                continue

            if poly.is_empty or not poly.is_valid:
                continue

            if not hasattr(poly, "exterior"):
                continue

            if len(poly.exterior.coords) < 4:
                continue

            try:
                poly = shape(feature["geometry"])
                if not poly.is_valid or poly.is_empty:
                    continue
                if len(poly.exterior.coords) < 4:
                    continue
            except:
                continue


            poly = shape(feature["geometry"])
            props = feature["properties"]

            record = {}

            record["patch_id"] = props["patch_id"]
            record["cell_id"] = i

            # ==============================
            # Basic geometry
            # ==============================
            area = poly.area
            perimeter = poly.length

            record["area"] = area
            record["perimeter"] = perimeter
            record["equivalent_diameter"] = np.sqrt(4 * area / np.pi)

            # ==============================
            # Axis / ratio
            # ==============================
            minx, miny, maxx, maxy = poly.bounds
            major_axis = max(maxx - minx, maxy - miny)
            minor_axis = min(maxx - minx, maxy - miny)

            record["major_axis_length"] = major_axis
            record["minor_axis_length"] = minor_axis

            aspect_ratio = major_axis / (minor_axis + 1e-8)
            record["aspect_ratio"] = aspect_ratio
            record["elongation"] = 1 - (minor_axis / (major_axis + 1e-8))

            # ==============================
            # Shape descriptors
            # ==============================
            circularity = (4 * np.pi * area) / (perimeter**2 + 1e-8)
            record["circularity"] = circularity

            record["compactness"] = perimeter**2 / (area + 1e-8)
            record["solidity"] = area / (poly.convex_hull.area + 1e-8)
            record["extent"] = area / ((major_axis * minor_axis) + 1e-8)
            record["eccentricity"] = np.sqrt(
                1 - (minor_axis**2 / (major_axis**2 + 1e-8))
            )

            # ==============================
            # PCA-based orientation
            # ==============================
            coords = np.array(poly.exterior.coords)
            coords_centered = coords - coords.mean(axis=0)
            cov = np.cov(coords_centered.T)
            eigvals, eigvecs = np.linalg.eig(cov)
            largest_index = np.argmax(eigvals)
            principal_vector = eigvecs[:, largest_index]
            orientation = np.arctan2(principal_vector[1], principal_vector[0])
            record["orientation"] = orientation

            # ==============================
            # Spatial features
            # ==============================
            centroid = poly.centroid

            record["centroid_x"] = centroid.x
            record["centroid_y"] = centroid.y

            record["centroid_x_norm"] = centroid.x / self.patch_width
            record["centroid_y_norm"] = centroid.y / self.patch_height

            dist = centroid.distance(self.patch_center)
            max_dist = np.sqrt(
                (self.patch_width / 2)**2 +
                (self.patch_height / 2)**2
            )

            record["distance_to_center_norm"] = dist / max_dist

            # ==============================
            # Boundary / complexity
            # ==============================
            record["perimeter_area_ratio"] = perimeter / (area + 1e-8)

            hull = poly.convex_hull
            record["convex_hull_area"] = hull.area
            record["convex_hull_perimeter"] = hull.length
            record["convexity_ratio"] = hull.length / (perimeter + 1e-8)
            record["boundary_roughness"] = perimeter / (hull.length + 1e-8)

            records.append(record)

        return pd.DataFrame(records)

    # ----------------------------------------------------
    # Save
    # ----------------------------------------------------
    def save_csv(self, df, output_path):
        df.to_csv(output_path, index=False)
