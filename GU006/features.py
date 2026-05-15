
import json
import numpy as np
import pandas as pd
import cv2

from shapely.geometry import shape, Point
from scipy import ndimage
from skimage.feature import local_binary_pattern


class CellFeatureExtractor:

    def __init__(
        self,
        geojson_path,
        image_array,
        patch_width,
        patch_height
    ):

        self.geojson_path = geojson_path
        self.image = image_array

        self.patch_width = patch_width
        self.patch_height = patch_height

        self.patch_center = Point(
            patch_width / 2,
            patch_height / 2
        )

        with open(geojson_path) as f:
            self.data = json.load(f)

    # =====================================================
    # polygon -> mask
    # =====================================================

    def polygon_to_mask(self, poly):

        mask = np.zeros(
            (self.patch_height, self.patch_width),
            dtype=np.uint8
        )

        coords = np.array(
            poly.exterior.coords
        ).astype(np.int32)

        cv2.fillPoly(mask, [coords], 1)

        return mask

    # =====================================================
    # radius features
    # =====================================================

    def compute_radius_features(self, poly):

        centroid = np.array([
            poly.centroid.x,
            poly.centroid.y
        ])

        boundary = np.array(
            poly.exterior.coords
        )

        dists = np.linalg.norm(
            boundary - centroid,
            axis=1
        )

        return dists.max(), dists.min()

    # =====================================================
    # caliper features
    # =====================================================

    def compute_caliper_features(self, poly):

        coords = np.array(
            poly.exterior.coords
        ).astype(np.float32)

        rect = cv2.minAreaRect(coords)

        w, h = rect[1]

        return max(w, h), min(w, h)

    # =====================================================
    # ellipse similarity
    # =====================================================

    def compute_ellipse_similarity(self, poly):

        coords = np.array(
            poly.exterior.coords
        ).astype(np.float32)

        if len(coords) < 5:
            return 0

        try:

            ellipse = cv2.fitEllipse(coords)

            (_, _), (MA, ma), _ = ellipse

            ellipse_area = (
                np.pi * (MA/2) * (ma/2)
            )

            similarity = abs(
                poly.area - ellipse_area
            ) / (ellipse_area + 1e-8)

            return similarity

        except:
            return 0

    # =====================================================
    # circle similarity
    # =====================================================

    def compute_circle_similarity(self, poly):

        area = poly.area

        radius = np.sqrt(area / np.pi)

        ideal_circle_area = (
            np.pi * radius**2
        )

        similarity = abs(
            area - ideal_circle_area
        ) / (ideal_circle_area + 1e-8)

        return similarity

    # =====================================================
    # blob features
    # =====================================================

    def compute_blob_feature(
        self,
        img,
        bright=True,
        small=True
    ):

        if img.size == 0:
            return 0

        if bright:

            thresh = (
                img.mean() + img.std()
            )

            binary = img > thresh

        else:

            thresh = (
                img.mean() - img.std()
            )

            binary = img < thresh

        labeled, num = ndimage.label(binary)

        total = 0

        for region_id in range(1, num + 1):

            size = np.sum(
                labeled == region_id
            )

            if small:

                if size <= 8:
                    total += size

            else:

                if size >= 32:
                    total += size

        return total

    # =====================================================
    # LBP
    # =====================================================

    def compute_lbp_histogram(
        self,
        img,
        bins=10
    ):

        if img.size == 0:

            return np.zeros(bins)

        lbp = local_binary_pattern(
            img,
            P=8,
            R=1,
            method="uniform"
        )

        hist, _ = np.histogram(
            lbp.ravel(),
            bins=bins,
            range=(0, bins),
            density=True
        )

        return hist

    # =====================================================
    # main
    # =====================================================

    def compute_all_features(self):

        records = []

        gray = self.image.copy()

        for i, feature in enumerate(
            self.data["features"]
        ):

            try:

                poly = shape(
                    feature["geometry"]
                )

                if not poly.is_valid:
                    continue

                if poly.area == 0:
                    continue

            except:
                continue

            props = feature.get(
                "properties",
                {}
            )

            record = {}

            # =================================================
            # metadata (2)
            # =================================================

            record["patch_id"] = props.get(
                "patch_id",
                "unknown"
            )

            record["cell_id"] = i

            # =================================================
            # geometry base
            # =================================================

            area = poly.area
            perimeter = poly.length

            minx, miny, maxx, maxy = poly.bounds

            width = maxx - minx
            height = maxy - miny

            major_axis = max(width, height)
            minor_axis = min(width, height)

            centroid = poly.centroid

            hull = poly.convex_hull

            # =================================================
            # original 23
            # =================================================

            record["area"] = area

            record["perimeter"] = perimeter

            record["equivalent_diameter"] = (
                np.sqrt(4 * area / np.pi)
            )

            record["major_axis_length"] = (
                major_axis
            )

            record["minor_axis_length"] = (
                minor_axis
            )

            record["aspect_ratio"] = (
                major_axis /
                (minor_axis + 1e-8)
            )

            record["elongation"] = (
                1 - (
                    minor_axis /
                    (major_axis + 1e-8)
                )
            )

            record["circularity"] = (
                4 * np.pi * area /
                (perimeter**2 + 1e-8)
            )

            record["compactness"] = (
                perimeter**2 /
                (area + 1e-8)
            )

            record["solidity"] = (
                area /
                (hull.area + 1e-8)
            )

            record["extent"] = (
                area /
                ((width * height) + 1e-8)
            )

            record["eccentricity"] = np.sqrt(
                1 - (
                    minor_axis**2 /
                    (major_axis**2 + 1e-8)
                )
            )

            coords = np.array(
                poly.exterior.coords
            )[:-1]

            centered = (
                coords -
                coords.mean(axis=0)
            )

            try:

                _, _, vh = np.linalg.svd(
                    centered
                )

                principal_axis = vh[0]

                orientation = np.arctan2(
                    principal_axis[1],
                    principal_axis[0]
                )

            except:
                orientation = 0

            record["orientation"] = orientation

            record["centroid_x"] = centroid.x
            record["centroid_y"] = centroid.y

            record["centroid_x_norm"] = (
                centroid.x /
                self.patch_width
            )

            record["centroid_y_norm"] = (
                centroid.y /
                self.patch_height
            )

            dist = centroid.distance(
                self.patch_center
            )

            max_dist = np.sqrt(
                (self.patch_width/2)**2 +
                (self.patch_height/2)**2
            )

            record[
                "distance_to_center_norm"
            ] = dist / max_dist

            record["perimeter_area_ratio"] = (
                perimeter /
                (area + 1e-8)
            )

            record["convex_hull_area"] = (
                hull.area
            )

            record["convex_hull_perimeter"] = (
                hull.length
            )

            record["convexity_ratio"] = (
                hull.length /
                (perimeter + 1e-8)
            )

            record["boundary_roughness"] = (
                perimeter /
                (hull.length + 1e-8)
            )

            # =================================================
            # mask
            # =================================================

            mask = self.polygon_to_mask(poly)

            coords_mask = np.where(mask > 0)

            if len(coords_mask[0]) == 0:
                continue

            pixel_values = gray[coords_mask]

            # =================================================
            # boundary clipping
            # =================================================

            minx = max(0, int(minx))
            miny = max(0, int(miny))

            maxx = min(
                self.patch_width,
                int(maxx)
            )

            maxy = min(
                self.patch_height,
                int(maxy)
            )

            crop = gray[
                miny:maxy,
                minx:maxx
            ]

            if crop.size == 0:
                continue

            # =================================================
            # intensity (6)
            # =================================================

            record["mean_intensity"] = (
                np.mean(pixel_values)
            )

            record["std_intensity"] = (
                np.std(pixel_values)
            )

            record["percentile_25"] = (
                np.percentile(
                    pixel_values,
                    25
                )
            )

            record["percentile_75"] = (
                np.percentile(
                    pixel_values,
                    75
                )
            )

            thresh_high = (
                pixel_values.mean() +
                pixel_values.std()
            )

            thresh_low = (
                pixel_values.mean() -
                pixel_values.std()
            )

            record["positive_fraction"] = (
                np.mean(
                    pixel_values > thresh_high
                )
            )

            record["negative_fraction"] = (
                np.mean(
                    pixel_values < thresh_low
                )
            )

            # =================================================
            # blob/granule (4)
            # =================================================

            record["small_bright_pixels"] = (
                self.compute_blob_feature(
                    crop,
                    bright=True,
                    small=True
                )
            )

            record["small_dark_pixels"] = (
                self.compute_blob_feature(
                    crop,
                    bright=False,
                    small=True
                )
            )

            record["large_bright_pixels"] = (
                self.compute_blob_feature(
                    crop,
                    bright=True,
                    small=False
                )
            )

            record["large_dark_pixels"] = (
                self.compute_blob_feature(
                    crop,
                    bright=False,
                    small=False
                )
            )

            # =================================================
            # moments (2)
            # =================================================

            moments = cv2.moments(mask)

            record["moment_1"] = (
                moments["mu20"]
            )

            record["moment_2"] = (
                moments["mu02"]
            )

            # =================================================
            # LBP center (10)
            # =================================================

            h, w = crop.shape

            cx1 = int(w * 0.25)
            cx2 = int(w * 0.75)

            cy1 = int(h * 0.25)
            cy2 = int(h * 0.75)

            center_crop = crop[
                cy1:cy2,
                cx1:cx2
            ]

            lbp_center = (
                self.compute_lbp_histogram(
                    center_crop,
                    bins=10
                )
            )

            for j in range(10):

                record[
                    f"lbp_center_{j+1}"
                ] = lbp_center[j]

            # =================================================
            # LBP periphery (10)
            # =================================================

            border_mask = np.ones_like(crop)

            border_mask[
                cy1:cy2,
                cx1:cx2
            ] = 0

            periphery_crop = (
                crop * border_mask
            )

            lbp_periphery = (
                self.compute_lbp_histogram(
                    periphery_crop,
                    bins=10
                )
            )

            for j in range(10):

                record[
                    f"lbp_periphery_{j+1}"
                ] = lbp_periphery[j]

            # =================================================
            # focus/ring (4)
            # =================================================

            record["image_sharpness"] = (
                cv2.Laplacian(
                    crop,
                    cv2.CV_64F
                ).var()
            )

            record["image_focus"] = (
                np.mean(
                    np.abs(
                        cv2.Sobel(
                            crop,
                            cv2.CV_64F,
                            1,
                            1
                        )
                    )
                )
            )

            edges = cv2.Canny(
                crop.astype(np.uint8),
                50,
                150
            )

            record["ring_width"] = (
                np.mean(edges > 0)
            )

            record["ring_intensity"] = (
                np.mean(crop[edges > 0])
                if np.sum(edges > 0) > 0
                else 0
            )

            # =================================================
            # geometry enhancement (6)
            # =================================================

            max_radius, min_radius = (
                self.compute_radius_features(
                    poly
                )
            )

            record["max_radius"] = (
                max_radius
            )

            record["min_radius"] = (
                min_radius
            )

            max_cal, min_cal = (
                self.compute_caliper_features(
                    poly
                )
            )

            record[
                "max_caliper_distance"
            ] = max_cal

            record[
                "min_caliper_distance"
            ] = min_cal

            record[
                "ellipse_similarity"
            ] = (
                self.compute_ellipse_similarity(
                    poly
                )
            )

            record[
                "circle_similarity"
            ] = (
                self.compute_circle_similarity(
                    poly
                )
            )

            records.append(record)

        return pd.DataFrame(records)
