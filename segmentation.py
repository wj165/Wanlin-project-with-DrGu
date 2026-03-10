import os
import json
import numpy as np
from csbdeep.utils import normalize
from skimage.transform import resize
from skimage.measure import find_contours, regionprops
from stardist.models import StarDist2D


class CellSegmenter:

    def __init__(
        self,
        model_name="2D_versatile_he",
        target_magnification=20,
        native_magnification=None,
        min_area=80,
        prob_thresh=0.15,
        nms_thresh=0.4
    ):

        self.model = StarDist2D.from_pretrained(model_name)

        self.target_mag = target_magnification
        self.native_mag = native_magnification
        self.min_area = min_area
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh

    def compute_scale_factor(self):

        if self.native_mag is None:
            return 1.0

        return self.target_mag / self.native_mag

    def resize_if_needed(self, image):

        scale_factor = self.compute_scale_factor()

        if scale_factor == 1.0:
            return image

        new_shape = (
            int(image.shape[0] * scale_factor),
            int(image.shape[1] * scale_factor)
        )

        resized = resize(
            image,
            new_shape,
            preserve_range=True,
            anti_aliasing=True
        ).astype(np.uint8)

        return resized

    def run(self, image):

        image = self.resize_if_needed(image)

        image_input = normalize(
            image,
            pmin=1,
            pmax=99.8,
            axis=(0, 1)
        )

        labels, _ = self.model.predict_instances(
            image_input,
            prob_thresh=self.prob_thresh,
            nms_thresh=self.nms_thresh
        )

        labels_filtered = self.filter_small_objects(labels)

        return labels_filtered

    def filter_small_objects(self, labels):

        filtered = np.zeros_like(labels, dtype=np.int32)
        new_id = 1

        regions = regionprops(labels)

        for region in regions:

            if region.area >= self.min_area:

                coords = region.coords
                filtered[coords[:, 0], coords[:, 1]] = new_id
                new_id += 1

        return filtered

    def labels_to_geojson(self, labels, patch_id, output_dir):

        features = []

        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != 0]

        for lab in unique_labels:

            mask = labels == lab
            contours = find_contours(mask.astype(float), 0.5)

            for contour in contours:

                coords = [[float(x[1]), float(x[0])] for x in contour]

                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords]
                    },
                    "properties": {
                        "patch_id": patch_id,
                        "cell_label": int(lab)
                    }
                }

                features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"{patch_id}.geojson")

        with open(output_path, "w") as f:
            json.dump(geojson, f)

        return output_path
