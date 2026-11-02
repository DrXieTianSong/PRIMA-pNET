import logging
from radiomics import featureextractor

TUMOR_LABEL = 6


def get_extractor(bin_width=25, label=TUMOR_LABEL):
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    settings = {
        "binWidth": bin_width,
        "resampledPixelSpacing": None,
        "interpolator": "sitkBSpline",
        "label": label,
        "geometryTolerance": 1e-3,
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.enableAllFeatures()
    return extractor
