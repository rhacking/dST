import numpy as np
from typing import Union, Tuple


def extract_3d(data: np.ndarray, center: np.ndarray, half_size: int):
    imax = np.clip(center + half_size + 1, 0, data.shape).astype(np.int)
    imin = np.clip(center - half_size, 0, data.shape).astype(np.int)

    subvol = data[imin[0]:imax[0], imin[1]:imax[1], imin[2]:imax[2]]

    max_missing = ((center + half_size + 1) - imax).astype(np.int)
    min_missing = (imin - (center - half_size)).astype(np.int)

    return np.pad(subvol, [(min_missing[i], max_missing[i]) for i in range(3)], mode='constant')


def crop_view(data: np.ndarray, crop: Union[float, Tuple[float, float, float]], center_crop: bool = True):
    if type(crop) == float or type(crop) == int:
        if crop > 0.99999:
            return data
        icropx = 1 - crop
        icropy = 1 - crop
        icropz = 1 - crop
    else:
        icropx = 1 - crop[0]
        icropy = 1 - crop[1]
        icropz = 1 - crop[2]

    w, h, l = data.shape

    if center_crop:
        view = data[int(w / 2 * icropx):int(-w / 2 * icropx),
                    int(h / 2 * icropy):int(-h / 2 * icropy),
                    int(l / 2 * icropz):int(-l / 2 * icropz)]
    else:
        view = data[:int(w * (1 - icropx)), :int(h * (1 - icropy)), :int(l * (1 - icropz))]

    return view


def show_ipv(data: np.ndarray):
    import ipyvolume as ipv
    return ipv.quickvolshow(data)


def threshold_otsu(image, nbins=256, ignore=0):
    from skimage.filters.thresholding import histogram
    # Check if the image is multi-colored or not
    if image.min() == image.max():
        raise ValueError("threshold_otsu is expected to work with images "
                         "having more than one color. The input image seems "
                         "to have just one color {0}.".format(image.min()))

    img_flat = image.ravel()
    img_flat = img_flat[img_flat != ignore]
    hist, bin_centers = histogram(img_flat, nbins)
    hist = hist.astype(float)

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of `weight1`/`mean1` should pair with zero values in
    # `weight2`/`mean2`, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold