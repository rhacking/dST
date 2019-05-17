from typing import Union, Tuple

import numpy as np

from dispim import Volume


def extract_3d(data: np.ndarray, center: np.ndarray, half_size: int):
    """
    Extract an area around a point in a 3d numpy array, zero padded as necessary such that the specified point is at the
    center

    :param data: The numpy array to extract from
    :param center: The point around which to extract
    :param half_size: The half-size of the extracted area (full size is half_size*2+1, where the th center point is
                        center)
    :return: The extracted area
    """
    # FIXME: Doesn't always return the expected shape
    imax = np.clip(center + half_size + 1, 0, data.shape).astype(np.int)
    imin = np.clip(center - half_size, 0, data.shape).astype(np.int)

    subvol = data[imin[0]:imax[0], imin[1]:imax[1], imin[2]:imax[2]]

    max_missing = ((center + half_size + 1) - imax).astype(np.int)
    min_missing = (imin - (center - half_size)).astype(np.int)

    return np.pad(subvol, [(min_missing[i], max_missing[i]) for i in range(3)], mode='constant')


def crop_view(data: np.ndarray, crop: Union[float, Tuple[float, float, float]], center_crop: bool = True):
    """
    Get a cropped view of a 3d numpy array (does not modify the input)

    :param data: The numpy array to crop
    :param crop: The percentage to crop in each dimension
    :param center_crop: If True, the crop is centered around the middle of the volume, otherwise, the crop expands from
                                    (0, 0, 0)
    :return: The cropped view
    """
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


def plot_ortho_overlayed(vol_a: Volume, vol_b: Volume, axis=2, pixel_size: float = 1.0) -> None:
    """
    Plot two axis-reduced volumes overlayed as two channels (red and green), taking into account the spacing of both volumes

    :param vol_a: The first volume to plot (red)
    :param vol_b: The second volume to plot (green)
    :param axis: The axis along which both volumes will be reduced
    :param pixel_size: The size of a pixel, relative to the spacing of the the volumes
    """
    from scipy.ndimage.interpolation import zoom
    import matplotlib.pyplot as plt

    vol_a_zoomed = np.mean(zoom(vol_a, np.array(vol_a.spacing) * pixel_size), axis=axis)
    vol_b_zoomed = np.mean(zoom(vol_b, np.array(vol_b.spacing) * pixel_size), axis=axis)
    b_channel = np.zeros_like(vol_a_zoomed)

    max_val = max(vol_a_zoomed.max(), vol_b_zoomed.max())
    min_val = min(vol_a_zoomed.min(), vol_b_zoomed.min())

    vol_a_zoomed = (vol_a_zoomed - min_val) / (max_val - min_val)
    vol_b_zoomed = (vol_b_zoomed - min_val) / (max_val - min_val)

    plt.imshow(np.stack([vol_a_zoomed, vol_b_zoomed, b_channel], axis=2))
    plt.show()


def show_ipv(data: np.ndarray):
    """
    Show a 3d visualization of 3d numpy array
    :param data: The numpy array to show
    :return: The ipyvolume figure
    """
    import ipyvolume as ipv
    return ipv.quickvolshow(data)


def threshold_otsu(image: np.ndarray, nbins: int = 256, ignore: int = 0) -> float:
    """
    Compute the Otsu threshold for a numpy array, without taking into account empty areas

    :param image: The volume to compute the threshold for
    :param nbins: The number of bins used
    :param ignore: The value to ignore
    :return: The Otsu threshold
    """
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
