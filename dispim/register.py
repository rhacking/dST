import logging
from typing import Tuple

import numpy as np
from dipy.align.transforms import TranslationTransform2D, TranslationTransform3D
from matplotlib import pyplot as plt

from dispim import Volume, metrack

logger = logging.getLogger(__name__)


def register_manual_translation(vol_a: Volume, vol_b: Volume) -> np.ndarray:
    """
    Show user input window to determine the translation in a two axes between two volumes

    :param vol_a: The first volume to show
    :param vol_b: The second volume to show
    :return: The translation entered by the user
    """
    plt.ion()
    fig, ax = plt.subplots(1, 2)
    w, h, l = vol_a.data.shape
    ax[0].imshow(
        np.mean(vol_a.data[w // 4:-w // 4, h // 4:-h // 4, l // 4:-l // 4], axis=2, dtype=np.float32).T / 2 ** 16)
    w, h, l = vol_b.data.shape
    ax[1].imshow(
        np.mean(vol_b.data[w // 4:-w // 4, h // 4:-h // 4, l // 4:-l // 4], axis=2, dtype=np.float32).T / 2 ** 16)

    points = plt.ginput(2, timeout=0)
    plt.close()

    fig, ax = plt.subplots(1, 2)
    w, h, l = vol_a.data.shape
    ax[0].imshow(
        np.mean(vol_a.data[w // 4:-w // 4, h // 4:-h // 4, l // 4:-l // 4], axis=1, dtype=np.float32).T / 2 ** 16)
    w, h, l = vol_b.data.shape
    ax[1].imshow(
        np.mean(vol_b.data[w // 4:-w // 4, h // 4:-h // 4, l // 4:-l // 4], axis=1, dtype=np.float32).T / 2 ** 16)

    points2 = plt.ginput(2, timeout=0)
    plt.close()

    return np.array([points[1][0] - points[0][0], points[1][1] - points[0][1], points2[1][1] - points2[0][1]])


def register_2d(vol_a: Volume, vol_b: Volume, axis: int = 2, transform_cls: type = TranslationTransform2D) -> Tuple[
                Volume, Volume]:
    """
    Perform axis-reduced registration on two volumes

    :param vol_a: The fixed volume
    :param vol_b: The moving volume
    :param axis: The axis along which to perform the reduction
    :param transform_cls: The type of registration transform to compute
    :return: The updated volumes
    """
    from dipy.align.imaffine import (MutualInformationMetric,
                                     AffineRegistration)
    from dispim.metrics import MUTUAL_INFORMATION_METRIC, MUTUAL_INFORMATION_GRADIENT_METRIC

    vol_a_flat = np.mean(vol_a.data, axis=axis)
    vol_b_flat = np.mean(vol_b.data, axis=axis)

    if metrack.is_tracked(MUTUAL_INFORMATION_METRIC) or metrack.is_tracked(MUTUAL_INFORMATION_GRADIENT_METRIC):
        def callback(value: float, gradient: float):
            metrack.append_metric(MUTUAL_INFORMATION_METRIC, (None, value))
            metrack.append_metric(MUTUAL_INFORMATION_GRADIENT_METRIC, (None, gradient))
    else:
        callback = None

    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop, sampling_type='grid')

    level_iters = [5000000, 1000000, 500000, 200000, 70000, 70000]
    sigmas = [15.0, 7.0, 3.0, 2.0, 1.0, 0.0]
    factors = [8, 4, 2, 2, 1, 1]

    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    transform = transform_cls()
    params0 = None
    axes = np.ones((4,), dtype=np.bool)
    axes[axis] = False
    starting_affine = vol_b.world_transform[axes][:, axes]

    affine = affreg.optimize(vol_a_flat, vol_b_flat, transform, params0,
                             vol_a.grid_to_world_2d(axis), vol_b.grid_to_world_2d(axis),
                             starting_affine=starting_affine)

    # TODO: Do something more clever
    if axis == 0:
        vol_b.world_transform[1:3, 1:3] = affine.affine[:2, :2]
        vol_b.world_transform[1:3, 3] = affine.affine[:2, 2]
    elif axis == 1:
        vol_b.world_transform[0, 0] = affine.affine[0, 0]
        vol_b.world_transform[2, 0] = affine.affine[1, 0]
        vol_b.world_transform[0, 2] = affine.affine[0, 1]
        vol_b.world_transform[2, 2] = affine.affine[1, 1]
    elif axis == 2:
        vol_b.world_transform[:2, :2] = affine.affine[:2, :2]
        vol_b.world_transform[:2, 3] = affine.affine[:2, 2]

    logger.debug('Registration transform: ' + str(vol_b.world_transform))

    return vol_a, vol_b


def register_com(vol_a: Volume, vol_b: Volume) -> Tuple[Volume, Volume]:
    """
    Perform center-of-mass registration on two volumes

    :param vol_a: The fixed volume
    :param vol_b: The moving volume
    :return: The updated volumes
    """
    from dipy.align.imaffine import transform_centers_of_mass

    affine = transform_centers_of_mass(vol_a, vol_a.grid_to_world, vol_b, vol_b.grid_to_world)

    vol_b.world_transform[:] = np.array(affine.affine)
    return vol_a, vol_b


def register_dipy(vol_a: Volume, vol_b: Volume,
                  sampling_prop: float = 1.0, crop: float = 0.8, transform_cls: type = TranslationTransform3D) -> Tuple[
                  Volume, Volume]:
    """
    Perform registration on two volumes

    :param vol_a: The fixed volume
    :param vol_b: The moving volume
    :param sampling_prop: Proportion of data to be used (0-1)
    :param crop: The value to use for cropping both volumes before registration (0-1
    :param transform_cls: The type of registration transform to compute
    :return: The updated volumes
    """
    from dipy.align.imaffine import (MutualInformationMetric,
                                     AffineRegistration)

    from dispim.util import crop_view
    from dispim.metrics import MUTUAL_INFORMATION_METRIC, MUTUAL_INFORMATION_GRADIENT_METRIC

    logger.debug('Sampling prop: ' + str(sampling_prop))
    logger.debug('Crop: ' + str(crop))

    subvol_a = crop_view(vol_a, crop, center_crop=False)
    subvol_b = crop_view(vol_b, crop, center_crop=False)

    logger.debug('Sub-volume A size: ' + str(subvol_a.shape))
    logger.debug('Sub-volume B size: ' + str(subvol_b.shape))

    level_iters = [40000, 10000, 1000, 500]

    sigmas = [7.0, 3.0, 1.0, 0.0]

    factors = [4, 2, 1, 1]

    if metrack.is_tracked(MUTUAL_INFORMATION_METRIC) or metrack.is_tracked(MUTUAL_INFORMATION_GRADIENT_METRIC):
        def callback(value: float, gradient: float):
            metrack.append_metric(MUTUAL_INFORMATION_METRIC, (None, value))
            metrack.append_metric(MUTUAL_INFORMATION_GRADIENT_METRIC, (None, gradient))
    else:
        callback = None

    affreg = AffineRegistration(
        metric=MutualInformationMetric(32, None if sampling_prop > 0.99 else sampling_prop, sampling_type='grid'),
        level_iters=level_iters,
        sigmas=sigmas,
        factors=factors)

    transform = transform_cls()
    params0 = None

    starting_affine = vol_b.world_transform
    affine = affreg.optimize(subvol_a, subvol_b, transform, params0,
                             subvol_a.grid_to_world, subvol_b.grid_to_world,
                             starting_affine=starting_affine)

    logger.debug('Registration transform: ' + str(transform))

    vol_b = Volume(vol_b, world_transform=np.array(affine.affine))

    return vol_a, vol_b
