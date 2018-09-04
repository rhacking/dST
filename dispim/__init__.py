#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import heapq
import logging
import math
import os
import random
import shutil
import warnings
from functools import partial
from typing import Tuple, Union, Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import progressbar
import scipy
# from memory_profiler import profile
from numba import jit, prange
from scipy import ndimage
from scipy.ndimage.filters import gaussian_laplace, minimum_filter

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', '.*output shape of zoom.*')

BlurFunction = Callable[[np.ndarray], np.ndarray]


class Volume(object):
    def __init__(self, data: np.ndarray, inverted: bool, spacing: Union[Tuple[float, float, float], np.ndarray],
                 is_skewed: bool = True):
        self.data = data
        self.world_transform = np.eye(4)
        self.is_skewed = is_skewed
        self.inverted = inverted
        self.spacing = np.array(spacing) if isinstance(spacing, tuple) else spacing

    @property
    def shape(self):
        return self.data.shape

    @property
    def grid_to_world(self) -> np.ndarray:
        result = np.array([
            [self.spacing[0], 0, 0, 0],
            [0, self.spacing[1] / (np.sqrt(2) if self.is_skewed else 1), 0, 0],
            [0, 0, self.spacing[2] * (np.sqrt(2) if self.is_skewed else 1), 0],
            [0, 0, 0, 1]
        ])
        if self.inverted:
            result = result @ np.array([
                [1, 0, 0, 0],
                [0, 1, 0, self.shape[1]],
                [0, 0, 1, self.shape[2]],
                [0, 0, 0, 1]
            ])
            result = result @ np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ])
        if self.is_skewed:
            shift = self.spacing[1] / math.sqrt(2) / (self.spacing[2] * math.sqrt(2))
            total_shift = math.ceil(self.shape[1] * shift)

            result = result @ np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, (-1 if self.inverted else 1) * self.spacing[1] / math.sqrt(2) / (self.spacing[2] * math.sqrt(2)), 1,
                 0],
                [0, 0, 0, 1]
            ])
            # result = result @ np.array([
            #     [1, 0, 0, 0],
            #     [0, 1, 0, 0],
            #     [0, 0, 1, total_shift if self.inverted else 0],
            #     [0, 0, 0, 1],
            # ])
        return result

    def get_center_of_mass(self):
        return ndimage.center_of_mass(self.data) * self.spacing

    def rot90(self):
        reorder_indices = np.array([0, 2, 1])
        new_data = np.swapaxes(self.data, 1, 2)[:, :, ::-1]
        return Volume(new_data, self.spacing[reorder_indices])

    def save_tiff_single(self, name: str, invert=False, swap_xy=False, path='out'):
        from tifffile import imsave
        logger.debug('Saving single tif {} with shape {}'.format(name, self.data.shape))
        if not os.path.exists(path):
            os.makedirs(path)
        if os.path.exists(path + '/' + name):
            shutil.rmtree(path + '/' + name)
        os.makedirs(path + '/' + name)
        data = self.data
        if invert:
            data = np.swapaxes(data, 0, 2)
            data = np.flipud(data)
            data = np.swapaxes(data, 0, 2)
        if swap_xy:
            data = np.swapaxes(data, 0, 1)
        data = np.swapaxes(data, 1, 0).swapaxes(2, 0)

        imsave(path + '/' + name + '/' + name + '.tif', data)

    def save_tiff(self, name: str, invert=False, swap_xy=False, path='out'):
        from tifffile import imsave
        if not os.path.exists(path):
            os.makedirs(path)
        if os.path.exists(path + '/' + name):
            shutil.rmtree(path + '/' + name)
        os.makedirs(path + '/' + name)
        data = self.data
        print(data.dtype)
        if invert:
            data = np.swapaxes(data, 0, 2)
            data = np.flipud(data)
            data = np.swapaxes(data, 0, 2)
            data = np.swapaxes(data, 0, 2)
        if swap_xy:
            data = np.swapaxes(data, 0, 1)
        for slc_index in range(data.shape[2]):
            imsave(path + '/' + name + '/' + name + str(slc_index) + '.tif',
                   (data[:, :, slc_index] / 2 ** 8).astype(np.uint8).T)

    def update(self, data: np.ndarray = None, inverted: bool = None,
               spacing: Union[Tuple[float, float, float], np.ndarray] = None,
               is_skewed: bool = None, world_transform: np.ndarray = None):
        if data is not None:
            self.data = data
        if world_transform is not None:
            self.world_transform = world_transform
        if is_skewed is not None:
            self.is_skewed = is_skewed
        if inverted is not None:
            self.inverted = inverted
        if spacing is not None:
            self.spacing = np.array(spacing) if isinstance(spacing, tuple) else spacing

        return self

    def crop_view(self, crop: float):
        if crop > 0.99999:
            return self
        w, h, l = self.shape
        icrop = 1 - crop
        view = self.data[int(w / 2 * icrop):int(-w / 2 * icrop),
               int(h / 2 * icrop):int(-h / 2 * icrop),
               int(l / 2 * icrop):int(-l / 2 * icrop)]
        return Volume(view, self.inverted, self.spacing, is_skewed=self.is_skewed)


def save_dual_tiff(name: str, vol_a: Volume, vol_b: Volume, path: str = 'out'):
    from tifffile import imsave
    if not os.path.exists(path):
        os.makedirs(path)
    if os.path.exists(path + '/' + name):
        shutil.rmtree(path + '/' + name)
    os.makedirs(path + '/' + name)
    print(vol_a.data.shape, vol_b.data.shape)
    data = np.array([vol_a.data, vol_b.data, np.zeros(vol_a.data.shape)])

    for slc_index in range(data.shape[3]):
        imsave(path + '/' + name + '/' + name + str(slc_index) + '.tif',
               (data[:, :, :, slc_index] / 2 ** 8).astype(np.uint8).T)


def save_dual_tiff_single(name: str, vol_a: Volume, vol_b: Volume, path: str = 'out'):
    from tifffile import imsave
    if not os.path.exists(path):
        os.makedirs(path)
    if os.path.exists(path + '/' + name):
        shutil.rmtree(path + '/' + name)
    os.makedirs(path + '/' + name)
    data = np.array([vol_a.data, vol_b.data, np.zeros(vol_a.data.shape)])

    data = np.swapaxes(data, 2, 1).swapaxes(3, 1)
    imsave(path + '/' + name + '/' + name + '.tif', data)


def compute_true_interval(vol: Volume, invert: bool, n: int = 100) -> float:
    from scipy.optimize import minimize_scalar
    resolution = vol.spacing
    rand_slice_indices = np.random.randint(0, vol.data.shape[2] - 1, n)

    def compute_error(x):
        shift = (resolution[2] + x) / resolution[1]
        if invert:
            shift = -shift

        error = 0
        for slice_index in rand_slice_indices:
            shifted_b = scipy.ndimage.shift(vol.data[512:-512, 512:-512, slice_index + 1], (0, shift), order=1)
            error += np.mean(
                (vol.data[512:-512, 512:-512, slice_index].astype(np.float) - shifted_b.astype(np.float)) ** 2)
        return error

    result = minimize_scalar(compute_error)

    return resolution[2] + result.x


def unshift_fast(vol: Volume, invert: bool = False, estimate_true_interval: bool = True) -> Volume:
    if estimate_true_interval:
        interval = compute_true_interval(vol, invert)
        vol.spacing[2] = interval
        logger.debug('Estimated volume interval: {}'.format(interval))

    if invert:
        data = unshift_fast_numbai(vol.data, vol.spacing)
        data = np.rot90(data, k=2, axes=(1, 2))
        return vol.update(data=data, inverted=False, spacing=(vol.spacing[0], vol.spacing[1] / np.sqrt(2),
                                                              vol.spacing[2] * np.sqrt(2)), is_skewed=False)
        # return Volume(data, False, (vol.spacing[0], vol.spacing[1] / np.sqrt(2),
        #                      vol.spacing[2] * np.sqrt(2)), is_skewed=False)
    else:
        return vol.update(data=unshift_fast_numba(vol.data, vol.spacing),
                          spacing=(vol.spacing[0], vol.spacing[1] / np.sqrt(2),
                                   vol.spacing[2] * np.sqrt(2)), is_skewed=False)
        # return Volume(unshift_fast_numba(vol.data, vol.spacing), False, (vol.spacing[0], vol.spacing[1] / np.sqrt(2),
        #                                                           vol.spacing[2] * np.sqrt(2)), is_skewed=False)


@jit(nopython=True, parallel=True)
def unshift_fast_numbai(data: np.ndarray, resolution: np.ndarray) -> np.ndarray:
    w, h, d = data.shape
    shift = resolution[1] / math.sqrt(2) / (resolution[2] * math.sqrt(2))
    total_shift = math.ceil(h * shift)
    result = np.zeros((w, h, d + total_shift), dtype=np.uint16)

    for layer in prange(h):
        layer_shift = (h - layer - 1) * shift
        for x in range(d + total_shift):
            x1 = np.int(np.floor(x - layer_shift))
            x2 = np.int(np.ceil(x - layer_shift))

            delta = x - layer_shift - x1

            val1 = data[:, layer, x1] if 0 <= x1 < d else np.zeros((w), dtype=np.uint16)
            val2 = data[:, layer, x2] if 0 <= x2 < d else np.zeros((w), dtype=np.uint16)

            result[:, layer, x] = val1 * (1 - delta) + val2 * delta

    return result


@jit(nopython=True, parallel=True)
def unshift_fast_numba(data: np.ndarray, resolution: np.ndarray) -> np.ndarray:
    w, h, d = data.shape
    shift = resolution[1] / math.sqrt(2) / (resolution[2] * math.sqrt(2))
    total_shift = math.ceil(h * shift)
    result = np.zeros((w, h, d + total_shift), dtype=np.uint16)

    for layer in prange(h):
        layer_shift = layer * shift
        for x in range(d + total_shift):
            x1 = np.int(np.floor(x - layer_shift))
            x2 = np.int(np.ceil(x - layer_shift))

            delta = x - layer_shift - x1

            val1 = data[:, layer, x1] if 0 <= x1 < d else np.zeros((w), dtype=np.uint16)
            val2 = data[:, layer, x2] if 0 <= x2 < d else np.zeros((w), dtype=np.uint16)

            result[:, layer, x] = val1 * (1 - delta) + val2 * delta

    return result


def unshift_fast_diag(vol: Volume, invert: bool = False) -> Volume:
    # FIXME: The resolution is incorrect!!!!! fix it!!!!
    interval = compute_true_interval(vol, invert)
    vol.spacing[2] = interval
    if invert:
        data = unshift_fast_numbai_diag(vol.data, vol.spacing)
        data = np.rot90(data, k=1, axes=(1, 2))
        return Volume(data, vol.spacing)
    else:
        return Volume(unshift_fast_numba_diag(vol.data, vol.spacing), vol.spacing)


@jit(nopython=True, parallel=True)
def unshift_fast_numbai_diag(data: np.ndarray, resolution: np.ndarray, invert: bool = False) -> np.ndarray:
    w, h, d = data.shape
    shift = resolution[2] / resolution[1]
    total_shift = math.ceil(d * shift)
    result = np.zeros((w, h + total_shift, d), dtype=np.uint16)

    for layer in prange(d):
        layer_shift = (d - layer - 1) * shift
        for x in range(h + total_shift):
            x1 = np.int(np.floor(x - layer_shift))
            x2 = np.int(np.ceil(x - layer_shift))

            delta = x - layer_shift - x1

            val1 = data[:, x1, layer] if 0 <= x1 < h else np.zeros((w), dtype=np.uint16)
            val2 = data[:, x2, layer] if 0 <= x2 < h else np.zeros((w), dtype=np.uint16)

            result[:, x, layer] = val1 * (1 - delta) + val2 * delta

    return result


@jit(nopython=True, parallel=True)
def unshift_fast_numba_diag(data: np.ndarray, resolution: np.ndarray, invert: bool = False) -> np.ndarray:
    w, h, d = data.shape
    shift = resolution[2] / resolution[1]
    total_shift = math.ceil(d * shift)
    result = np.zeros((w, h + total_shift, d), dtype=np.uint16)

    for layer in prange(d):
        layer_shift = layer * shift
        for x in range(h + total_shift):
            x1 = np.int(np.floor(x - layer_shift))
            x2 = np.int(np.ceil(x - layer_shift))

            delta = x - layer_shift - x1

            val1 = data[:, x1, layer] if 0 <= x1 < h else np.zeros((w), dtype=np.uint16)
            val2 = data[:, x2, layer] if 0 <= x2 < h else np.zeros((w), dtype=np.uint16)

            result[:, x, layer] = val1 * (1 - delta) + val2 * delta

    return result


def unshift(vol: Volume, invert: bool) -> Volume:
    """
    Deskew the given volume

    :param vol: The volume to deskew
    :param invert: Whether the stage was shifted from left to right (or right to left)
    :return: The deskewed volume
    """
    # Determine the shift step
    shift = vol.spacing[2] / vol.spacing[1]
    logger.debug('Unshifing with slice shift of {} and volume shape of {}'.format(shift, vol.data.shape))

    new_vol = [np.zeros(vol.data.shape[0:1])]
    # Compute the maximum shift required
    total_shift = math.ceil(vol.data.shape[2] * shift)
    # If the stage direction is inverted, invert the shifting as well
    if invert:
        shift = -shift

    data = vol.data

    # Scale the volume to accommodate the maximum shift possible
    if not invert:
        data = np.pad(data, ((0, 0), (0, total_shift), (0, 0)), mode='constant', constant_values=0)
    else:
        data = np.pad(data, ((0, 0), (total_shift, 0), (0, 0)), mode='constant', constant_values=0)

    # Iterate over all slices
    with progressbar.ProgressBar(max_value=data.shape[2], redirect_stderr=True) as bar:
        for i, slc in bar(enumerate(data.swapaxes(0, 2).swapaxes(1, 2))):
            # Perform the actual shifting (order-3 spline interpolation)
            slc = scipy.ndimage.interpolation.shift(slc, (0, shift * i), order=2)
            new_vol.append(slc)

    new_vol = new_vol[1:]

    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    # print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)

    # Swap the axes such that they're in order again (x, y, z)
    result = np.array(new_vol)
    result = np.swapaxes(result, 0, 2)
    result = np.swapaxes(result, 0, 1)
    logger.debug("Shape after unshifting: {}".format(result.shape))
    result_vol = Volume(result, vol.spacing)

    return result_vol


def register_manual_translation(vol_a: Volume, vol_b: Volume) -> np.ndarray:
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


def register_syn(vol_a: Volume, vol_b: Volume):
    from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
    from dipy.align.imwarp import DiffeomorphicMap
    from dipy.align.metrics import CCMetric

    metric = CCMetric(3)
    level_iters = [8, 8, 4]
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)

    mapping = sdr.optimize(vol_a.data, vol_b.data, vol_a.grid_to_world, vol_b.grid_to_world, None)

    transformed_b = mapping.transform(vol_b.data)

    return vol_a, vol_b.update(data=transformed_b)


def register_2d(vol_a: Volume, vol_b: Volume):
    from dipy.align.imaffine import (transform_centers_of_mass,
                                     MutualInformationMetric,
                                     AffineRegistration,
                                     AffineMap,
                                     )
    from dipy.align.transforms import AffineTransform2D
    vol_a_flat = np.mean(vol_a.data, axis=2)
    vol_b_flat = np.mean(vol_b.data, axis=2)

    nbins = 32
    sampling_prop = 0.65
    metric = MutualInformationMetric(nbins, sampling_prop)

    level_iters = [50000, 3000, 80]

    sigmas = [5.0, 2.0, 0.0]

    factors = [4, 2, 1]

    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    transform = AffineTransform2D()
    params0 = None

    I = np.eye(4)

    init = transform_centers_of_mass(vol_a_flat, I, vol_b_flat, I)

    starting_affine = init.affine

    affine = affreg.optimize(vol_a_flat, vol_b_flat, transform, params0,
                             I, I, starting_affine=starting_affine)

    b_trans = vol_b.world_transform
    b_trans[0:2, 0:2] = affine.affine[0:2, 0:2]
    b_trans[0:2, 3] = affine.affine[0:2, 2]

    return vol_a, vol_b.update(world_transform=b_trans)


def register_dipy(vol_a: Volume, vol_b: Volume, init_translation: Optional[np.ndarray] = None, crop: float = 0.6):
    from dipy.align.imaffine import (transform_centers_of_mass,
                                     MutualInformationMetric,
                                     AffineRegistration,
                                     AffineMap,
                                     )
    from dipy.align.transforms import (TranslationTransform3D,
                                       RigidTransform3D,
                                       AffineTransform3D)

    subvol_a = vol_a.crop_view(crop)
    subvol_b = vol_b.crop_view(crop)

    print(subvol_b.grid_to_world)

    logger.debug('Sub-volume A sgaoe: ' + str(subvol_a.shape))
    logger.debug('Sub-volume B sgaoe: ' + str(subvol_b.shape))

    if init_translation is not None:
        init = AffineMap(np.array([
            [1, 0, 0, init_translation[0]],
            [0, 1, 0, init_translation[1]],
            [0, 0, 1, init_translation[2]],
            [0, 0, 0, 1]
        ]), vol_a.shape, vol_a.grid_to_world, vol_b.shape, vol_b.grid_to_world)
    else:
        init = transform_centers_of_mass(subvol_a.data,
                                         subvol_a.grid_to_world,
                                         subvol_b.data,
                                         subvol_b.grid_to_world)

    nbins = 32
    sampling_prop = 0.65
    metric = MutualInformationMetric(nbins, sampling_prop)

    level_iters = [50000, 3000, 80]

    sigmas = [5.0, 2.0, 0.0]

    factors = [4, 2, 1]

    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    transform = TranslationTransform3D()
    params0 = None
    starting_affine = init.affine
    translation = affreg.optimize(subvol_a.data, subvol_b.data, transform, params0,
                                  subvol_a.grid_to_world, subvol_b.grid_to_world,
                                  starting_affine=starting_affine)

    logger.debug('Registration translation: ' + str(translation))

    transform = RigidTransform3D()
    params0 = None
    starting_affine = translation.affine
    rigid = affreg.optimize(subvol_a.data, subvol_b.data, transform, params0,
                            subvol_a.grid_to_world, subvol_b.grid_to_world,
                            starting_affine=starting_affine)

    logger.debug('Registration rigid: ' + str(rigid))

    affreg = AffineRegistration(metric=metric,
                                level_iters=[50000, 2000, 120],
                                sigmas=sigmas,
                                factors=factors)

    transform = AffineTransform3D()
    params0 = None
    starting_affine = rigid.affine

    affine = affreg.optimize(subvol_a.data, subvol_b.data, transform, params0,
                             subvol_a.grid_to_world, subvol_b.grid_to_world,
                             starting_affine=starting_affine)

    logger.debug('Registration affine: ' + str(affine))

    vol_b.world_transform = np.array(affine.affine)

    return vol_a, vol_b


def make_isotropic(vol_a: Volume, vol_b: Volume):
    import scipy.interpolate
    min_res = min(np.min(vol_a.spacing), np.min(vol_b.spacing))
    # zoomed_volA = scipy.interpolate.interp1d(ax, vol_a.data, axis=2)(np.linspace(0, len(ax)-1, len(ax)*(vol_a.resolution[2]/vol_b.resolution[2]))).astype(np.uint16)
    zoomed_volA = scipy.ndimage.zoom(vol_a.data, vol_a.spacing / min_res, order=1)
    # zoomed_volB = scipy.interpolate.interp1d(bx, vol_b.data, axis=1)(np.linspace(0, len(bx)-1, len(bx)*(vol_b.resolution[1]/vol_a.resolution[1]))).astype(np.uint16)
    zoomed_volB = scipy.ndimage.zoom(vol_b.data, vol_b.spacing / min_res, order=1)

    result_vol_a = vol_a.update(zoomed_volA,
                                spacing=(vol_a.spacing[0], vol_a.spacing[1], vol_b.spacing[2]))
    result_vol_b = vol_b.update(zoomed_volB,
                                spacing=(vol_b.spacing[0], vol_a.spacing[1], vol_b.spacing[2]))

    return result_vol_a, result_vol_b


def fuse(vol_a: Volume, vol_b: Volume):
    # FIXME: Do something clever with is_skewed
    return Volume(np.floor_divide(vol_a.data + vol_b.data, 2).astype(np.uint16), False, vol_a.spacing, is_skewed=False)


def deconvolve_rl(vol: Volume, n: int, blur: BlurFunction) -> Volume:
    # TODO: Possibly switch this to use the skimage function (blur function must then be converted to PSF first)
    estimate = vol.data
    with progressbar.ProgressBar(max_value=n, redirect_stderr=True) as bar:
        for i in bar(range(n)):
            estimate = estimate * blur(vol.data / (blur(estimate) + 1E-9))

    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)

    est_min = np.min(estimate)
    est_max = np.max(estimate)

    return Volume(np.clip((estimate + est_min) / (est_max - est_min), 0.0, 1.0), vol.spacing)


def deconvolve(vol_a: Volume, vol_b: Volume, n: int, blurA: BlurFunction, blurB: BlurFunction) -> Volume:
    view_a, view_b = vol_a.data, vol_b.data
    # view_a = np.clip(
    #     scipy.ndimage.zoom(view_a, (2, 2, 2)), 0.0, 1.0)
    # view_b = np.clip(
    #     scipy.ndimage.zoom(view_b, (2, 2, 2)), 0.0, 1.0)

    psf_A = np.zeros((17, 17, 17))
    psf_A[8, 8, 8] = 1
    psf_A = blurA(psf_A)

    psf_B = np.zeros((17, 17, 17))
    psf_B[8, 8, 8] = 1
    psf_B = blurB(psf_B)

    estimate = (view_a + view_b) / 2
    with progressbar.ProgressBar(max_value=n, redirect_stderr=True) as bar:
        for i in bar(range(n)):
            estimate_A = estimate * blurA(view_a / (blurA(estimate) + 1E-9))
            estimate = estimate_A * blurB(view_b / (blurB(estimate_A) + 1E-9))

    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)

    return Volume(estimate, vol_a.spacing)


# PSF Extraction
def gaussian_1d(x, A, mu, sigma):
    return (A * np.exp(-.5 * ((x - mu) / (sigma)) ** 2) /
            (sigma * math.sqrt(2 * math.pi)))


def gaussian_2d(x, A, mu_x, mu_y, sigma):
    return (A * np.exp(-(
            (x[0] - mu_x) ** 2 / (2 * sigma ** 2) +
            (x[1] - mu_y) ** 2 / (2 * sigma ** 2)
    )))


def gaussian_3d(x, A, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z):
    return (A * np.exp(-(
            (x[0] - mu_x) ** 2 / (2 * sigma_x ** 2) +
            (x[1] - mu_y) ** 2 / (2 * sigma_y ** 2) +
            (x[2] - mu_z) ** 2 / (2 * sigma_z ** 2)
    )))


def gaussian_fit_1d(data):
    from scipy.optimize import curve_fit

    xs = np.arange(data.shape[0])

    upper_bounds = [np.inf, data.shape[0], np.inf]
    popt, _ = curve_fit(gaussian_1d, xs, data[xs], bounds=(np.zeros((3,)), upper_bounds))

    return popt[2]


def fit_gaussian2(data: np.ndarray):
    from scipy.optimize import curve_fit

    xs = np.arange(data.shape[0])
    ys = np.arange(data.shape[1])
    xv, yv = np.meshgrid(xs, ys)
    xv = xv.ravel()
    yv = yv.ravel()
    xdata = np.vstack((xv, yv))

    upper_bounds = [np.inf, data.shape[0], data.shape[1], np.inf]
    popt, _ = curve_fit(gaussian_2d, xdata, data[xv, yv], bounds=(np.zeros((4,)), upper_bounds))

    return popt[3]


def fit_gaussian(data: np.ndarray):
    from scipy.optimize import curve_fit
    xs = np.arange(data.shape[0])
    ys = np.arange(data.shape[1])
    zs = np.arange(data.shape[2])
    xv, yv, zv = np.meshgrid(xs, ys, zs)
    xv = xv.ravel()
    yv = yv.ravel()
    zv = zv.ravel()
    xdata = np.vstack((xv, yv, zv))

    # start = [data.shape[0]//2, data.shape[1]//2, data.shape[2]//2, r, r, r]
    upper_bounds = [np.inf, data.shape[0], data.shape[1], data.shape[2], np.inf, np.inf, np.inf]
    p0 = (1, data.shape[0] // 2, data.shape[1] // 2, data.shape[2] // 2, 1, 1, 1)

    popt, _ = curve_fit(gaussian_3d, xdata, data[xv, yv, zv], bounds=(np.zeros((7,)), upper_bounds), p0=p0)

    # print(popt[4:])
    return popt[4:]


def extract_1d(data, point, area):
    return data[max(0, point - area):min(data.shape[0], point + area)]


def extract_2d(data, point, area):
    return data[max(0, point[0] - area):min(data.shape[0], point[0] + area),
           max(0, point[1] - area):min(data.shape[1], point[1] + area)]


def compute_psf(vol: Volume):
    area_size = 8

    data = vol.data

    data_slice_averages = np.max(data, axis=(0, 1))
    slice_indices = heapq.nlargest(150, range(len(data_slice_averages)), key=data_slice_averages.take)
    import skimage.feature
    blobs = []
    for slice_index in slice_indices:
        blobs_slice = skimage.feature.peak_local_max(data[:, :, slice_index], min_distance=5, exclude_border=True,
                                                     indices=True, threshold_abs=0.15)
        for blob_slice in blobs_slice:
            blobs.append((blob_slice[0], blob_slice[1], slice_index))

    np.random.shuffle(blobs)

    sigmas_z = []
    sigmas_xy = []

    logger.info("Processing {} blobs...".format(min(len(blobs), 1000)))
    for blob in blobs[:1000]:
        blob_area_1d = extract_1d(data[blob[0], blob[1], :], blob[2], area_size)
        blob_area_2d = extract_2d(data[:, :, blob[2]], [blob[0], blob[1]], area_size)

        try:
            sigmas_z.append(gaussian_fit_1d(blob_area_1d))
            sigmas_xy.append(fit_gaussian2(blob_area_2d))
        except RuntimeError as e:
            logger.warning(e)
            pass

    return np.median(sigmas_z, axis=0) * vol.spacing[2], np.median(sigmas_xy, axis=0) * vol.spacing[0]


def load_volumes(paths: List[str], spacing: Tuple[float, float, float], scale: float = None, is_skewed: bool = True):
    # FIXME: Only support loading a single volume or single volume pair
    from tifffile import imread
    import gc

    datas = []
    for path in paths:
        logger.info("Loading volume from {}".format(path))
        data = imread(path)

        logger.info("Initial volume shape: {}".format(data.shape))

        if len(data.shape) > 3:
            if scale is not None:
                data_a = scipy.ndimage.zoom(data[:, 0, :, :], scale, order=1)
                data_b = scipy.ndimage.zoom(data[:, 1, :, :], scale, order=1)

                datas.append(data_a)
                datas.append(data_b)
            else:
                datas.append(data[:, 0, :, :])
                datas.append(data[:, 1, :, :])
        else:
            if scale is not None:
                data = scipy.ndimage.zoom(data, scale, order=1)
            datas.append(data)

        logger.debug("Final volume shape: {}".format(data.shape))
        del data
        gc.collect()

    volumes = []

    print(datas[0].nbytes)
    print(datas[0].dtype)
    print(datas[0].max())

    # TODO: Handle different data types (aside from uint16)

    for i in range(len(datas)):
        volumes.append(Volume(datas[i].swapaxes(0, 2).swapaxes(0, 1), bool(i), spacing, is_skewed=is_skewed))
        gc.collect()

    return tuple(volumes)


# def process(args):
#     import gc
#     if args.pixel_size > args.interval:
#         logging.warning('The pixel size is greater than the interval. Arguments may have been swapped. ')
#
#     if 'deconvolve' in args.operations and args.deconvolve_sigma is None:
#         logging.error('Blurring operation must be specified when deconvolving')
#         import sys
#         sys.exit(1)
#
#     logging.basicConfig(level=logging.INFO)
#     logging.info('Loading volume a...')
#
#     result = load_volumes([args.spim_a, args.spim_b], (args.pixel_size, args.pixel_size, args.interval / math.sqrt(2)), args.scale)
#
#     for op in args.operations:
#         if op == 'deskew':
#             logging.info('Deskewing volume a...')
#             vol_a = unshift(result[0], invert=False)
#             logging.info('Deskewing volume b...')
#             vol_b = unshift(result[1], invert=False)
#             result = (vol_a, vol_b)
#         elif op == 'register':
#             logging.info('Registering...')
#             vol_b = result[1].rot90()
#             result = register_ants(result[0], vol_b)
#         elif op == 'fuse':
#             logging.info('Fusing...')
#             result = fuse(result[0], result[1])
#         elif op == 'deconvolve_seperate':
#             if np.all(result[0].resolution != result[1].resolution):
#                 logging.error('Both volumes must have equal resolution to deconvolve. ')
#             if np.all(result[0].data.shape != result[1].data.shape):
#                 logging.error('Both volumes must have equal dimensions to deconvolve. ')
#
#             logging.info('Deconvolving...')
#             sigma = args.deconvolve_sigma
#             blur_a = lambda vol: scipy.ndimage.filters.gaussian_filter1d(vol, args.deconvolve_sigma, axis=2)
#             blur_b = lambda vol: scipy.ndimage.filters.gaussian_filter1d(vol, args.deconvolve_sigma, axis=1)
#
#             result = (deconvolve_rl(result[0], 24, blur_a), deconvolve_rl(result[1], 24, blur_b))
#         elif op == 'deconvolve':
#             if np.all(result[0].resolution != result[1].resolution):
#                 logging.error('Both volumes must have equal resolution to deconvolve. ')
#             if np.all(result[0].data.shape != result[1].data.shape):
#                 logging.error('Both volumes must have equal dimensions to deconvolve. ')
#
#             logging.info('Deconvolving...')
#             sigma = args.deconvolve_sigma
#             vol_a = result[0]
#             vol_b = result[1]
#             for sigma in [0.25, 0.5, 0.75, 1.0, 1.25]:
#                 blur_a = lambda vol: scipy.ndimage.filters.gaussian_filter1d(vol, sigma, axis=2)
#                 blur_b = lambda vol: scipy.ndimage.filters.gaussian_filter1d(vol, sigma, axis=1)
#
#                 result = deconvolve(result[0], result[1], 24, blur_a, blur_b)
#
#     if isinstance(result, tuple):
#         if args.save_rg:
#             logging.info('Saving volume a/b...')
#             save_dual_tiff(args.output+'_AB', result[0], result[1])
#         else:
#             logging.info('Saving volume a...')
#             if args.single_file_out:
#                 result[0].save_tiff_single(args.output+'_A_single', swap_xy=args.swap_xy_a)
#             result[0].save_tiff(args.output+'_A', swap_xy=args.swap_xy_a)
#             logging.info('Saving volume b...')
#             if args.single_file_out:
#                 result[1].save_tiff_single(args.output+'_B_single', swap_xy=args.swap_xy_b)
#             result[1].save_tiff(args.output+'_B', swap_xy=args.swap_xy_b)
#     else:
#         logging.info('Saving volume...')
#         if args.single_file_out:
#             result.save_tiff_single(args.output+"_single")
#         result.save_tiff(args.output)


def extract_psf(args):
    # if 5 > 2: return
    logger.info('Loading volume...')
    from tifffile import imread
    data = imread(args.volume).swapaxes(0, 1)
    # vol_a = Volume(data_a.swapaxes(0, 2).swapaxes(0, 1 - args.swap_xy_a),
    #                (args.pixel_size, args.pixel_size, args.interval / math.sqrt(2)))
    # w, h, l = 20, 40, 60
    # data = np.zeros((w, h, l))
    # origin = np.array((w/2, h/2, l/2))
    # for x in range(w):
    #     for y in range(h):
    #         for z in range(l):
    #             data[x, y, z] = gaussian_3d([x, y, z], 10, 5, 10, 2, 5, 2) + gaussian_3d([x, y, z], 15, 18, 25, 2, 5, 2)
    #             # data[x, y, z] = np.linalg.norm(origin-[x, y, z]) < 4
    # ydata = gaussian_3d(xdata.T, 5, 10, 15, 3, 3, 3)
    # fit_gaussian(data)
    compute_psf(data)


def localMinima(data, threshold):
    from numpy import ones, nonzero, transpose

    if threshold is not None:
        peaks = data < threshold
    else:
        peaks = ones(data.shape, dtype=data.dtype)

    peaks &= data == minimum_filter(data, size=(3,) * data.ndim)
    return transpose(nonzero(peaks))


def blobLOG(data, scales=range(1, 10, 1), threshold=-30.0):
    """Find blobs. Returns [[scale, x, y, ...], ...]"""
    from numpy import empty, asarray

    data = asarray(data)
    scales = asarray(scales)

    log = empty((len(scales),) + data.shape, dtype=data.dtype)
    for slog, scale in zip(log, scales):
        slog[...] = scale ** 2 * gaussian_laplace(data, scale)

    for slog, scale in zip(log, scales):
        # plt.title(scale)
        # plt.imshow(slog[slog.shape[0] // 2], vmin=np.min(log), vmax=np.max(log))
        # plt.colorbar()
        # plt.show()
        pass

    plt.title(scale)
    plt.imshow(data[data.shape[0] // 2])
    plt.colorbar()
    # plt.show()

    peaks = localMinima(log, threshold=threshold)
    peaks[:, 0] = scales[peaks[:, 0]]
    return peaks


def sphereIntersection(r1, r2, d):
    # https://en.wikipedia.org/wiki/Spherical_cap#Application

    valid = (d < (r1 + r2)) & (d > 0)
    return (np.pi * (r1 + r2 - d) ** 2
            * (d ** 2 + 6 * r2 * r1
               + 2 * d * (r1 + r2)
               - 3 * (r1 - r2) ** 2)
            / (12 * d)) * valid


def circleIntersection(r1, r2, d):
    # http://mathworld.wolfram.com/Circle-CircleIntersection.html
    from numpy import arccos, sqrt

    return (r1 ** 2 * arccos((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1))
            + r2 ** 2 * arccos((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2))
            - sqrt((-d + r1 + r2) * (d + r1 - r2)
                   * (d - r1 + r2) * (d + r1 + r2)) / 2)


def findBlobs(img, scales=range(1, 10), threshold=30.0, max_overlap=0.05):
    from numpy import ones, triu, seterr
    old_errs = seterr(invalid='ignore')

    peaks = blobLOG(img, scales=scales, threshold=-threshold)
    radii = peaks[:, 0]
    positions = peaks[:, 1:]

    print(peaks)

    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)

    if positions.shape[1] == 2:
        intersections = circleIntersection(radii, radii.T, distances)
        volumes = np.pi * radii ** 2
    elif positions.shape[1] == 3:
        intersections = sphereIntersection(radii, radii.T, distances)
        volumes = 4 / 3 * np.pi * radii ** 3
    else:
        raise ValueError("Invalid dimensions for position ({}), need 2 or 3."
                         .format(positions.shape[1]))

    delete = ((intersections > (volumes * max_overlap))
              # Remove the smaller of the blobs
              & ((radii[:, None] < radii[None, :])
                 # Tie-break
                 | ((radii[:, None] == radii[None, :])
                    & triu(ones((len(peaks), len(peaks)), dtype='bool'))))
              ).any(axis=1)

    seterr(**old_errs)
    return peaks[~delete]
