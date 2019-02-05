#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import math
import os
import shutil
import warnings
from typing import Tuple, Union, Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import progressbar
import scipy
from numba import jit, prange
from scipy import ndimage

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', '.*output shape of zoom.*')

BlurFunction = Callable[[np.ndarray], np.ndarray]

debug = False


class Volume(object):
    def __init__(self, data: np.ndarray, inverted: bool, spacing: Union[Tuple[float, float, float], np.ndarray],
                 is_skewed: bool = True, flipped: List[bool] = None):
        self.psf = None
        self.data = data
        self.world_transform = np.eye(4)
        self.is_skewed = is_skewed
        self.inverted = inverted
        self.flipped = [False, False, False] if flipped is None else flipped
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
        result = result @ np.array([
            [-1 if self.flipped[0] else 1, 0, 0, 0],
            [0, -1 if self.flipped[1] else 1, 0, 0],
            [0, 0, -1 if self.flipped[2] else 1, 0],
            [0, 0, 0, 1]
        ])
        if self.is_skewed:
            result = result @ np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, (-1 if self.inverted else 1) * self.spacing[1] / math.sqrt(2) / (self.spacing[2] * math.sqrt(2)), 1,
                 0],
                [0, 0, 0, 1]
            ])
        return result

    def grid_to_world_2d(self, red_axis):
        g2w = self.grid_to_world
        axes = np.ones((4,), dtype=np.bool)
        axes[red_axis] = False
        return g2w[axes][:, axes]

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
               is_skewed: bool = None, world_transform: np.ndarray = None, flipped: Tuple[bool, bool, bool] = None):
        # TODO: Handle this more nicely
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
        if flipped is not None:
            self.flipped = flipped

        return self

    def crop_view(self, crop: Union[float, Tuple[float, float, float]], center_crop: bool = True):
        if type(crop) == float or type(crop) == int:
            if crop > 0.99999:
                return self
            icropx = 1 - crop
            icropy = 1 - crop
            icropz = 1 - crop
        else:
            icropx = 1 - crop[0]
            icropy = 1 - crop[1]
            icropz = 1 - crop[2]

        w, h, l = self.shape

        if center_crop:
            view = self.data[int(w / 2 * icropx):int(-w / 2 * icropx),
                   int(h / 2 * icropy):int(-h / 2 * icropy),
                   int(l / 2 * icropz):int(-l / 2 * icropz)]
        else:
            view = self.data[:int(w * (1 - icropx)), :int(h * (1 - icropy)), :int(l * (1 - icropz))]

        # FIXME: Do something clever
        result = Volume(view, self.inverted, self.spacing, is_skewed=self.is_skewed, flipped=self.flipped)
        result.psf = self.psf
        return result


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


def unshift_fast(vol: Volume, invert: bool = False, estimate_true_interval: bool = True, rotate: bool = True) -> Volume:
    if estimate_true_interval:
        interval = compute_true_interval(vol, invert)
        vol.spacing[2] = interval
        logger.debug('Estimated volume interval: {}'.format(interval))

    if invert:
        data = unshift_fast_numbai(vol.data, vol.spacing)
        if rotate:
            data = np.rot90(data, k=2, axes=(1, 2))
        return vol.update(data=data, inverted=False, spacing=(vol.spacing[0], vol.spacing[1] / np.sqrt(2),
                                                              vol.spacing[2] * np.sqrt(2)), is_skewed=False)
    else:
        return vol.update(data=unshift_fast_numba(vol.data, vol.spacing),
                          spacing=(vol.spacing[0], vol.spacing[1] / np.sqrt(2),
                                   vol.spacing[2] * np.sqrt(2)), is_skewed=False)


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


def unshift_fast_diag(vol: Volume, invert: bool = False, rotate: bool = True) -> Volume:
    # FIXME: The resolution is incorrect!!!!! fix it!!!!
    # interval = compute_true_interval(vol, invert)
    # vol.spacing[2] = interval
    if invert:
        data = unshift_fast_numbai_diag(vol.data, vol.spacing)
        if rotate:
            data = np.rot90(data, k=1, axes=(1, 2))
        return vol.update(data=data, is_skewed=False, inverted=False)
    else:
        return vol.update(data=unshift_fast_numba_diag(vol.data, vol.spacing))


@jit(nopython=True, parallel=True)
def unshift_fast_numbai_diag(data: np.ndarray, resolution: np.ndarray) -> np.ndarray:
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
def unshift_fast_numba_diag(data: np.ndarray, resolution: np.ndarray) -> np.ndarray:
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
    from dipy.align.metrics import CCMetric

    metric = CCMetric(3)
    level_iters = [8, 8, 4]
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)

    mapping = sdr.optimize(vol_a.data, vol_b.data, vol_a.grid_to_world, vol_b.grid_to_world, None)

    transformed_b = mapping.transform(vol_b.data)

    return vol_a, vol_b.update(data=transformed_b)


def register_2d(vol_a: Volume, vol_b: Volume, axis=2):
    from dipy.align.imaffine import (transform_centers_of_mass,
                                     MutualInformationMetric,
                                     AffineRegistration,
                                     )
    from dipy.align.transforms import RigidTransform2D
    vol_a_flat = np.mean(vol_a.data, axis=axis)
    vol_b_flat = np.mean(vol_b.data, axis=axis)

    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)

    level_iters = [500000, 50000, 3000, 2000]
    sigmas = [7.0, 5.0, 2.0, 0.0]
    factors = [8, 4, 2, 1]

    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    transform = RigidTransform2D()
    params0 = None
    print(vol_a.grid_to_world_2d(axis), vol_b.grid_to_world_2d(axis))
    init = transform_centers_of_mass(vol_a_flat, vol_a.grid_to_world_2d(axis), vol_b_flat, vol_b.grid_to_world_2d(axis))

    starting_affine = init.affine

    affine = affreg.optimize(vol_a_flat, vol_b_flat, transform, params0,
                             vol_a.grid_to_world_2d(axis), vol_b.grid_to_world_2d(axis),
                             starting_affine=starting_affine)

    b_trans = affine.affine
    b_trans = np.insert(b_trans, axis, np.zeros((3,)), 0)
    b_trans = np.insert(b_trans, axis, np.zeros((4,)), 1)
    b_trans[axis, axis] = 1

    return vol_a, vol_b.update(world_transform=b_trans)


def register_dipy(vol_a: Volume, vol_b: Volume,
                  sampling_prop: float = 1.0, crop: float = 0.8):
    from dipy.align.imaffine import (transform_centers_of_mass,
                                     MutualInformationMetric,
                                     AffineRegistration,
                                     AffineMap,
                                     )
    from dipy.align.transforms import (TranslationTransform3D,
                                       RigidTransform3D,
                                       AffineTransform3D)

    print(sampling_prop)

    # FIXME: Cropping doesn't work!!!!
    subvol_a = vol_a.crop_view(crop, center_crop=False)
    subvol_b = vol_b.crop_view(crop, center_crop=False)

    logger.debug('Sub-volume A size: ' + str(subvol_a.shape))
    logger.debug('Sub-volume B size: ' + str(subvol_b.shape))

    level_iters = [20000, 10000, 5500, 800]

    sigmas = [12.0, 3.0, 1.0, 0.0]

    factors = [16, 4, 2, 1]

    affreg = AffineRegistration(metric=MutualInformationMetric(32, sampling_prop, sampling_type='grid'),
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    transform = TranslationTransform3D()
    params0 = None
    starting_affine = vol_b.world_transform
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

    # affreg = AffineRegistration(metric=MutualInformationMetric(32, sampling_prop, sampling_type='grid'),
    #                             level_iters=[2000, 1000],
    #                             sigmas=[1.0, 0.0],
    #                             factors=[2, 1])
    #
    # transform = AffineTransform3D()
    # params0 = None
    # starting_affine = rigid.affine
    #
    # affine = affreg.optimize(subvol_a.data, subvol_b.data, transform, params0,
    #                          subvol_a.grid_to_world, subvol_b.grid_to_world,
    #                          starting_affine=starting_affine)
    #
    # logger.debug('Registration affine: ' + str(affine))

    vol_b.world_transform = np.array(rigid.affine)

    return vol_a, vol_b


def make_isotropic(vol_a: Volume, vol_b: Volume):
    import scipy.interpolate
    min_res = min(np.min(vol_a.spacing), np.min(vol_b.spacing))
    # zoomed_volA = scipy.interpolate.interp1d(ax, vol_a.data, axis=2)(np.linspace(0, len(ax)-1, len(ax)*(vol_a.resolution[2]/vol_b.resolution[2]))).astype(np.uint16)
    zoomed_volA = scipy.ndimage.zoom(vol_a.data, vol_a.spacing / min_res, order=1)
    # zoomed_volB = scipy.interpolate.interp1d(bx, vol_b.data, axis=1)(np.linspace(0, len(bx)-1, len(bx)*(vol_b.resolution[1]/vol_a.resolution[1]))).astype(np.uint16)
    zoomed_volB = scipy.ndimage.zoom(vol_b.data, vol_b.spacing / min_res, order=1)

    result_vol_a = vol_a.update(zoomed_volA,
                                spacing=(min_res,) * 3)
    result_vol_b = vol_b.update(zoomed_volB,
                                spacing=(min_res,) * 3)

    return result_vol_a, result_vol_b


def fuse(vol_a: Volume, vol_b: Volume):
    # FIXME: Do something clever with is_skewed
    return Volume(np.floor_divide(vol_a.data + vol_b.data, 2).astype(np.uint16), False, vol_a.spacing, is_skewed=False)


def deconvolve_rl(vol: Volume, n: int, blur: BlurFunction) -> Volume:
    # TODO: Possibly switch this to use the skimage function (blur function must then be converted to PSF first)
    estimate = vol.data
    with progressbar.ProgressBar(max_value=n, redirect_stderr=True) as bar:
        for _ in bar(range(n)):
            estimate = estimate * blur(vol.data / (blur(estimate) + 1E-9))

    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)

    est_min = np.min(estimate)
    est_max = np.max(estimate)

    return Volume(np.clip((estimate + est_min) / (est_max - est_min), 0.0, 1.0), False, vol.spacing, is_skewed=False)


def deconvolve(vol_a: Volume, vol_b: Volume, n: int, blurA: BlurFunction, blurB: BlurFunction) -> Volume:
    view_a, view_b = vol_a.data, vol_b.data

    estimate = (view_a + view_b) / 2
    with progressbar.ProgressBar(max_value=n, redirect_stderr=True) as bar:
        for _ in bar(range(n)):
            estimate_A = estimate * blurA(view_a / (blurA(estimate) + 1E-9))
            estimate = estimate_A * blurB(view_b / (blurB(estimate_A) + 1E-9))

    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)

    return Volume(estimate, False, vol_a.spacing, is_skewed=False)


def density_to_multiview_data(density, psf_A, psf_B, out=None):
    from scipy.signal import fftconvolve
    """
    Takes a 2D image input, returns a stack of multiview data
    """
    if out is None:
        multiview_data = np.zeros(
            (2,) + density.shape, dtype=np.uint16)
    else:
        multiview_data = out

    """
    Simulate the imaging process by applying multiple blurs
    """
    multiview_data[0, :, :, :] = fftconvolve(density, psf_A, 'same')
    multiview_data[1, :, :, :] = fftconvolve(density, psf_B, 'same')
    return multiview_data


def multiview_data_to_density(multiview_data, psf_A, psf_B, out=None):
    from scipy.signal import fftconvolve
    """
    The transpose of the density_to_multiview_data operation we perform above.
    """
    if out is None:
        density = np.zeros(multiview_data.shape[2:], dtype=np.uint16)
    else:
        density = out
        density.fill(0)

    np.add(density, fftconvolve(multiview_data[0, :, :, :], psf_A, 'same') / 2, out=density, casting='unsafe')
    np.add(density, fftconvolve(multiview_data[1, :, :, :], psf_B, 'same') / 2, out=density, casting='unsafe')
    return density


def fft_centered(arr, newshape):
    # Return the center newshape portion of the array.
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


import arrayfire as af


def rfft_af(x, s=None, axes=None):
    if s is None:
        s = x.shape

    x_d = af.from_ndarray(x)
    r_d = af.fft3_r2c(x_d, *s)
    del x_d
    r = r_d.to_ndarray()
    del r_d
    return r


def irfft_af(x, s, axes=None):
    x_d = af.from_ndarray(x)
    r_d = af.fft3_c2r(x_d)
    del x_d
    r = r_d.to_ndarray()
    del r_d
    return r


def deconvolve_psf(vol_a: Volume, vol_b: Volume, n: int, psf_A, psf_B) -> Volume:
    # from astropy.convolution import convolve_fft
    from functools import partial
    from scipy.signal import fftconvolve
    # FIXME: "Most (all?) FFT packages only work well (performance-wise) with sizes that do not have any large prime
    # FIXME: factors. Rounding the signal and kernel size up to the next power of two is a common practice that may
    # FIXME; result in a (very) significant speed-up." (
    # FIXME: https://stackoverflow.com/questions/18384054/what-are-the-downsides-of
    # FIXME: -convolution-by-fft-compared-to-realspace-convolution)
    view_a, view_b = vol_a.data.astype(np.float), vol_b.data.astype(np.float)

    psf_A = psf_A.astype(np.float) / np.sum(psf_A).astype(np.float)
    psf_B = psf_B.astype(np.float) / np.sum(psf_B).astype(np.float)
    psf_Ai = psf_A[::-1, ::-1, ::-1]
    psf_Bi = psf_B[::-1, ::-1, ::-1]

    np.fft.rfftn = rfft_af
    np.fft.irfftn = irfft_af

    estimate = (view_a + view_b) / 2

    # convolve = partial(convolve_fft,
    #                    fftn=partial(pyfftw.interfaces.numpy_fft.rfftn, threads=40),
    #                    ifftn=partial(pyfftw.interfaces.numpy_fft.irfftn, threads=40),
    #                    allow_huge=True)
    convolve = partial(fftconvolve, mode='same')

    with progressbar.ProgressBar(max_value=n, redirect_stderr=True) as bar:
        for _ in bar(range(n)):
            estimate = estimate * convolve(view_a / (convolve(estimate, psf_A) + 1e-6), psf_Ai)
            estimate = estimate * convolve(view_b / (convolve(estimate, psf_B) + 1e-6), psf_Bi)

    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)

    # TODO: Rescaling might be unwanted
    e_min, e_max = np.percentile(estimate, [0.001, 99.999])
    estimate = ((np.clip(estimate, e_min, e_max) - e_min) / (e_max - e_min) * (2 ** 16 - 1)).astype(np.uint16)

    # estimate = np.clip(estimate, 0, 2**16-1).astype(np.uint16)

    return Volume(estimate, False, vol_a.spacing, is_skewed=False)


def deconvolve_matlab(vol_a: Volume, vol_b: Volume, n: int, psf_A, psf_B) -> Volume:
    import matlab.engine
    view_a, view_b = vol_a.data.astype(np.float), vol_b.data.astype(np.float)

    psf_A = psf_A.astype(np.float) / np.sum(psf_A).astype(np.float)
    psf_B = psf_B.astype(np.float) / np.sum(psf_B).astype(np.float)

    eng = matlab.engine.start_matlab()

    estimate = eng.jrl(matlab.double(view_a.tolist()), matlab.double(view_b.tolist()), matlab.double(psf_A.tolist()),
                       matlab.double(psf_B.tolist()), n)
    estimate = estimate - np.zeros_like(view_a)

    return Volume(estimate, False, vol_a.spacing, is_skewed=False)


# def deconvolve_psf(vol_a: Volume, vol_b: Volume, n: int, psf_A, psf_B) -> Volume:
#     from scipy.signal import fftconvolve
#     from scipy import fftpack
#     # FIXME: "Most (all?) FFT packages only work well (performance-wise) with sizes that do not have any large prime
#     # FIXME: factors. Rounding the signal and kernel size up to the next power of two is a common practice that may
#     # FIXME; result in a (very) significant speed-up." (
#     # FIXME: https://stackoverflow.com/questions/18384054/what-are-the-downsides-of
#     # FIXME: -convolution-by-fft-compared-to-realspace-convolution)
#     view_a, view_b = vol_a.data.astype(np.float), vol_b.data.astype(np.float)
#
#     psf_A = psf_A.astype(np.float) / np.sum(psf_A).astype(np.float)
#     psf_B = psf_B.astype(np.float) / np.sum(psf_B).astype(np.float)
#     psf_Ai = psf_A[::-1, ::-1, ::-1]
#     psf_Bi = psf_B[::-1, ::-1, ::-1]
#
#     estimate = (view_a + view_b) / 2
#
#     s1 = np.array(view_a.shape)
#     s2 = np.array(psf_A.shape)
#
#     shape = s1 + s2 - 1
#
#     fshape = [fftpack.helper.next_fast_len(int(d)) for d in shape]
#     fslice = tuple([slice(0, int(sz)) for sz in shape])
#     psf_A_fft = np.fft.rfftn(psf_A, fshape)
#     psf_B_fft = np.fft.rfftn(psf_B, fshape)
#     psf_Ai_fft = np.fft.rfftn(psf_Ai, fshape)
#     psf_Bi_fft = np.fft.rfftn(psf_Bi, fshape)
#
#     if debug:
#         last = estimate
#         debug_f = open('convolve_debug.txt', 'w')
#
#     with progressbar.ProgressBar(max_value=n, redirect_stderr=True) as bar:
#         for _ in bar(range(n)):
#             # TODO: Maybe fix the dtype of estimate during the deconvolution to save memory
#             # print((view_a / (fftconvolve(estimate, psf_A, mode='same') + 1e-7)).min())
#             # print((view_a / (fftconvolve(estimate, psf_A, mode='same') + 1e-7)).max())
#             # amax = (fftconvolve(estimate, psf_A, mode='same') + 1e-7).argmin()
#             # print(estimate.flatten()[amax])
#             # print(estimate.flatten()[amax]/(fftconvolve(estimate, psf_A, mode='same') + 1e-7).min())
#             # print(((fftconvolve(estimate, psf_A, mode='same') + 1e-7)).min())
#             # print(((fftconvolve(estimate, psf_A, mode='same') + 1e-7)).max())
#             # print(estimate.max())
#             # print(estimate.min())
#             # estimate = estimate * fftconvolve(view_a / (fftconvolve(estimate, psf_A, mode='same') + 1e-7), psf_Ai,
#             #                                   mode='same')
#             #
#             # estimate = estimate * fftconvolve(view_b / (fftconvolve(estimate, psf_B, mode='same') + 1e-7), psf_Bi,
#             #                                   mode='same')
#
#             blurA = view_a / (fft_centered(np.fft.irfftn(np.fft.rfftn(estimate, fshape) * psf_A_fft, fshape)[fslice],
#                                            s1) + 1e-4)
#             estimate = estimate * fft_centered(np.fft.irfftn(np.fft.rfftn(blurA, fshape) * psf_Ai_fft, fshape)[fslice],
#                                                s1)
#
#             blurB = view_b / (fft_centered(np.fft.irfftn(np.fft.rfftn(estimate, fshape) * psf_B_fft, fshape)[fslice],
#                                            s1) + 1e-4)
#             estimate = estimate * fft_centered(np.fft.irfftn(np.fft.rfftn(blurB, fshape) * psf_Bi_fft, fshape)[fslice],
#                                                s1)
#
#             # print(np.percentile(estimate, (99, 99.5, 99.8, 99.9)))
#             # print(np.sum(np.abs(estimate-estimate_A)))
#             if debug:
#                 diff = np.mean(np.abs(estimate.astype(np.float) - last.astype(np.float)))
#                 debug_f.write(str(diff) + '\n')
#                 last = estimate
#             # density_to_multiview_data(estimate, psf_A, psf_B, out=expected_data)
#             # #     multiview_data_to_visualization(expected_data, outfile='expected_data.tif')
#             # "Done constructing."
#             # """
#             # Take the ratio between the measured data and the expected data.
#             # Store this ratio in 'expected_data'
#             # """
#             # expected_data += 1e-6  # Don't want to divide by 0!
#             # np.divide(mvd, expected_data, out=expected_data)
#             # """
#             # Apply the transpose of the expected data operation to the correction factor
#             # """
#             # multiview_data_to_density(expected_data, psf_A, psf_B, out=correction_factor)
#             # """
#             # Multiply the old estimate by the correction factor to get the new estimate
#             # """
#             # np.multiply(estimate, correction_factor, out=estimate)
#
#     CURSOR_UP_ONE = '\x1b[1A'
#     ERASE_LINE = '\x1b[2K'
#     print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)
#
#     if debug:
#         debug_f.close()
#
#     # TODO: Rescaling might be unwanted
#     e_min, e_max = np.percentile(estimate, [0.001, 99.999])
#     estimate = ((np.clip(estimate, e_min, e_max) - e_min) / (e_max - e_min) * (2 ** 16 - 1)).astype(np.uint16)
#
#     # estimate = np.clip(estimate, 0, 2**16-1).astype(np.uint16)
#
#     return Volume(estimate, False, vol_a.spacing, is_skewed=False)


def deconvolve_fast(vol_a, vol_b, n, psf_A, psf_B):
    r = deconvolve_fast_numba(vol_a.data.astype(np.float), vol_b.data.astype(np.float), n, psf_A.astype(np.float),
                              psf_B.astype(np.float))

    return Volume(r, False, vol_a.spacing, is_skewed=False)


from numpy.fft import fftn, ifftn


@jit(parallel=True)
def deconvolve_fast_numba(view_a: np.ndarray, view_b: np.ndarray, n: int, psf_A, psf_B) -> np.ndarray:
    psf_A = psf_A / np.sum(psf_A)
    psf_B = psf_B / np.sum(psf_B)
    psf_Ai = psf_A[::-1, ::-1, ::-1]
    psf_Bi = psf_B[::-1, ::-1, ::-1]

    psf_A_fft = fftn(psf_A, view_a.shape)
    psf_B_fft = fftn(psf_B, view_a.shape)
    psf_Ai_fft = fftn(psf_Ai, view_a.shape)
    psf_Bi_fft = fftn(psf_Bi, view_a.shape)

    estimate = (view_a + view_b) / 2

    for _ in (range(n)):
        estimate = estimate * ifftn(fftn(view_a / (ifftn(fftn(estimate) * psf_A_fft) + 1e-7)) * psf_Ai_fft)
        estimate = estimate * ifftn(fftn(view_b / (ifftn(fftn(estimate) * psf_B_fft) + 1e-7)) * psf_Bi_fft)

        print(_)

    # TODO: Rescaling might be unwanted
    e_min, e_max = np.percentile(estimate, [0.001, 99.999])
    estimate = ((np.clip(estimate, e_min, e_max) - e_min) / (e_max - e_min) * (2 ** 16 - 1)).astype(np.uint16)

    return estimate


def gpu_fftconvolve(x, y):
    import reikna.cluda as cluda
    from reikna.fft import FFT

    api = cluda.any_api()
    thr = api.Thread.create(device_filters={'include_devices': 'amd'})

    x_dev = thr.to_device(x.astype(np.complex))
    y_dev = thr.to_device(y.astype(np.complex))

    fft = FFT(x_dev)
    fftc = fft.compile(thr)

    x_fft_dev = thr.empty_like(x_dev)
    y_fft_dev = thr.empty_like(y_dev)

    fftc(x_fft_dev, x_dev)
    fftc(y_fft_dev, y_dev)

    result_fft = x_fft_dev * y_fft_dev

    result = thr.empty_like(result_fft)
    fftc(result, result_fft, inverse=True)

    result_cpu = thr.from_device(result)

    thr.release()

    return result_cpu


def deconvolve_cluda(vol_a: Volume, vol_b: Volume, n: int, psf_A, psf_B) -> Volume:
    import reikna.cluda as cluda
    from reikna.fft import FFT
    view_a, view_b = vol_a.data.astype(np.float), vol_b.data.astype(np.float)

    psf_A = psf_A.astype(np.float) / np.sum(psf_A).astype(np.float)
    psf_B = psf_B.astype(np.float) / np.sum(psf_B).astype(np.float)
    psf_Ai = psf_A[::-1, ::-1, ::-1]
    psf_Bi = psf_B[::-1, ::-1, ::-1]

    print(np.sum(psf_A))
    print(np.sum(psf_B))

    print(psf_A.dtype)

    mvd = np.zeros((2,) + vol_a.shape)
    mvd[0, :, :, :] = vol_a.data
    mvd[1, :, :, :] = vol_b.data

    api = cluda.any_api()
    thr = api.Thread.create(interactive=True)
    a_dev = thr.to_device(np.ascontiguousarray(view_a.astype(np.complex64)))
    b_dev = thr.to_device(np.ascontiguousarray(view_b.astype(np.complex64)))

    psfa_dev = thr.to_device(np.fft.fftn(psf_A, a_dev.shape))
    psfb_dev = thr.to_device(np.fft.fftn(psf_B, a_dev.shape))
    psfai_dev = thr.to_device(np.fft.fftn(psf_Ai, a_dev.shape))
    psfbi_dev = thr.to_device(np.fft.fftn(psf_Bi, a_dev.shape))

    estimate = (a_dev + b_dev) / 2

    compfft = FFT(estimate)
    compfftc = compfft.compile(thr)

    temp = thr.empty_like(estimate)

    if debug:
        last = estimate
        debug_f = open('convolve_debug.txt', 'w')

    with progressbar.ProgressBar(max_value=n, redirect_stderr=True) as bar:
        for _ in bar(range(n)):
            # temp = TemporaryManager.array(estimate.shape. estimate.dype)
            compfftc(temp, estimate)
            temp = temp * psfa_dev
            compfftc(temp, temp, inverse=True)
            temp = a_dev / temp
            compfftc(temp, temp)
            temp = temp * psfai_dev
            compfftc(temp, temp, inverse=True)
            estimate = estimate * temp

            compfftc(temp, estimate)
            temp = temp * psfb_dev
            compfftc(temp, temp, inverse=True)
            temp = b_dev / temp
            compfftc(temp, temp)
            temp = temp * psfbi_dev
            compfftc(temp, temp, inverse=True)
            estimate = estimate * temp
            # estimate = estimate * compfftc(temp,
            #                                compfftc(temp, b_dev / compfftc(temp, compfftc(temp, estimate) * psfb_dev,
            #                                                                inverse=True)) * psfbi_dev, inverse=True)
            # estimate = estimate * compfftc(view_a / (fftconvolve(estimate, psf_A, mode='same') + 1e-7), psf_Ai, mode='same')

            # estimate = estimate * compfftc(view_b / (compfftc(estimate, psf_B, mode='same') + 1e-7), psf_Bi, mode='same')

            if debug:
                diff = np.mean(np.abs(thr.to_device(estimate).astype(np.float) - last.astype(np.float)))
                debug_f.write(str(diff) + '\n')
                last = estimate

    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)

    if debug:
        debug_f.close()

    # TODO: Rescaling might be unwanted
    estimate = thr.from_device(estimate).astype(np.float)
    e_min, e_max = np.percentile(estimate, [0.001, 99.999])
    estimate = ((np.clip(estimate, e_min, e_max) - e_min) / (e_max - e_min) * (2 ** 16 - 1)).astype(np.uint16)

    # estimate = np.clip(estimate, 0, 2**16-1).astype(np.uint16)

    return Volume(estimate, False, vol_a.spacing, is_skewed=False)


def extract_psf(vol: Volume, min_size: int = 50, max_size: int = 140, crop: float = 1.0, psf_half_width: int = 7):
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops
    data = vol.data
    thr = threshold_otsu(data)
    data_bin = data > thr

    points = [np.array(r.centroid, dtype=np.int) for r in regionprops(label(data_bin))
              if min_size <= r.area <= max_size]

    blob_images = []
    for point in points:
        blob_images.append(extract_3d(data, point, psf_half_width))

    median_blob = np.median(blob_images, axis=0)
    print(median_blob.mean())
    print(median_blob.max())
    print(np.median(median_blob))
    print(len(points))

    try:
        import tifffile
        tifffile.imsave('psf' + str(np.random.rand()), median_blob)
    except PermissionError:
        pass

    return median_blob


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

    # TODO: Make this different
    from hashlib import sha1

    for i in range(len(datas)):
        # TODO: Handle loading n volumes properly or don't handle them at all
        volumes.append(Volume(datas[i].swapaxes(0, 2).swapaxes(0, 1), bool(i), spacing, is_skewed=is_skewed))
        logger.debug(volumes[-1].data.mean())
        logger.debug(hash(volumes[-1].data.tostring()))
        gc.collect()

    return tuple(volumes)


def deconvolve_diag(vol_a: Volume, vol_b: Volume, n: int, sigma_z: float, sigma_xy: float) -> Volume:
    from scipy.signal import fftconvolve
    # FIXME?: "Most (all?) FFT packages only work well (performance-wise) with sizes that do not have any large prime
    # FIXME?: factors. Rounding the signal and kernel size up to the next power of two is a common practice that may result
    # FIXME?: in a (very) significant speed-up." (https://stackoverflow.com/questions/18384054/what-are-the-downsides-of
    # FIXME?: -convolution-by-fft-compared-to-realspace-convolution)
    view_a, view_b = vol_a.data, vol_b.data
    # view_a = np.clip(
    #     scipy.ndimage.zoom(view_a, (2, 2, 2)), 0.0, 1.0)
    # view_b = np.clip(
    #     scipy.ndimage.zoom(view_b, (2, 2, 2)), 0.0, 1.0)

    psf_A = np.zeros((81, 81, 81))
    psf_A[40, 40, 40] = 1
    psf_A = scipy.ndimage.gaussian_filter(psf_A, (sigma_xy, sigma_xy, sigma_z))
    psf_A = scipy.ndimage.rotate(psf_A, -45, axes=(1, 2))
    psf_A = psf_A[20:-20, 20:-20, 20:-20]

    psf_B = np.zeros((80, 80, 80))
    psf_B[40, 40, 40] = 1
    psf_B = scipy.ndimage.gaussian_filter(psf_B, (sigma_xy, sigma_xy, sigma_z))
    psf_B = scipy.ndimage.rotate(psf_B, 45, axes=(1, 2))
    psf_B = psf_B[20:-20, 20:-20, 20:-20]

    psf_Ai = psf_A[::-1, ::-1, ::-1]
    psf_Bi = psf_B[::-1, ::-1, ::-1]

    estimate = (view_a + view_b) / 2
    with progressbar.ProgressBar(max_value=n, redirect_stderr=True) as bar:
        for _ in bar(range(n)):
            # TODO: Maybe fix the dtype of estimate during the deconvolution to save memory
            estimate_A = estimate * fftconvolve(view_a / (fftconvolve(estimate, psf_A, mode='same') + 1E-9), psf_Ai,
                                                mode='same')
            estimate = estimate_A * fftconvolve(view_b / (fftconvolve(estimate_A, psf_B, mode='same') + 1E-9), psf_Bi,
                                                mode='same')

    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)

    estimate = np.clip(estimate, 3, 2 ** 16 - 3).astype(np.uint16)

    return Volume(estimate, False, vol_a.spacing)


def extract_3d(data, center, half_size):
    imax = np.clip(center + half_size + 1, 0, data.shape).astype(np.int)
    imin = np.clip(center - half_size, 0, data.shape).astype(np.int)

    subvol = data[imin[0]:imax[0], imin[1]:imax[1], imin[2]:imax[2]]

    max_missing = ((center + half_size + 1) - imax).astype(np.int)
    min_missing = (imin - (center - half_size)).astype(np.int)

    return np.pad(subvol, [(min_missing[i], max_missing[i]) for i in range(3)], mode='constant')
