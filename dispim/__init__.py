#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import shutil
import warnings
from typing import Tuple, Union, Callable

import math
import matplotlib.pyplot as plt
import numpy as np
import progressbar
import scipy
from dipy.align.transforms import TranslationTransform3D, TranslationTransform2D
from numba import jit, prange
from scipy import ndimage

from dispim import metrack

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', '.*output shape of zoom.*')

BlurFunction = Callable[[np.ndarray], np.ndarray]

debug = False


class Volume(np.ndarray):
    def __new__(cls, input_array, inverted: bool = None,
                spacing: Union[Tuple[float, float, float], np.ndarray] = None,
                is_skewed: bool = None, flipped: Tuple[bool] = None,
                world_transform: np.ndarray = None, psf: np.ndarray = None):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__Ø
        obj = np.asarray(input_array).view(cls)
        # set the new 'info' attribute to the value passed
        obj.inverted = inverted if inverted is not None else getattr(input_array, 'inverted', False)
        obj.spacing = spacing if spacing is not None else getattr(input_array, 'spacing', (1, 1, 1))
        obj.is_skewed = is_skewed if is_skewed is not None else getattr(input_array, 'is_skewed', False)
        obj.flipped = flipped if flipped is not None else getattr(input_array, 'flipped', (False, False, False))
        obj.world_transform = world_transform if world_transform is not None else getattr(input_array,
                                                                                          'world_transform', np.eye(4))
        obj.psf = psf if psf is not None else getattr(input_array, 'psf', None)

        obj.flags.writeable = False
        obj.initialized = True
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(InfoArray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. InfoArray():
        #    obj is None
        #    (we're in the middle of the InfoArray.__new__
        #    constructor, and self.info will be set when we return to
        #    InfoArray.__new__)
        if obj is None: return
        # From view casting - e.g arr.view(InfoArray):
        #    obj is arr
        #    (type(obj) can be InfoArray)
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is InfoArray
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'info', because this
        # method sees all creation of default objects - with the
        # InfoArray.__new__ constructor, but also with
        # arr.view(InfoArray).
        self.inverted = getattr(obj, 'inverted', False)
        self.spacing = getattr(obj, 'spacing', (1, 1, 1))
        self.is_skewed = getattr(obj, 'is_skewed', False)
        self.flipped = getattr(obj, 'flipped', (False, False, False))
        self.world_transform = getattr(obj, 'world_transform', np.eye(4))
        self.psf = getattr(obj, 'psf', None)
        # We do not need to return anything

    @property
    def grid_to_world(self) -> np.ndarray:
        result = np.array([
            [self.spacing[0], 0, 0, 0],
            [0, self.spacing[1] / (np.sqrt(2) if self.is_skewed else 1), 0, 0],
            [0, 0, self.spacing[2] * (np.sqrt(2) if self.is_skewed else 1), 0],
            [0, 0, 0, 1]
        ])
        result = result @ np.array([
            [-1 if self.flipped[0] else 1, 0, 0, self.shape[0] if self.flipped[0] else 0],
            [0, -1 if self.flipped[1] else 1, 0, self.shape[1] if self.flipped[1] else 0],
            [0, 0, -1 if self.flipped[2] else 1, self.shape[2] if self.flipped[2] else 0],
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

    def grid_to_world_2d(self, red_axis: int) -> np.ndarray:
        g2w = self.grid_to_world
        axes = np.ones((4,), dtype=np.bool)
        axes[red_axis] = False
        return g2w[axes][:, axes]

    def __setattr__(self, name, value):
        if hasattr(self, 'initialized'):
            """"""
            msg = "'%s' has no attribute %s" % (self.__class__,
                                                name)
            raise AttributeError(msg)
        else:
            np.ndarray.__setattr__(self, name, value)


def save_dual_tiff(name: str, vol_a: Volume, vol_b: Volume, path: str = 'out'):
    from tifffile import imsave
    if not os.path.exists(path):
        os.makedirs(path)
    if os.path.exists(path + '/' + name):
        shutil.rmtree(path + '/' + name)
    os.makedirs(path + '/' + name)
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
        return Volume(data, inverted=False, spacing=(vol.spacing[0], vol.spacing[1] / np.sqrt(2),
                                                     vol.spacing[2] * np.sqrt(2)), is_skewed=False)
    else:
        return Volume(unshift_fast_numba(vol.data, vol.spacing),
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


def register_2d(vol_a: Volume, vol_b: Volume, axis: int = 2, transform_cls: type = TranslationTransform2D) -> Tuple[
    Volume, Volume]:
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

    level_iters = [1000000, 500000, 200000, 70000, 70000]
    sigmas = [7.0, 3.0, 2.0, 1.0, 0.0]
    factors = [4, 2, 2, 1, 1]

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

    return vol_a, vol_b


def register_com(vol_a: Volume, vol_b: Volume) -> Tuple[Volume, Volume]:
    from dipy.align.imaffine import transform_centers_of_mass

    affine = transform_centers_of_mass(vol_a, vol_a.grid_to_world, vol_b, vol_b.grid_to_world)

    vol_b.world_transform[:] = np.array(affine.affine)
    return vol_a, vol_b


def register_dipy(vol_a: Volume, vol_b: Volume,
                  sampling_prop: float = 1.0, crop: float = 0.8, transform_cls: type = TranslationTransform3D) -> Tuple[
    Volume, Volume]:
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

    level_iters = [10000, 1000, 500]

    sigmas = [3.0, 1.0, 0.0]

    factors = [2, 1, 1]

    if metrack.is_tracked(MUTUAL_INFORMATION_METRIC) or metrack.is_tracked(MUTUAL_INFORMATION_GRADIENT_METRIC):
        def callback(value: float, gradient: float):
            metrack.append_metric(MUTUAL_INFORMATION_METRIC, (None, value))
            metrack.append_metric(MUTUAL_INFORMATION_GRADIENT_METRIC, (None, gradient))
    else:
        callback = None

    affreg = AffineRegistration(metric=MutualInformationMetric(32, sampling_prop, sampling_type='random'),
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


def deconvolve(vol_a: Volume, vol_b: Volume, n: int, psf_a: np.ndarray, psf_b: np.ndarray) -> Volume:
    # from astropy.convolution import convolve_fft
    from functools import partial
    from scipy.signal import fftconvolve
    view_a, view_b = vol_a.astype(np.float), vol_b.astype(np.float)

    psf_a = psf_a.astype(np.float) / np.sum(psf_a).astype(np.float)
    psf_b = psf_b.astype(np.float) / np.sum(psf_b).astype(np.float)
    psf_Ai = psf_a[::-1, ::-1, ::-1]
    psf_Bi = psf_b[::-1, ::-1, ::-1]

    estimate = (view_a + view_b) / 2

    convolve = partial(fftconvolve, mode='same')

    with progressbar.ProgressBar(max_value=n, redirect_stderr=True) as bar:
        for _ in bar(range(n)):
            estimate = estimate * convolve(view_a / (convolve(estimate, psf_a) + 1e-6), psf_Ai)
            estimate = estimate * convolve(view_b / (convolve(estimate, psf_b) + 1e-6), psf_Bi)

    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)

    # TODO: Rescaling might be unwanted
    e_min, e_max = np.percentile(estimate, [0.05, 99.95])
    estimate = ((np.clip(estimate, e_min, e_max) - e_min) / (e_max - e_min) * (2 ** 16 - 1)).astype(np.uint16)

    return Volume(estimate, inverted=False, spacing=vol_a.spacing, is_skewed=False)


def deconvolve_gpu_chunked(vol_a: Volume, vol_b: Volume, n: int, psf_a: np.ndarray, psf_b: np.ndarray, nchunks: int,
                           blind: bool = False) -> Volume:
    import arrayfire as af
    result = np.zeros(vol_a.shape, np.float32)
    chunk_size = vol_a.shape[2] // nchunks
    for i in range(nchunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < nchunks - 1 else vol_a.shape[2]
        lpad = int(psf_a.shape[2] * 3.7) if i > 0 else 0
        rpad = int(psf_a.shape[2] * 3.7) if i < nchunks - 1 else 0
        with metrack.Context(f'Chunk {i}'):
            if not blind:
                chunk_est = deconvolve_gpu(Volume(vol_a[:, :, start - lpad:end + rpad], False, (1, 1, 1)),
                                           Volume(vol_b[:, :, start - lpad:end + rpad], False, (1, 1, 1)), n, psf_a,
                                           psf_b)
            else:
                chunk_est = deconvolve_gpu_blind(Volume(vol_a[:, :, start - lpad:end + rpad], False, (1, 1, 1)),
                                                 Volume(vol_b[:, :, start - lpad:end + rpad], False, (1, 1, 1)), n, 5,
                                                 psf_a, psf_b)

        af.device_gc()

        if rpad > 0:
            result[:, :, start:end] = chunk_est[:, :, lpad:-rpad]
        else:
            result[:, :, start:end] = chunk_est[:, :, lpad:]

    e_min, e_max = np.percentile(result, [0.002, 99.998])
    result = ((np.clip(result, e_min, e_max) - e_min) / (e_max - e_min) * (2 ** 16 - 1)).astype(np.uint16)

    return Volume(result, inverted=False, spacing=(1, 1, 1))


def deconvolve_gpu(vol_a: Volume, vol_b: Volume, n: int, psf_a: np.ndarray, psf_b: np.ndarray) -> Volume:
    from functools import partial
    from dispim.metrics import DECONV_MSE_DELTA
    import arrayfire as af

    view_a, view_b = vol_a.astype(np.float), vol_b.astype(np.float)

    psf_a = psf_a.astype(np.float) / np.sum(psf_a).astype(np.float)
    psf_b = psf_b.astype(np.float) / np.sum(psf_b).astype(np.float)
    psf_Ai = psf_a[::-1, ::-1, ::-1]
    psf_Bi = psf_b[::-1, ::-1, ::-1]

    view_a = af.cast(af.from_ndarray(view_a), af.Dtype.f32)
    view_b = af.cast(af.from_ndarray(view_b), af.Dtype.f32)

    psf_a = af.cast(af.from_ndarray(psf_a), af.Dtype.f32)
    psf_b = af.cast(af.from_ndarray(psf_b), af.Dtype.f32)
    psf_Ai = af.cast(af.from_ndarray(psf_Ai), af.Dtype.f32)
    psf_Bi = af.cast(af.from_ndarray(psf_Bi), af.Dtype.f32)

    estimate = (view_a + view_b) / 2

    convolve = partial(af.fft_convolve3)

    with progressbar.ProgressBar(max_value=n, redirect_stderr=True) as bar:
        for _ in bar(range(n)):
            if metrack.is_tracked(DECONV_MSE_DELTA):
                prev = estimate
            estimate = estimate * convolve(view_a / (convolve(estimate, psf_a) + 10), psf_Ai)
            estimate = estimate * convolve(view_b / (convolve(estimate, psf_b) + 10), psf_Bi)

            if metrack.is_tracked(DECONV_MSE_DELTA):
                metrack.append_metric(DECONV_MSE_DELTA, (_, float(np.mean((prev - estimate) ** 2))))

    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)

    return Volume(estimate.to_ndarray(), inverted=False, spacing=vol_a.spacing, is_skewed=False)


def deconvolve_gpu_blind(vol_a: Volume, vol_b: Volume, n: int, m: int, psf_a: np.ndarray, psf_b: np.ndarray) -> Volume:
    from functools import partial
    import arrayfire as af
    view_a, view_b = vol_a.astype(np.float), vol_b.astype(np.float)

    psf_a = psf_a.astype(np.float) / np.sum(psf_a).astype(np.float)
    psf_b = psf_b.astype(np.float) / np.sum(psf_b).astype(np.float)
    padding = tuple(
        (int(s // 2 - psf_a.shape[i]), int((s - s // 2) - psf_a.shape[i])) for i, s in enumerate(view_a.shape))
    print(psf_a.shape)
    print(psf_b.shape)
    psf_a = np.pad(psf_a,
                   tuple(((s - psf_a.shape[i]) // 2, (s - psf_a.shape[i]) - (s - psf_a.shape[i]) // 2) for i, s in
                         enumerate(view_a.shape)), 'constant')
    print(psf_b.shape)
    psf_b = np.pad(psf_b,
                   tuple(((s - psf_b.shape[i]) // 2, (s - psf_b.shape[i]) - (s - psf_b.shape[i]) // 2) for i, s in
                         enumerate(view_b.shape)), 'constant')
    print(psf_a.shape, view_a.shape)
    print(psf_b.shape, view_b.shape)
    # psf_Ai = psf_a[::-1, ::-1, ::-1]
    # psf_Bi = psf_b[::-1, ::-1, ::-1]

    view_a = af.cast(af.from_ndarray(view_a), af.Dtype.u16)
    view_b = af.cast(af.from_ndarray(view_b), af.Dtype.u16)

    psf_a = af.cast(af.from_ndarray(psf_a), af.Dtype.f32)
    psf_b = af.cast(af.from_ndarray(psf_b), af.Dtype.f32)
    # psf_Ai = af.cast(af.from_ndarray(psf_Ai), af.Dtype.f32)
    # psf_Bi = af.cast(af.from_ndarray(psf_Bi), af.Dtype.f32)

    estimate = (view_a + view_b) / 2

    convolve = partial(af.fft_convolve3)

    print('hey')

    lamb = 0.002

    with progressbar.ProgressBar(max_value=n, redirect_stderr=True) as bar:
        for _ in bar(range(n)):
            for j in range(m):
                psf_a = psf_a * convolve(view_a / (convolve(psf_a, estimate) + 1e-1), estimate[::-1, ::-1, ::-1])
            for j in range(m):
                estimate = estimate * convolve(view_a / (convolve(estimate, psf_a) + 10), psf_a[::-1, ::-1, ::-1])
            for j in range(m):
                psf_b = psf_b * convolve(view_b / (convolve(psf_b, estimate) + 1e-1), estimate[::-1, ::-1, ::-1])
            for j in range(m):
                estimate = estimate * convolve(view_b / (convolve(estimate, psf_b) + 10), psf_b[::-1, ::-1, ::-1])

    del psf_a, psf_b, view_a, view_b

    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)

    return Volume(estimate.to_ndarray(), inverted=False, spacing=vol_a.spacing, is_skewed=False)


def extract_psf(vol: Volume, min_size: int = 50, max_size: int = 140, psf_half_width: int = 7) -> np.ndarray:
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops
    from dispim.util import extract_3d
    from dispim.metrics import PSF_SIGMA_XY, PSF_SIGMA_Z
    data = vol
    thr = threshold_otsu(data)
    data_bin = data > thr

    points = [np.array(r.centroid, dtype=np.int) for r in regionprops(label(data_bin))
              if min_size <= r.area <= max_size]

    logger.debug(f'Found {len(points)} objects')

    # points = np.random.choice(points, size=min(len(points), 12000), replace=False)
    points = points[np.random.choice(len(points), min(len(points), 12000), replace=False), :]

    blob_images = []
    for point in points[min(10000, len(points) - 1)]:
        blob_images.append(extract_3d(data, point, psf_half_width))

        if metrack.is_tracked(PSF_SIGMA_XY) or metrack.is_tracked(PSF_SIGMA_Z):
            height, center_x, center_y, width_x, width_y, rotation = fitgaussian(blob_images[-1][psf_half_width, :, :])
            scale = vol.shape[0]
            if width_x > width_y:
                metrack.append_metric(PSF_SIGMA_Z, (None, width_x * scale))
                metrack.append_metric(PSF_SIGMA_XY, (None, width_y * scale))
            else:
                metrack.append_metric(PSF_SIGMA_Z, (None, width_y * scale))
                metrack.append_metric(PSF_SIGMA_XY, (None, width_x * scale))

    median_blob = np.median(blob_images, axis=0)

    return median_blob


def gaussian(height, center_x, center_y, width_x, width_y, rotation):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)

    rotation = np.deg2rad(rotation)
    center_x = center_x * np.cos(rotation) - center_y * np.sin(rotation)
    center_y = center_x * np.sin(rotation) + center_y * np.cos(rotation)

    def rotgauss(x, y):
        xp = x * np.cos(rotation) - y * np.sin(rotation)
        yp = x * np.sin(rotation) + y * np.cos(rotation)
        g = height * np.exp(
            -(((center_x - xp) / width_x) ** 2 +
              ((center_y - yp) / width_y) ** 2) / 2.)
        return g

    return rotgauss


def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X * data).sum() / total
    y = (Y * data).sum() / total
    col = data[:, int(y)]
    width_x = np.sqrt(abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
    height = data.max()
    return height, x, y, width_x, width_y, 0.0


def fitgaussian(data):
    import scipy.optimize
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)
    p, success = scipy.optimize.leastsq(errorfunction, params)
    return p


def apply_registration(vol_a: Volume, vol_b: Volume) -> Tuple[Volume, Volume]:
    from scipy.ndimage import affine_transform
    min_res = min(np.min(vol_a.spacing), np.min(vol_b.spacing))

    logger.debug(f"Min res: {min_res}")
    logger.debug(f"Grid-to-world A:\n{vol_a.grid_to_world}")
    logger.debug(f"Grid-to-world B:\n{vol_b.grid_to_world}")
    logger.debug(f"World transform:\n{vol_b.world_transform}")

    grid_to_world_final = np.eye(4) * np.array([min_res, min_res, min_res, 1])

    transform_a = np.linalg.inv(vol_a.grid_to_world)
    transform_a = transform_a @ grid_to_world_final

    logger.debug(f'Final A transform:\n{transform_a}')

    final_shape = (np.ceil(
        np.linalg.inv(transform_a) @ np.array([vol_a.shape[0], vol_a.shape[1], vol_a.shape[2], 1]))).astype(
        np.int)[:3]

    vol_a = Volume(affine_transform(vol_a, transform_a, output_shape=final_shape, order=2),
                   spacing=(min_res, min_res, min_res), is_skewed=False, inverted=False)

    transform_b = np.linalg.inv(vol_b.grid_to_world)
    transform_b = transform_b @ vol_b.world_transform
    transform_b = transform_b @ grid_to_world_final

    logger.debug(f'Final B transform:\n{transform_b}')

    transformed = affine_transform(vol_b, transform_b, output_shape=final_shape, order=2)

    logger.debug(f'Transformed A average value: {np.mean(vol_a)}')
    logger.debug(f'Transformed B average value: {np.mean(transformed)}')

    return vol_a, Volume(transformed, spacing=vol_a.spacing)
