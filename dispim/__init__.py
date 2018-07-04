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
from memory_profiler import profile
from numba import jit, prange
from scipy import ndimage
from scipy.ndimage.filters import gaussian_laplace, minimum_filter

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', '.*output shape of zoom.*')

BlurFunction = Callable[[np.ndarray], np.ndarray]


class Volume(object):
    def __init__(self, data: np.ndarray, resolution: Union[Tuple[float, float, float], np.ndarray]):
        self.data = data
        self.resolution = np.array(resolution) if isinstance(resolution, tuple) else resolution

    def get_center_of_mass(self):
        return ndimage.center_of_mass(self.data) * self.resolution

    def rot90(self):
        reorder_indices = np.array([0, 2, 1])
        new_data = np.swapaxes(self.data, 1, 2)[:, :, ::-1]
        return Volume(new_data, self.resolution[reorder_indices])

    def save_tiff_single(self, name: str, invert=False, swap_xy=False, dtype=np.float32, path='out'):
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
            # .swapaxes(0, 2).swapaxes(0, 1)
        data = np.swapaxes(data, 1, 0).swapaxes(2, 0)
        # print((np.unique(data.T.astype(dtype))))
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

    def save_nib(self, name: str):
        import nibabel as nib
        if not os.path.exists('out'):
            os.makedirs('out')
        if os.path.exists('out/' + name):
            shutil.rmtree('out/' + name)
        os.makedirs('out/' + name)
        data = self.data
        nib.save(nib.spatialimages.SpatialImage(np.clip(data, 0.0, 1.0), None), 'out/' + name + '/' + name + '.nii')


def save_dual_tiff(name: str, vol_a: Volume, vol_b: Volume, path: str = 'out'):
    from tifffile import imsave
    if not os.path.exists(path):
        os.makedirs(path)
    if os.path.exists(path + '/' + name):
        shutil.rmtree(path + '/' + name)
    os.makedirs(path + '/' + name)
    data = np.array([vol_a.data, vol_b.data, np.zeros(vol_a.data.shape)])

    for slc_index in range(data.shape[3]):
        imsave(path + '/' + name + '/' + name + str(slc_index) + '.tiff',
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


def unshift_fast(vol: Volume, invert: bool = False) -> Volume:
    # FIXME: The resolution is incorrect!!!!! fix it!!!!
    if invert:
        data = unshift_fast_numbai(vol.data, vol.resolution)
        data = np.rot90(data, k=2, axes=(1, 2))
        return Volume(data, (vol.resolution[0], vol.resolution[1] / np.sqrt(2),
                             vol.resolution[2] * np.sqrt(2)))
    else:
        return Volume(unshift_fast_numba(vol.data, vol.resolution), (vol.resolution[0], vol.resolution[1] / np.sqrt(2),
                                                                     vol.resolution[2] * np.sqrt(2)))


@jit(nopython=True, parallel=True)
def unshift_fast_numbai(data: np.ndarray, resolution: np.ndarray, invert: bool = False) -> np.ndarray:
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
def unshift_fast_numba(data: np.ndarray, resolution: np.ndarray, invert: bool = False) -> np.ndarray:
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
    if invert:
        return Volume(unshift_fast_numbai_diag(vol.data, vol.resolution), vol.resolution)
    else:
        return Volume(unshift_fast_numba_diag(vol.data, vol.resolution), vol.resolution)


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
    shift = vol.resolution[2] / vol.resolution[1]
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
    result_vol = Volume(result, vol.resolution)

    return result_vol


def get_com_image_shift(vol_a: Volume, vol_b: Volume):
    comA = vol_a.get_center_of_mass()
    comB = vol_b.get_center_of_mass()
    return comB - comA


def shift_to_mat(shift: np.ndarray):
    return np.array([
        [1, 0, 0, shift[0]],
        [0, 1, 0, shift[1]],
        [0, 0, 1, shift[2]],
        [0, 0, 0, 1]
    ])


def compute_basic_cost(vol_a: Volume, vol_b: Volume, shift: np.ndarray) -> float:
    mat = shift_to_mat(shift / vol_a.resolution[0])
    points_a = np.zeros((4, np.product(vol_a.data.shape)))
    # for i in range(1000):
    #     point_a = np.random.rand(3) * vol_a.data.shape
    #     points_a[:, i] = np.append(point_a, [1])

    i = 0
    n = 40
    xpoints = np.linspace(0, vol_a.data.shape[0], n)
    ypoints = np.linspace(0, vol_a.data.shape[1], n)
    zpoints = np.linspace(0, vol_a.data.shape[2], n)

    xv, yv, zv = np.meshgrid(xpoints, ypoints, zpoints)
    xv = xv.ravel()
    yv = yv.ravel()
    zv = zv.ravel()
    points_a = np.vstack((xv, yv, zv, np.ones(xv.shape[0])))

    points_b = mat @ points_a
    return float(np.sum(np.abs(scipy.ndimage.map_coordinates(vol_a.data, points_a[:3, :], order=0) -
                               scipy.ndimage.map_coordinates(vol_b.data, points_b[:3, :], order=0))))


def compute_shift_gradient(vol_a: Volume, vol_b: Volume, shift: np.ndarray,
                           cost: Callable[[Volume, Volume, np.ndarray], float] = compute_basic_cost,
                           h_size: float = 0.05):
    gradient = np.zeros((3,))
    f = partial(cost, vol_a, vol_b)
    for i in range(3):
        h = np.zeros((3,))
        h[i] = h_size
        gradient[i] = f(shift + h) - f(shift - h)
        gradient[i] /= 2 * h_size

    return gradient


def register_manual_translation(vol_a: Volume, vol_b: Volume) -> np.ndarray:
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(np.mean(vol_a.data, axis=2, dtype=np.float32).T / 2 ** 16)
    ax[1].imshow(np.mean(vol_b.data, axis=2, dtype=np.float32).T / 2 ** 16)

    # plt.show()
    points = plt.ginput(2)
    plt.close()

    return np.array([points[1][0] - points[0][0], points[1][1] - points[0][1], 0])


def register(vol_a: Volume, vol_b: Volume, n: int = 5, alpha: float = 0.015):
    shift = get_com_image_shift(vol_a, vol_b)
    print(compute_basic_cost(vol_a, vol_b, shift))
    for i in range(n):
        gradient = compute_shift_gradient(vol_a, vol_b, shift)
        shift -= alpha / (i + 1) * gradient
        print('gradient: ' + str(gradient))
        print(shift)

    print(compute_basic_cost(vol_a, vol_b, shift))

    return align_volumes(vol_a, vol_b, shift)


class GeneticRegisterer(object):
    def __init__(self, vol_a: Volume, vol_b: Volume, base: np.ndarray,
                 cost: Callable[[Volume, Volume, np.ndarray], float], population_size: int):
        self.population = base + np.random.randn(population_size, 3) * 0.03
        self.population_size = population_size
        self.evaluate = partial(cost, vol_a, vol_b)
        self.best_scores = []

    def get_best(self):
        scores = np.zeros(self.population_size)
        for i, v in enumerate(self.population):
            scores[i] = self.evaluate(v) * 2

        return self.population[np.argmin(scores)]

    def next_gen(self):
        scores = np.zeros(self.population_size)
        for i, v in enumerate(self.population):
            scores[i] = self.evaluate(v) * 2

        new_pop = np.zeros((self.population_size, 3))
        new_pop[:4] = self.population[np.argsort(scores)[:4]]
        for i in range(4, self.population_size - 4):
            parent = self.population[self.choose_random_weighted(1 / scores)]
            new_pop[i] = self.mutate(parent)

        self.population = new_pop

        self.best_scores.append(np.min(scores))
        print(self.best_scores[-1])

    @staticmethod
    def mutate(parent):
        return parent + np.random.randn(3) * 0.3 if random.random() < 0.22 else np.copy(parent)

    @staticmethod
    def choose_random_weighted(weights):
        total = np.sum(weights)
        current = 0
        pick = random.uniform(0, total)
        for i, weight in enumerate(weights):
            current += weight
            if current >= pick:
                return i


def register_genetic(vol_a: Volume, vol_b: Volume):
    shift = get_com_image_shift(vol_a, vol_b)
    zoomed_a, zoomed_b = make_isotropic(vol_a, vol_b)
    registerer = GeneticRegisterer(zoomed_a, zoomed_b, shift, compute_basic_cost, 90)
    # i = 0
    # while len(registerer.best_scores) == 0 or registerer.best_scores[-1] > 70:
    for i in range(0):
        print('gen: ' + str(i))
        registerer.next_gen()
        i += 1

    shift = registerer.get_best()

    plt.plot(registerer.best_scores)
    plt.show()

    return align_volumes(zoomed_a, zoomed_b, shift)


def register_ants(vol_a: Volume, vol_b: Volume):
    import ants
    zoomed_a, zoomed_b = make_isotropic(vol_a, vol_b)
    img_a = ants.from_numpy(zoomed_a.data)
    img_b = ants.from_numpy(zoomed_b.data)
    # init = ants.affine_initializer(img_a, img_b)
    transforms = ants.registration(img_a, img_b, type_of_transform='Translation', verbose=True)
    result_img = ants.apply_transforms(img_a, img_b, transformlist=transforms['fwdtransforms'], verbose=True)
    result_vol = Volume(result_img.numpy(), zoomed_b.resolution)
    return zoomed_a, result_vol


@profile
def register_sitk(vol_a: Volume, vol_b: Volume):
    zoomed_a, zoomed_b = make_isotropic(vol_a, vol_b)
    import SimpleITK as sitk
    import sys
    print(sys.getsizeof(zoomed_a))
    print(sys.getsizeof(zoomed_b))
    # crop_offset = np.multiply(zoomed_a.data.shape, 1.0) // 2
    # crop_offset = crop_offset.astype(np.int)
    # center = np.floor_divide(zoomed_a.data.shape, 2)

    img_a = sitk.GetImageFromArray(zoomed_a.data)
    img_b = sitk.GetImageFromArray(zoomed_b.data)
    print(sys.getsizeof(img_a))
    print(img_b.GetDimension())
    # img_a = sitk.GetImageFromArray(vol_a.data)
    # img_a.SetSpacing(vol_a.resolution)
    # img_b = sitk.GetImageFromArray(vol_b.data)
    # img_b.SetSpacing(vol_b.resolution)

    parameterMap = sitk.GetDefaultParameterMap('translation')
    parameterMap['AutomaticTransformInitialization'] = ['true']
    parameterMap['MaximumNumberOfSamplingAttempt'] = ['4']

    itk_filter = sitk.ElastixImageFilter()
    itk_filter.LogToConsoleOn()
    itk_filter.SetFixedImage(img_a)
    itk_filter.SetMovingImage(img_b)
    itk_filter.SetParameterMap(parameterMap)
    print(itk_filter.GetParameter(1, 'FixedImageDimension'))

    result_img2 = sitk.GetArrayFromImage(itk_filter.Execute())

    result_img = sitk.GetArrayFromImage(itk_filter.GetResultImage())
    print(np.max(np.abs(result_img - zoomed_a.data)))
    print(np.max(np.abs(result_img2 - zoomed_a.data)))
    important_func()
    # zoomed_a, zoomed_b = make_isotropic(vol_a, Volume(result_img, img_b.GetSpacing()))
    return zoomed_a, Volume(result_img, zoomed_b.resolution)


def register_dipy(vol_a: Volume, vol_b: Volume, init_translation: Optional[np.ndarray] = None):
    from dipy.align.imaffine import (transform_centers_of_mass,
                                     MutualInformationMetric,
                                     AffineRegistration)
    from dipy.align.transforms import (TranslationTransform3D,
                                       RigidTransform3D)

    # zoomed_a, zoomed_b = make_isotropic(vol_a, vol_b)

    grid2world_a = np.array([
        [vol_a.resolution[0], 0, 0, 0],
        [0, vol_a.resolution[1], 0, 0],
        [0, 0, vol_a.resolution[2], 0],
        [0, 0, 0, 1]
    ])

    grid2world_b = np.array([
        [vol_b.resolution[0], 0, 0, 0],
        [0, vol_b.resolution[1], 0, 0],
        [0, 0, vol_b.resolution[2], 0],
        [0, 0, 0, 1]
    ])

    I = np.eye(4)

    c_of_mass = transform_centers_of_mass(vol_a.data, I, vol_b.data, I)
    if init_translation is not None:
        com_affine = c_of_mass.get_affine()
        com_affine[0:3, 3] = init_translation
        c_of_mass.set_affine(com_affine)
        print(c_of_mass)
        # transformed_b = c_of_mass.transform(vol_b.data)

        # return vol_a, Volume(transformed_b, vol_b.resolution)

    nbins = 32
    sampling_prop = None
    metric = MutualInformationMetric(nbins, sampling_prop)

    level_iters = [10000, 1000, 250]

    sigmas = [5.0, 2.0, 0.0]

    factors = [4, 2, 1]

    affreg = AffineRegistration(metric=metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    transform = TranslationTransform3D()
    params0 = None
    starting_affine = c_of_mass.affine
    translation = affreg.optimize(vol_a.data, vol_b.data, transform, params0,
                                  I, I,
                                  starting_affine=starting_affine)

    transform = RigidTransform3D()
    params0 = None
    starting_affine = translation.affine
    rigid = affreg.optimize(vol_a.data, vol_b.data, transform, params0,
                            I, I,
                            starting_affine=starting_affine)

    # affreg = AffineRegistration(metric=metric,
    #                             level_iters=[100, 75, 50],
    #                             sigmas=sigmas,
    #                             factors=factors)

    # transform = AffineTransform3D()
    # params0 = None
    # starting_affine = rigid.affine
    # affine = affreg.optimize(zoomed_a.data, zoomed_b.data, transform, params0,
    #                          I, I,
    #                          starting_affine=starting_affine)

    transformed_b = rigid.transform(vol_b.data)

    return vol_a, Volume(transformed_b, vol_b.resolution)


def register_skimage(vol_a: Volume, vol_b: Volume):
    zoomed_a, zoomed_b = make_isotropic(vol_a, vol_b)
    from skimage.feature import register_translation
    # from scipy.ndimage import shift

    max_shape = np.max([zoomed_a.data.shape, zoomed_b.data.shape], axis=0)
    print(max_shape)
    zoomed_a_padded = np.pad(zoomed_a.data, [(0, max_shape[i] - zoomed_a.data.shape[i]) for i in range(3)], 'constant',
                             constant_values=0)
    zoomed_b_padded = np.pad(zoomed_b.data, [(0, max_shape[i] - zoomed_b.data.shape[i]) for i in range(3)], 'constant',
                             constant_values=0)

    shift, _, _ = register_translation(zoomed_a_padded, zoomed_b_padded)
    shifted = ndimage.shift(zoomed_b_padded, shift, order=1)
    return Volume(zoomed_a_padded, zoomed_a.resolution), Volume(shifted, zoomed_b.resolution)


# def register_elastix(vol_a: Volume, vol_b: Volume):
#     zoomed_a, zoomed_b = make_isotropic(vol_a, vol_b)
#     zoomed_a_im = pyelastix.Image(zoomed_a.data)
#     zoomed_b_im = pyelastix.Image(zoomed_b.data)
#     zoomed_a_im.spacing = zoomed_a.resolution/1000
#     zoomed_b_im.spacing = zoomed_b.resolution/1000
#     zoomed_a_im.origin = zoomed_a.get_center_of_mass()/1000
#     zoomed_b_im.origin = zoomed_b.get_center_of_mass()/1000
#
#     params = pyelastix.get_default_params(type='RIGID')
#     params.StepLength = 0.005
#     params.Scales = 2000000.0
#     params.AutomaticScalesEstimation = False
#     params.MaximumNumberOfIterations = 2
#     params.AutomaticTransformInitialization = True
#     vol_b_deformed, field = pyelastix.register(zoomed_b_im.astype('float32'), zoomed_a_im.astype('float32'), params, verbose=2)
#     min = np.min(vol_b_deformed)
#     max = np.max(vol_b_deformed)
#
#     return zoomed_a, Volume((vol_b_deformed-min)/(max-min), zoomed_b.resolution)
#
#
# def register_itk(vol_a: Volume, vol_b: Volume):
#     zoomed_a, zoomed_b = make_isotropic(vol_a, vol_b)
#     selx = sitk.ElastixImageFilter()
#     selx.SetFixedImage(sitk.GetImageFromArray(zoomed_a.data))
#     selx.SetMovingImage(sitk.GetImageFromArray(zoomed_b.data))
#
#     selx.LogToConsoleOn()
#     selx.Set
#     selx.Execute()
#
#     return zoomed_a, Volume(sitk.GetArrayFromImage(selx.GetResultImage()), zoomed_b.resolution)

@profile
def make_isotropic(vol_a: Volume, vol_b: Volume):
    import scipy.interpolate
    ax = np.arange(vol_a.data.shape[2])
    bx = np.arange(vol_b.data.shape[1])
    print(vol_a.data.dtype)
    print(vol_b.data.dtype)
    print('test1')
    min_res = min(np.min(vol_a.resolution), np.min(vol_b.resolution))
    print(min_res)
    # zoomed_volA = scipy.interpolate.interp1d(ax, vol_a.data, axis=2)(np.linspace(0, len(ax)-1, len(ax)*(vol_a.resolution[2]/vol_b.resolution[2]))).astype(np.uint16)
    zoomed_volA = scipy.ndimage.zoom(vol_a.data, vol_a.resolution / min_res, order=1)
    print('test2')
    # zoomed_volB = scipy.interpolate.interp1d(bx, vol_b.data, axis=1)(np.linspace(0, len(bx)-1, len(bx)*(vol_b.resolution[1]/vol_a.resolution[1]))).astype(np.uint16)
    zoomed_volB = scipy.ndimage.zoom(vol_b.data, vol_b.resolution / min_res, order=1)

    print(vol_a.resolution / min_res)
    print(vol_b.resolution / min_res)

    result_vol_a = Volume(zoomed_volA,
                          (vol_a.resolution[0], vol_a.resolution[1], vol_b.resolution[2]))
    result_vol_b = Volume(zoomed_volB,
                          (vol_b.resolution[0], vol_a.resolution[1], vol_b.resolution[2]))

    return result_vol_a, result_vol_b


def align_volumes(vol_a: Volume, vol_b: Volume, shift: np.ndarray):
    # TODO: Resulting volumes must have equal resolution
    shifted_volB = np.clip(scipy.ndimage.interpolation.shift(vol_b.data, -shift / vol_b.resolution),
                           0.0, 1.0)

    new_shape = np.min([vol_a.data.shape, shifted_volB.shape], axis=0)
    result_vol_a = Volume(vol_a.data[:new_shape[0], :new_shape[1], :new_shape[2]], vol_a.resolution)
    result_vol_b = Volume(shifted_volB[:new_shape[0], :new_shape[1], :new_shape[2]], vol_b.resolution)
    return result_vol_a, result_vol_b


def remove_extremes(data, sigma):
    mean = np.mean(data)
    data[data - mean > sigma] = mean + sigma
    data[data - mean < -sigma] = mean - sigma


def fuse(vol_a: Volume, vol_b: Volume):
    return Volume((vol_a.data + vol_b.data) / 2, vol_a.resolution)


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

    return Volume(np.clip((estimate + est_min) / (est_max - est_min), 0.0, 1.0), vol.resolution)


def deconvolve(vol_a: Volume, vol_b: Volume, n: int, blurA: BlurFunction, blurB: BlurFunction) -> Volume:
    view_a, view_b = vol_a.data, vol_b.data
    # view_a = np.clip(
    #     scipy.ndimage.zoom(view_a, (2, 2, 2)), 0.0, 1.0)
    # view_b = np.clip(
    #     scipy.ndimage.zoom(view_b, (2, 2, 2)), 0.0, 1.0)

    print('hey')

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

    return Volume(estimate, vol_a.resolution)


def fuse_basic(vol_a: np.ndarray, vol_b: np.ndarray, pixel_size: float, interval: float, trans: np.ndarray):
    # z_point_count = volB.shape[0]
    z_point_count = int(math.floor((vol_a.shape[2] * interval) / math.sqrt(2) / pixel_size + 0.5))
    z_points = np.linspace(0, vol_a.shape[2], z_point_count)

    total_points = vol_a.shape[0] * vol_a.shape[1] * z_point_count

    points_A = [np.zeros((total_points,)), np.zeros((total_points,)), np.zeros((total_points,))]
    points_B = [np.zeros((total_points,)), np.zeros((total_points,)), np.zeros((total_points,))]

    i = 0

    for x in range(vol_a.shape[0]):
        for y in range(vol_a.shape[1]):
            for z_index in range(z_point_count):
                z = z_points[z_index]

                points_A[0][i] = x
                points_A[1][i] = y
                points_A[2][i] = z

                loc = trans @ np.array([x, y, z, 1])
                points_B[0][i] = loc[0]
                points_B[1][i] = loc[1]
                points_B[2][i] = loc[2]

                i += 1

    values_A = np.clip(ndimage.map_coordinates(vol_a, points_A, order=3), 0.0, 1.0)
    values_B = np.clip(ndimage.map_coordinates(vol_b, points_B, order=3), 0.0, 1.0)

    values_A = values_A.reshape((vol_a.shape[0], vol_a.shape[1], z_point_count))
    values_B = values_B.reshape((vol_a.shape[0], vol_a.shape[1], z_point_count))

    result = np.zeros((vol_a.shape[0], vol_a.shape[1], z_point_count), dtype='float64')

    for x in range(vol_a.shape[0]):
        for y in range(vol_a.shape[1]):
            for z_index in range(z_point_count):
                valA = values_A[x, y, z_index]
                valB = values_B[x, y, z_index]
                avg_val = (valA + valB) / 2
                result[x, y, z_index] = avg_val

    return result


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

    return np.median(sigmas_z, axis=0) * vol.resolution[2], np.median(sigmas_xy, axis=0) * vol.resolution[0]


def load_volumes(paths: List[str], spacing: Tuple[float, float, float], scale: float = None):
    from tifffile import imread
    import gc

    datas = []
    for path in paths:
        logger.info("Loading volume from {}".format(path))
        data = imread(path)

        logger.info("Initial volume shape: {}".format(data.shape))

        if len(data.shape) > 3:
            # data = data[:, 0, :, :]  # TODO: What does the second dimension actually represent in this case?
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

    # TODO: Handle different data types (aside from uint16)

    for i in range(len(datas)):
        volumes.append(Volume(datas[i].swapaxes(0, 2).swapaxes(0, 1), spacing))
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
