import logging
from typing import Tuple

import math
import numpy as np
import scipy
from numba import jit, prange

from dispim import Volume

logger = logging.getLogger(__name__)


def compute_true_interval(vol: Volume, invert: bool, n: int = 100) -> float:
    from scipy.optimize import minimize_scalar
    from dispim.util import crop_view
    vol = crop_view(vol, 0.4)
    resolution = vol.spacing
    rand_slice_indices = np.random.randint(0, vol.data.shape[2] - 1, n)

    def compute_error(x):
        shift = (resolution[2] + x) / resolution[1]
        if invert:
            shift = -shift

        error = 0
        for slice_index in rand_slice_indices:
            shifted_b = scipy.ndimage.shift(vol[:, :, slice_index + 1], (0, shift), order=1)
            error += np.mean(
                (vol[:, :, slice_index].astype(np.float) - shifted_b.astype(np.float)) ** 2)
        return error

    result = minimize_scalar(compute_error)

    return resolution[2] + result.x


def unshift_fast(vol: Volume, invert: bool = False, estimate_true_interval: bool = True, rotate: bool = True) -> Volume:
    if estimate_true_interval:
        interval = compute_true_interval(vol, invert)
        vol = Volume(vol, spacing=vol.spacing[:2] + (interval,))
        logger.debug('Estimated volume interval: {}'.format(interval))

    # FIXME: Metadata is lost here

    if invert:
        data = unshift_fast_numbai(np.array(vol), vol.spacing)
        if rotate:
            data = np.rot90(data, k=2, axes=(1, 2))
        return Volume(data, inverted=False, spacing=(vol.spacing[0], vol.spacing[1] / np.sqrt(2),
                                                     vol.spacing[2] * np.sqrt(2)), is_skewed=False)
    else:
        return Volume(unshift_fast_numba(np.array(vol), vol.spacing),
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

            val1 = data[:, layer, x1] if 0 <= x1 < d else np.zeros(w, dtype=np.uint16)
            val2 = data[:, layer, x2] if 0 <= x2 < d else np.zeros(w, dtype=np.uint16)

            result[:, layer, x] = val1 * (1 - delta) + val2 * delta

    return result


def unshift_fast_diag(vol: Volume, invert: bool = False, estimate_true_interval: bool = True,
                      rotate: bool = True) -> Volume:
    if estimate_true_interval:
        interval = compute_true_interval(vol, invert)
        vol = Volume(vol, spacing=vol.spacing[:2] + (interval,))
        logger.debug('Estimated volume interval: {}'.format(interval))

    # FIXME: Metadata is lost here

    if invert:
        data = unshift_fast_numbai_diag(np.array(vol), vol.spacing)
        if rotate:
            data = np.rot90(data, k=2, axes=(1, 2))
        return Volume(data, inverted=False, is_skewed=False)
    else:
        return Volume(unshift_fast_numba_diag(np.array(vol), vol.spacing), is_skewed=False)


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


def center_psf(psf: np.ndarray, psf_half_size: int) -> np.ndarray:
    from scipy.ndimage import center_of_mass
    from dispim.util import extract_3d
    com = center_of_mass(psf)
    return extract_3d(psf, np.array(com), psf_half_size)


def apply_registration(vol_a: Volume, vol_b: Volume, order: int = 2) -> Tuple[Volume, Volume]:
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

    # TODO: Remove this, unnecessary
    final_shape = (np.ceil(
        np.linalg.inv(transform_a) @ np.array([vol_a.shape[0], vol_a.shape[1], vol_a.shape[2], 1]))).astype(
        np.int)[:3]

    # psf = vol_a.psf
    #
    # psf_shape_old = psf.shape
    # psf_shape_transformed = (np.ceil(
    #     np.linalg.inv(transform_a) @ np.array([psf.shape[0], psf.shape[1], psf.shape[2], 1]))).astype(
    #     np.int)[:3]
    #
    # psf = affine_transform(psf, transform_a, output_shape=psf_shape_transformed, order=3)
    # psf = center_psf(psf, psf_shape_old[0])

    vol_a = Volume(affine_transform(vol_a, transform_a, output_shape=final_shape, order=order),
                   spacing=(min_res, min_res, min_res), is_skewed=False, inverted=False,
                   )

    transform_b = np.linalg.inv(vol_b.grid_to_world)
    transform_b = transform_b @ vol_b.world_transform
    transform_b = transform_b @ grid_to_world_final

    logger.debug(f'Final B transform:\n{transform_b}')

    # psf = vol_b.psf
    #
    # psf_shape_old = psf.shape
    # psf_shape_transformed = (np.ceil(
    #     np.linalg.inv(transform_b) @ np.array([psf.shape[0], psf.shape[1], psf.shape[2], 1]))).astype(
    #     np.int)[:3]
    #
    # psf = affine_transform(psf, transform_b, output_shape=psf_shape_transformed, order=3)
    # psf = center_psf(psf, psf_shape_old[0])

    transformed = affine_transform(vol_b, transform_b, output_shape=final_shape, order=order)

    logger.debug(f'Transformed A average value: {np.mean(vol_a)}')
    logger.debug(f'Transformed B average value: {np.mean(transformed)}')

    return vol_a, Volume(transformed, spacing=vol_a.spacing, is_skewed=False, inverted=False)
