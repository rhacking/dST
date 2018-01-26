import math
import shutil
import os

import numpy as np
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
from tifffile import imsave

from typing import Tuple


class Volume(object):
    def __init__(self, data: np.ndarray, resolution: Tuple[float, float, float]):
        self.data = data
        self.resolution = np.array(resolution)

    def get_center_of_mass(self):
        return ndimage.center_of_mass(self.data) * self.resolution

    def rot90(self):
        reorder_indices = np.array([0, 2, 1])
        new_data = np.swapaxes(self.data, 1, 2)[:, :, ::-1]
        return Volume(new_data, tuple(self.resolution[reorder_indices]))

    def save_tiff(self, name: str, invert=False):
        if os.path.exists(name):
            shutil.rmtree(name)
        os.makedirs(name)
        data = self.data
        if invert:
            data = np.swapaxes(data, 0, 2)
            data = np.flipud(data)
            data = np.swapaxes(data, 0, 2)
        for slc_index in range(data.shape[2]):
            imsave(name+'/'+name+str(slc_index)+'.tiff', (data[:, :, slc_index]*(2**8-8)).astype('uint8').T)


def unshift(vol: np.ndarray, pixel_size: float, interval: float, invert: bool = False, axis: int = 1):
    """
    Deskew the given volume

    :param vol: The volume to deskew
    :param pixel_size: The width/height of a voxels (assumed to be equal)
    :param interval: The stage interval
    :param invert: Whether the stage was shifted from left to right (or right to left)
    :param axis: The axis toe shift in (should be the axis that represents up)
    :return: The deskewed volume
    """
    # Determine the shift step
    shift = np.array([0, 0], dtype='float64')
    shift[axis] = interval / math.sqrt(2) / pixel_size

    new_vol = []
    # Compute the maximum shift required
    total_shift = (np.ceil(vol.shape[2] * shift)).astype('uint16')

    # If the stage direction is inverted, invert the shifting as well
    if invert:
        shift = -shift

    # Scale the volume to accommodate the maximum shift possible
    if not invert:
        vol = np.pad(vol, ((0, total_shift[0]), (0, total_shift[1]), (0, 0)), mode='constant', constant_values=0.0)
    else:
        vol = np.pad(vol, ((total_shift[0], 0), (total_shift[1], 0), (0, 0)), mode='constant', constant_values=0.0)

    # Iterate over all slices
    for i, slc in enumerate(vol.swapaxes(0, 2).swapaxes(1, 2)):
        # Perform the actual shifting (order-3 spline interpolation)
        slc = scipy.ndimage.interpolation.shift(slc, tuple(shift*i), order=1)
        new_vol.append(slc)


    # Swap the axes such that they're in order again (x, y, z)
    result = np.array(new_vol, dtype='float64')
    result = np.swapaxes(result, 0, 2)
    result = np.swapaxes(result, 0, 1)

    return result


def align_volumes(volA: np.ndarray, volB: np.ndarray, shift: np.ndarray, pixel_size: float, interval: float):
    #volB = np.pad(volB, [(0, math.ceil(max(0, shift[i]))) for i in range(len(shift))], mode='constant', constant_values=0.0)
    zoomed_volB = np.clip(scipy.ndimage.zoom(volB, (1, interval/math.sqrt(2)/pixel_size, 1)), 0.0, 1.0)
    zoomed_volA = np.clip(scipy.ndimage.zoom(volA, (1, 1, interval/math.sqrt(2)/pixel_size)), 0.0, 1.0)
    shifted_volB = np.clip(scipy.ndimage.interpolation.shift(zoomed_volB, -shift/pixel_size), 0.0, 1.0)
    new_shape = np.min([zoomed_volA.shape, shifted_volB.shape], axis=0)
    return (zoomed_volA[:new_shape[0], :new_shape[1], :new_shape[2]], shifted_volB[:new_shape[0], :new_shape[1], :new_shape[2]])


def remove_extremes(data, sigma):
    mean = np.mean(data)
    data[data - mean > sigma] = mean + sigma
    data[data - mean < -sigma] = mean - sigma


def deconvolve(volA: np.ndarray, volB: np.ndarray, n, blurA, blurB):
    estimate = (volA+volB)/2
    for i in range(n):
        estimate_A = estimate * blurA(volA / (blurA(estimate)+1E-5))
        estimate = estimate_A * blurB(volB / (blurB(estimate)+1E-5))
        # estimate = np.clip(estimate * blurB(volB / blurB(estimate)), 0.0, 1.0)

    remove_extremes(estimate, 0.67)
    est_min = np.min(estimate)
    est_max = np.max(estimate)
    return np.clip((estimate + est_min) / (est_max - est_min), 0.0, 1.0)


def fuse_basic(volA: np.ndarray, volB: np.ndarray, pixel_size: float, interval: float, trans: np.ndarray):
    # z_point_count = volB.shape[0]
    z_point_count = math.floor((volA.shape[2]*interval)/math.sqrt(2)/pixel_size+0.5)
    z_points = np.linspace(0, volA.shape[2], z_point_count)

    total_points = volA.shape[0] * volA.shape[1]  * z_point_count

    points_A = [np.zeros((total_points)), np.zeros((total_points)), np.zeros((total_points))]
    points_B = [np.zeros((total_points)), np.zeros((total_points)), np.zeros((total_points))]

    i = 0

    for x in range(volA.shape[0]):
        for y in range(volA.shape[1]):
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

    values_A = np.clip(ndimage.map_coordinates(volA, points_A, order=3), 0.0, 1.0)
    values_B = np.clip(ndimage.map_coordinates(volB, points_B, order=3), 0.0, 1.0)

    values_A = values_A.reshape((volA.shape[0], volA.shape[1], z_point_count))
    values_B = values_B.reshape((volA.shape[0], volA.shape[1], z_point_count))

    result = np.zeros((volA.shape[0], volA.shape[1], z_point_count), dtype='float64')

    for x in range(volA.shape[0]):
        for y in range(volA.shape[1]):
            for z_index in range(z_point_count):
                valA = values_A[x, y, z_index]
                valB = values_B[x, y, z_index]
                avg_val = (valA + valB) / 2
                result[x, y, z_index] = avg_val

    return result
