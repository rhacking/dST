#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Tuple, Union, List

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats
from scipy.stats import norm

import dispim

Rotation = Union[Tuple[float, float, float], np.ndarray]


class Samplable(object):
    def sample_points(self, points: np.ndarray) -> List[float]:
        raise NotImplementedError()

    def sample_plane(self, origin: np.ndarray, v1: np.ndarray, v2: np.ndarray,
                     plane_step: float,
                     plane_size: Tuple[int, int]) -> np.ndarray:
        """
        Sample this volume in a specified plane

        :param origin: The origin of the sampling plane
        :param v1: The first (unit) vector describing the direction of the sampling plane
        :param v2: The second (unit) vector describing the direction of the sampling plane
        :param plane_step: The distance between two sampling points in the plane
        :param plane_size: The size of the sampling plane
        :return: A 2d ndarray representing the values of the points that were sampled
        """
        if abs(np.dot(v1, v2)) > 0.00001:
            raise ValueError('Plane vectors must be perpendicular')

        xpoints = np.linspace(0, plane_size[0], int(plane_size[0] / plane_step))
        ypoints = np.linspace(0, plane_size[1], int(plane_size[1] / plane_step))

        xv, yv = np.meshgrid(xpoints, ypoints)
        xv = xv.ravel()
        yv = yv.ravel()
        local_points = np.vstack((xv, yv, np.zeros(xv.shape[0])))
        v3 = np.cross(v2, v1)
        trans = np.vstack([v1, v2, v3])
        points = (trans @ local_points).T + origin

        samples = self.sample_points(points)
        return np.clip(np.reshape(samples, (len(ypoints), len(xpoints))).astype(np.float64), 0.0, 1.0).T

    def sample_volume(self, origin: np.ndarray, v1: np.ndarray, v2: np.ndarray,
                      plane_step: float, depth_step: float,
                      volume_size: Tuple[int, int, int]) -> np.ndarray:
        """
        Sample this volume in a specified plane

        :param volume_size:
        :param depth_step:
        :param origin: The origin of the sampling plane
        :param v1: The first (unit) vector describing the direction of the sampling plane
        :param v2: The second (unit) vector describing the direction of the sampling plane
        :param plane_step: The distance between two sampling points in the plane
        :return: A 2d ndarray representing the values of the points that were sampled
        """
        if abs(np.dot(v1, v2)) > 0.00001:
            raise ValueError('Plane vectors must be perpendicular')

        v3 = np.cross(v2, v1)

        xpoints = np.linspace(0, volume_size[0], round(volume_size[0] / plane_step))
        ypoints = np.linspace(0, volume_size[1], round(volume_size[1] / plane_step))
        zpoints = np.linspace(0, volume_size[2], round(volume_size[2] / depth_step))

        xv, yv, zv = np.meshgrid(xpoints, ypoints, zpoints)
        del xpoints, ypoints, zpoints
        xv = xv.ravel()
        yv = yv.ravel()
        zv = zv.ravel()
        local_points = [xv, yv, zv]
        del xv, yv, zv
        trans = np.vstack([v1, v2, v3])
        points = (trans @ local_points).T + origin
        del local_points

        samples = self.sample_points(points)
        del points
        return np.clip(np.reshape(samples, (round(volume_size[1] / plane_step), round(volume_size[0] / plane_step),
                                            round(volume_size[2] / depth_step))).astype('float64'), 0.0, 1.0).swapaxes(
            0, 1)


class SampleSphere(Samplable):
    def __init__(self, radius: float):
        self.radius = radius
        # self.origin = np.random.rand(3)*4-2
        self.origin = np.array([0, 0, 0])

    def sample_points(self, points: np.ndarray):
        return [1.0 if np.linalg.norm(point) <= self.radius and np.all(point >= 0) else 0.0 for point in points]


class SampleSpheres(Samplable):
    def __init__(self, radius: float):
        self.radius = radius
        self.origins = [np.array([0, 0, 0])]

    def sample_points(self, points: np.ndarray):
        return [1 if np.any(np.linalg.norm((point - self.origins) * np.array([1, 1, 1]), axis=1) <= self.radius) else 0
                for point in points]


class SampleBeads(Samplable):
    def __init__(self, n: int):
        self.positions = np.random.rand(n, 3) * np.array([15, 15, 15]) - np.array([7.5, 7.5, 7.5])
        self.radii = 0.09
        self.largest = 0

    def sample_points(self, points: np.ndarray):
        result = np.max(
            -np.clip((scipy.spatial.distance.cdist(points, self.positions) - self.radii - 0.15) / 0.15, -1.0, 0.0),
            axis=1)
        result_largest = np.max(result)
        if result_largest > self.largest:
            self.largest = result_largest
        return result


class SampleEllipsoid(Samplable):
    def __init__(self, size: Tuple[float, float, float]):
        self.scale = 1 / np.array(size)
        theta = np.pi / 4
        self.rotation = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])

    def sample_points(self, points: np.ndarray):
        return np.logical_and(np.linalg.norm((self.rotation @ points.T).T * self.scale, axis=1) - 1 <= 0,
                              (self.rotation @ points.T)[1, :] >= 0)


class SampleVolume(Samplable):
    def __init__(self, data: np.ndarray, resolution: np.ndarray):
        self.data = data
        self.resolution = resolution

    def to_array_space(self, points: np.ndarray):
        """
        Transform a list of points from world space into array space.

        :param points: The list of points to transform
        :return: An ndarray containing the transformed points
        """
        return np.array([point / self.resolution for point in points])

    def sample_points(self, points: np.ndarray):
        """
        Samples multiple points in this volume. The coordinates are in world-
        space.

        @param points: The list of points to sample (2D)
        @return: A list of values sampled from the volume, in the order of
        the points input array
        """
        x, y, z = self.to_array_space(points).T[0:3]
        return scipy.ndimage.map_coordinates(self.data, [x, y, z])

    # def show(self):
    #     mlab.pipeline.volume(mlab.pipeline.scalar_field(self.data))


def show_slice(points):
    plt.imshow(points, cmap=plt.get_cmap('Greys'), interpolation='nearest')
    plt.show()


def generate_linear_volume(size: np.ndarray, resolution: np.ndarray, start=0,
                           stop=1):
    array_size = np.around(size / resolution).astype('int')
    data = np.linspace(start, stop, array_size[0])[:, np.newaxis, np.newaxis]
    data = np.repeat(data, array_size[1], axis=1)
    data = np.repeat(data, array_size[2], axis=2)

    volume = SampleVolume(data, resolution)
    return volume


def fwhm_to_sigma(fwhm):
    return fwhm / (2 * math.sqrt(2 * math.log(2)))


def scan(vol: Samplable, right: bool, pixel_size: float = 0.3, pixel_samples: int = 2,
         interval: float = 0.3, plane_size=6, gaussian: bool = True,
         gaussian_delta: float = 0.078, gaussian_FWHM: float = 1.0) -> dispim.Volume:
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, math.sqrt(2) / 2, -math.sqrt(2) / 2]) if right else \
        np.array([0, math.sqrt(2) / 2, math.sqrt(2) / 2])
    v3 = []
    scan_vol = []
    start = np.array([-plane_size, -plane_size * (math.sqrt(2) / 2), 0])
    volume_depth = 40.0

    weights = []

    if gaussian:
        v3 = np.cross(v1, v2)
        slices_side = gaussian_FWHM / gaussian_delta * 2
        sigma = fwhm_to_sigma(gaussian_FWHM)
        weights = [norm.pdf(x, loc=0, scale=sigma) for x in
                   np.linspace(-slices_side * gaussian_delta, slices_side * gaussian_delta, int(slices_side * 2 + 1))]

    for z in np.linspace(-volume_depth / 2, volume_depth / 2, int(volume_depth / interval)):
        if not gaussian:
            points = vol.sample_plane(start + np.array([0, 0, z]),
                                      v1, v2, pixel_size, (plane_size * 2, plane_size * 2))
            scan_vol.append(points)
        else:
            sheet_points = []
            for i in np.linspace(-slices_side * gaussian_delta, slices_side * gaussian_delta, int(slices_side * 2 + 1)):
                shifted_start = start + v3 * i
                # print(shifted_start - start)
                points = vol.sample_plane(shifted_start + np.array([0, 0, z]),
                                          v1, v2, pixel_size / pixel_samples, (plane_size * 2, plane_size * 2))
                m, n = points.shape
                block_size = pixel_samples
                points = points.reshape(m // block_size, block_size, n // block_size, block_size).mean((1, 3),
                                                                                                       keepdims=True).reshape(
                    m // block_size, n // block_size)
                # if np.linalg.norm(v3*i+np.array([0, 0, z])) < 0.00001:
                #     print(np.max(points))
                sheet_points.append(points)

            # blurred_slice = scipy.ndimage.filters.gaussian_filter(sheet_points, (sigma/gaussian_delta, 0, 0), mode='constant')[int((len(sheet_points)-1)/2)]
            blurred_slice = np.average(sheet_points, axis=0, weights=weights)
            # if np.max(blurred_slice) > 0.004:
            #     plt.plot(np.max(sheet_points, axis=(1, 2)))
            #     plt.show()
            #     blurred = scipy.ndimage.filters.gaussian_filter(sheet_points, (sigma/gaussian_delta, 0, 0), mode='constant')
            #     plt.plot(np.max(blurred, axis=(1, 2)))
            #     plt.show()
            scan_vol.append(blurred_slice)
            # shifted_start = start + v3 * -slices_side*gaussian_delta + np.array([0, 0, z])
            # slab = vol.sample_volume(shifted_start, v1, v2, pixel_size/pixel_samples, gaussian_delta,
            #                            (plane_size * 2, plane_size * 2, gaussian_delta*int(slices_side*2+1)))
            # # slice = np.average(slab, axis=2, weights=weights)
            # slice = scipy.ndimage.filters.gaussian_filter(slab, (0, 0, sigma / gaussian_delta))[:, :,
            #     slices_side]
            # del slab
            # m, n = slice.shape
            # block_size = pixel_samples
            # slice = slice.reshape(m // block_size, block_size, n // block_size, block_size).mean((1, 3), keepdims=True)\
            #     .reshape(m // block_size, n // block_size)
            # scan_vol.append(slice)

    # Simply return the scanned volume
    result = np.array(scan_vol)
    result = np.swapaxes(result, 0, 2)
    result = np.swapaxes(result, 0, 1)
    result_vol = dispim.Volume(result, (pixel_size, pixel_size, interval / math.sqrt(2)))
    return result_vol


def plot3d(data, spacing):
    field = mlab.pipeline.scalar_field(data)
    field.spacing = spacing
    mlab.pipeline.volume(field, vmin=0, vmax=0.8)


def test():
    vol = SampleSphere(4)
    # vol = SampleBeads(40)
    # vol = SampleEllipsoid((3, 3, 7))

    print("Scanning from A")
    scan_vol_A = scan(vol, False, gaussian=False, gaussian_FWHM=0.2)
    scan_vol_A = dispim.Volume((scan_vol_A.data * (2 ** 16 - 1)).astype(np.uint16), scan_vol_A.spacing)
    scan_vol_A.save_tiff('test_A')
    print("Deskewing A")
    deskewed_A = dispim.unshift_fast(scan_vol_A, False)
    deskewed_A.save_tiff_single('sim_beads_A')
    deskewed_A.data *= 1
    print(np.max(deskewed_A.data))
    deskewed_A.save_tiff('test_A_deskew')
    #
    print("Scanning from B")
    scan_vol_B = scan(vol, True, gaussian=False, gaussian_FWHM=0.2)
    scan_vol_B = dispim.Volume((scan_vol_B.data * (2 ** 16 - 1)).astype(np.uint16), scan_vol_B.spacing)
    scan_vol_B.save_tiff('test_B')
    print("Deskewing B")
    deskewed_B = dispim.unshift_fast(scan_vol_B, True)
    # deskewed_B.save_tiff_single('sim_beads_B')
    deskewed_B.save_tiff('test_B_deskew')
    # deskewed_B = deskewed_B.rot90()

    # print("Aligning images")
    # shift = dispim.get_com_image_shift(deskewed_A, deskewed_B)
    # zoomedA, zoomedB = dispim.register_elastix(deskewed_A, deskewed_B)
    # za, zb = dispim.make_isotropic(deskewed_A, deskewed_B)
    # za.save_tiff('test_a_nii')
    # zb.save_tiff('test_b_nii')
    # za, zb = dispim.align_volumes(za, zb, np.array([0, 0, 0]))
    # za.save_tiff('test_a_nii2')
    # zb.save_tiff('test_b_nii2')
    # zoomedA2, zoomedB2 = dispim.register_ants(deskewed_A, deskewed_B)
    # zoomedA, zoomedB = dispim.register_genetic(deskewed_A, deskewed_B)
    # zoomedA, zoomedB = dispim.regist(deskewed_A, deskewed_B)
    # zoomedA, zoomedB = dispim.align_volumes(*dispim.make_isotropic(deskewed_A, deskewed_B), shift)
    # # zoomedA, zoomedB = dispim.register_   genetic(deskewed_A, deskewed_B)
    # # # zoomedA, zoomedB = dispim.register_genetic(deskewed_A, deskewed_B)
    # # # zoomedA_bad, zoomedB_bad = dispim.register(deskewed_A, deskewed_B, n=0)
    # #
    # zoomedA.save_tiff('align_A')
    # zoomedB.save_tiff('align_B')
    # dispim.save_dual_tiff('align', zoomedA, zoomedB)
    # dispim.save_dual_tiff('align_ants', zoomedA2, zoomedB2)
    # # #
    # print("Deconvolving")
    # sigma = fwhm_to_sigma(2.5)
    # trunc = 2.5 / sigma
    # blurA = lambda vol: scipy.ndimage.filters.gaussian_filter(vol, (0, 0, sigma / zoomedA.resolution[2]), truncate=trunc)
    # blurB = lambda vol: scipy.ndimage.filters.gaussian_filter(vol, (0, sigma / zoomedB.resolution[1], 0), truncate=trunc)
    #
    # result = dispim.deconvolve(zoomedA, zoomedB, 8, blurA, blurB)
    # result.save_tiff('deconv')

    # result_bad = dispim.deconvolve(zoomedA_bad, zoomedB_bad, 0, blurA, blurB)
    # result_bad.save_tiff('deconv_bad')


if __name__ == '__main__':
    test()
