#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 13:06:50 2017

@author: Roel
"""

import math
import os
from typing import Tuple, Union, List
import shutil

import matplotlib.pyplot as plt
import numpy as np
from mayavi import mlab
import scipy
import scipy.stats
from tifffile import imsave

import dispim

Rotation = Union[Tuple[float, float, float], np.ndarray]


class VirtualVolume(object):
    """
    Represents the result of a scan. The class contains both the collected data and the transformations indicating its
    transform in the world.
    """
    def __init__(self, data: np.ndarray, origin: Tuple[float, float, float], resolution: Tuple[float, float, float],
                 rotation: Rotation, mirrored: bool):
        """
        Instantiates a new VirtualVolume with a given 3d ndarray of data and a set of transformations indicating its
        transform in the world.

        :param data: The voxel data of the VirtualVolume
        :param origin: The origin of the VirtualVolume in the world
        :param resolution: The resolution of the voxel grid
        :param rotation: The rotation of the VirtualVolume in the world
        """
        self.data = data
        self.origin = np.array(origin)
        self.resolution = np.array(resolution)
        self.rotation_euler = rotation
        if rotation[0] != 0:
            theta = rotation[0]
            self.rotation = np.array([
                [1, 0,              0],
                [0, np.cos(theta),  -np.sin(theta)],
                [0, np.sin(theta),  np.cos(theta)]
            ])
        elif rotation[0] != 1:
            theta = rotation[1]
            self.rotation = np.array([
                [np.cos(theta),     0, np.sin(theta)],
                [0,                 1, 0],
                [-np.sin(theta),    0, np.cos(theta)]
            ])
        elif rotation[0] != 2:
            theta = rotation[2]
            self.rotation = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta),  0],
                [0,             0,              1]
            ])
        else:
            self.rotation = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
        self.mirrored = mirrored

        self.rotation4 = np.eye(4, 4)
        self.rotation4[0:3, 0:3] = self.rotation

        self.trans_mat = np.eye(4, 4)
        if self.mirrored:
            self.trans_mat = self.trans_mat @ np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        self.trans_mat = self.trans_mat @ np.array([
            [1, 0, 0, self.origin[0]],
            [0, 1, 0, self.origin[1]],
            [0, 0, 1, self.origin[2]],
            [0, 0, 0, 1],
        ])
        self.trans_mat = self.trans_mat @ self.rotation4
        self.trans_mat = self.trans_mat @ np.array([
            [self.resolution[0], 0, 0, 0],
            [0, self.resolution[1], 0, 0],
            [0, 0, self.resolution[2], 0],
            [0, 0, 0,                  1]
        ])

    def get_center_of_mass(self):
        return scipy.ndimage.center_of_mass(self.data) * self.resolution

    def get_voxel(self, x: float, y: float, z: float):
        loc = np.array([x, y, z, 1])
        # loc = loc * self.resolution
        # loc = self.rotation @ loc
        # loc += self.origin
        # if self.mirrored:
        #     loc[2] = -loc[2]

        return (self.trans_mat @ loc)[0:3]

    def update_data(self, data: np.ndarray):
        return VirtualVolume(data, tuple(self.origin), tuple(self.resolution), self.rotation_euler, self.mirrored)

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

    def rot90(self):
        reorder_indices = np.array([0, 2, 1])
        new_data = np.swapaxes(self.data, 1, 2)[:, :, ::-1]
        return VirtualVolume(new_data, tuple(self.origin[reorder_indices]), tuple(self.resolution[reorder_indices]), tuple(np.array(self.rotation_euler)[reorder_indices]), self.mirrored)


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
        points = []
        w = 0
        xpoints = np.linspace(0, plane_size[0], int(plane_size[0] / plane_step))
        ypoints = np.linspace(0, plane_size[1], int(plane_size[1] / plane_step))

        xv, yv = np.meshgrid(xpoints, ypoints)
        xv = xv.ravel()
        yv = yv.ravel()
        local_points = np.vstack((xv, yv, np.zeros(xv.shape[0])))
        v3 = np.cross(v2, v1)
        trans = np.vstack([v1, v2, v3])
        points = (trans @ local_points).T + origin
        # points2 = []
        # for y in ypoints:
        #     w += 1
        #     for x in xpoints:
        #         points2.append(origin + x * v1 + y * v2)
        #
        # print(np.sum(np.abs(points-points2)))
        # print('x max: ' + str(np.max(np.array(points)[:, 2])))
        # print('x min: ' + str(np.min(np.array(points)[:, 2])))
        # print(points)
        # print(points2)
        samples = self.sample_points(points)
        # ind = np.argmax(samples)
        # print(points[ind])
        return np.clip(np.reshape(samples, (len(ypoints), len(xpoints))).astype('float64'), 0.0, 1.0).T

    def compute_similarity(self, vol: VirtualVolume):
        diff = 0
        max_diff = 0
        max_x = -10000
        min_x = 1000000
        avg_point = np.array([0, 0, 0], dtype='float64')
        num_points = 0
        for x, slc in enumerate(vol.data):
            for y, line in enumerate(slc):
                for z, val in enumerate(line):
                    loc = vol.get_voxel(x, y, z)
                    real_val = self.sample_points(np.array([loc]))[0]
                    avg_point += loc*val
                    num_points += val
                    if loc[1] < min_x:
                        min_x = loc[1]
                    if loc[1] > max_x:
                        max_x = loc[1]
                    if np.linalg.norm(loc) < 0.1:
                        print('huh')
                    if real_val > 0.2:
                        diff += abs(val - real_val)
                        max_diff += 1

        print(avg_point/num_points)

        print(min_x)
        print(max_x)

        return 1-diff/max_diff


class SampleSphere(Samplable):
    def __init__(self, radius: float):
        self.radius = radius
        # self.origin = np.random.rand(3)*4-2
        self.origin = np.array([0, 0, 0])

    def sample_points(self, points: np.ndarray):
        return [1.0 if np.linalg.norm(point) <= self.radius else 0.0 for point in points]


class SampleSpheres(Samplable):
    def __init__(self, radius: float):
        self.radius = radius
        # self.origins = np.random.rand(5, 3)*np.array([4, 1, 3])-np.array([3, 3, 7])
        self.origins = [np.array([0, 0, 0])]

    def sample_points(self, points: np.ndarray):
        return [1 if np.any(np.linalg.norm((point-self.origins) * np.array([1, 1, 1]), axis=1) <= self.radius) else 0 for point in points]


class SampleBeads(Samplable):
    def __init__(self, n: int):
        self.positions = np.random.rand(n, 3) * 14 - 7
        self.radii = np.random.random(n) * 0.9 + 0.3

    def sample_points(self, points: np.ndarray):
        return np.min(scipy.spatial.distance.cdist(points, self.positions)-self.radii, axis=1) <= 0


class SampleEllipsoid(Samplable):
    def __init__(self, size: Tuple[float, float, float]):
        self.scale = 1/np.array(size)
        theta = np.pi/4
        self.rotation = np.array([
                [1, 0,              0],
                [0, np.cos(theta),  -np.sin(theta)],
                [0, np.sin(theta),  np.cos(theta)]
        ])

    def sample_points(self, points: np.ndarray):
        return np.logical_and(np.linalg.norm((self.rotation@points.T).T*self.scale, axis=1) - 1 <= 0, (self.rotation@points.T)[1, :] >= 0)


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

    def show(self):
        mlab.pipeline.volume(mlab.pipeline.scalar_field(self.data))

#
#    def save_tiff(path: str):
#        with TiffWriter.


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


def gaussian_value(x, mu, sigma):
    return scipy.stats.norm.pdf(x, mu, sigma)


# def scan(vol: Samplable, right: bool, pixel_size: float = 0.1,
#          interval: float = 1.0, plane_size=12, gaussian: bool = True, gaussian_samples: int = 7) -> ScanVolume:
#     v2 = (np.array([0, math.sqrt(2) / 2, -math.sqrt(2) / 2]) if right else
#             np.array([0, math.sqrt(2) / 2, math.sqrt(2) / 2]))
#     scan_vol = []
#     start = np.array([-plane_size, -plane_size*(math.sqrt(2)/2), 0])
#     volume_depth = 68.0
#     for z in np.linspace(-volume_depth/2, volume_depth/2, int(volume_depth/interval)):
#         points = vol.sample_plane(start + np.array([0, 0, z]),
#                                    np.array([1, 0, 0]),
#                                    v2, pixel_size, (plane_size*2, plane_size*2))
#         scan_vol.append(points)
#
#     if not gaussian:
#         # Simply return the scanned volume
#         result = np.array(scan_vol)
#         result = np.swapaxes(result, 0, 2)
#         result = np.swapaxes(result, 0, 1)
#         result_vol = ScanVolume(result, (-plane_size, plane_size, -volume_depth/2) if False else (-plane_size, plane_size if right else -plane_size, -32), (pixel_size, pixel_size, interval / math.sqrt(2)),
#                                 ((-math.pi / 4) if right else (math.pi / 4), 0, 0), False)
#     else:
#         # Blur the volume
#         blurred_vol = []
#         gaussian_samples_side = (gaussian_samples-1)//2
#         weights = [gaussian_value(x, gaussian_samples_side, gaussian_samples/(2*math.sqrt(2*math.log(2)))) for x in range(gaussian_samples)]
#         for i in range(gaussian_samples_side, len(scan_vol)-gaussian_samples_side):
#             sheet_slices = scan_vol[i-gaussian_samples_side:i+gaussian_samples_side+1]
#             blurred_slice = np.average(sheet_slices, axis=0, weights=weights)
#             blurred_vol.append(blurred_slice)
#
#         result = np.array(blurred_vol)
#         result = np.swapaxes(result, 0, 2)
#         result = np.swapaxes(result, 0, 1)
#
#         # FIXME: The origin is incorrect (the first few slices can't be sampled)
#         result_vol = ScanVolume(result, (-plane_size, -plane_size, -volume_depth/2), (pixel_size, pixel_size, interval / math.sqrt(2)),
#                                 ((-math.pi / 4) if right else (math.pi / 4), 0, 0), False)
#
#     return result_vol

def fwhm_to_sigma(fwhm):
    return fwhm / (2 * math.sqrt(2 * math.log(2)))


def scan(vol: Samplable, right: bool, pixel_size: float = 0.2,
         interval: float = 0.3, plane_size=12, gaussian: bool = True,
         gaussian_delta: float = 0.05, gaussian_FWHM: float = 1) -> dispim.Volume:
    v2 = (np.array([0, math.sqrt(2) / 2, -math.sqrt(2) / 2]) if right else
            np.array([0, math.sqrt(2) / 2, math.sqrt(2) / 2]))
    scan_vol = []
    start = np.array([-plane_size, -plane_size*(math.sqrt(2)/2), 0])
    volume_depth = 68.0

    if gaussian:
        v3 = np.cross(np.array([1, 0, 0]), v2)
        slices_side = gaussian_FWHM / gaussian_delta
        sigma = fwhm_to_sigma(gaussian_FWHM)
        weights = [gaussian_value(x, 0, sigma) for x in
                   np.linspace(-slices_side * gaussian_delta, slices_side * gaussian_delta, int(slices_side * 2 + 1))]

    for z in np.linspace(-volume_depth/2, volume_depth/2, int(volume_depth/interval)):
        if not gaussian:
            points = vol.sample_plane(start + np.array([0, 0, z]),
                                       np.array([1, 0, 0]),
                                       v2, pixel_size, (plane_size*2, plane_size*2))
            scan_vol.append(points)
        else:
            sheet_points = []
            for i in np.linspace(-slices_side*gaussian_delta, slices_side*gaussian_delta, int(slices_side*2+1)):
                shifted_start = start + v3*i
                points = vol.sample_plane(shifted_start + np.array([0, 0, z]),
                                          np.array([1, 0, 0]),
                                          v2, pixel_size, (plane_size * 2, plane_size * 2))

                sheet_points.append(points)

            blurred_slice = np.average(sheet_points, axis=0, weights=weights)
            scan_vol.append(blurred_slice)

    # Simply return the scanned volume
    result = np.array(scan_vol)
    result = np.swapaxes(result, 0, 2)
    result = np.swapaxes(result, 0, 1)
    # result_vol = VirtualVolume(result, (-plane_size, plane_size, -volume_depth / 2) if False else (-plane_size, plane_size if right else -plane_size, -32), (pixel_size, pixel_size, interval / math.sqrt(2)),
    #                            ((-math.pi / 4) if right else (math.pi / 4), 0, 0), False)
    result_vol = dispim.Volume(result, (pixel_size, pixel_size, interval / math.sqrt(2)))
    return result_vol

def unshift(vol: VirtualVolume, invert=False):
    new_data = dispim.unshift(vol.data, vol.resolution[0], vol.resolution[2] * math.sqrt(2), invert=invert)
    new_vol = vol.update_data(new_data)
    return new_vol

def plot3d(data, spacing):
    field = mlab.pipeline.scalar_field(data)
    field.spacing = spacing
    fig = mlab.pipeline.volume(field, vmin=0, vmax=0.8)

def save_tiff(data, name: str, invert=False):
    if os.path.exists(name):
        shutil.rmtree(name)
    os.makedirs(name)
    if invert:
        data = np.swapaxes(data, 0, 2)
        data = np.flipud(data)
        data = np.swapaxes(data, 0, 2)
    for slc_index in range(data.shape[2]):
        imsave(name+'/'+name+str(slc_index)+'.tiff', (data[:, :, slc_index]*(2**8-1)).astype('uint8').T)


def test():
    # vol = SampleSphere(4)
    vol = SampleBeads(32)
    # vol = SampleEllipsoid((3, 3, 7))

    # vol.sample_plane(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, math.sqrt(2), math.sqrt(2)]), 1, (10, 10))

    print("Scanning from A")
    scan_vol_A = scan(vol, False, gaussian=True, gaussian_FWHM=2.5)
    scan_vol_A.save_tiff('test_A')
    print("Deskewing A")
    deskewed_A = unshift(scan_vol_A)
    # print(vol.compute_similarity(deskewed_A))
    deskewed_A.save_tiff('test_A_deskew')
    print("Calculating cof for A")
    com_A = deskewed_A.get_center_of_mass()

    print("Scanning from B")
    scan_vol_B = scan(vol, True, gaussian=True, gaussian_FWHM=2.5)
    scan_vol_B.save_tiff('test_B')
    print("Deskewing B")
    deskewed_B = unshift(scan_vol_B, invert=True)
    # print(vol.compute_similarity(deskewed_B))
    deskewed_B.save_tiff('test_B_deskew')
    deskewed_B = deskewed_B.rot90()
    com_B = deskewed_B.get_center_of_mass()

    shift = com_B - com_A

    print(deskewed_A.data.dtype)
    print(deskewed_A.data.shape)
    print(deskewed_B.data.shape)
    zoomedA, zoomedB = dispim.align_volumes(deskewed_A.data, deskewed_B.data, shift, deskewed_A.resolution[0], deskewed_A.resolution[2]*math.sqrt(2))
    print(zoomedA.shape)
    print(zoomedB.shape)

    sigma = fwhm_to_sigma(2.5)
    trunc = 2 * 2.5 / sigma
    blurA = lambda vol: scipy.ndimage.filters.gaussian_filter(vol, (0, 0, sigma), truncate=trunc)
    blurB = lambda vol: scipy.ndimage.filters.gaussian_filter(vol, (0, sigma, 0), truncate=trunc)

    result = dispim.deconvolve(zoomedA, zoomedB, 6, blurA, blurB)

    save_tiff(result, 'deconv')

    save_tiff(zoomedA, 'align_A')
    save_tiff(zoomedB, 'align_B')
    save_tiff((zoomedA+zoomedB)/2.0, 'fused')

    # trans = np.array([
    #     [1/deskewed_B.resolution[0], 0, 0, 0],
    #     [0, 1/deskewed_B.resolution[1], 0, 0],
    #     [0, 0, 1/deskewed_B.resolution[2], 0],
    #     [0, 0, 0, 1],
    # ])
    # trans = trans @ np.array([
    #     [1, 0, 0, shift[0]],
    #     [0, 1, 0, shift[1]],
    #     [0, 0, 1, shift[2]],
    #     [0, 0, 0,        1]
    # ])
    # trans = trans @ np.array([
    #     [deskewed_A.resolution[0], 0, 0, 0],
    #     [0, deskewed_A.resolution[1], 0, 0],
    #     [0, 0, deskewed_A.resolution[2], 0],
    #     [0, 0, 0, 1],
    # ])
    #
    # # fused = dispim.fuse_basic(deskewed_A.data, deskewed_B.data, np.linalg.inv(deskewed_B.trans_mat) @ deskewed_A.trans_mat)
    # fused = dispim.fuse_basic(deskewed_A.data, deskewed_B.data, deskewed_A.resolution[0], deskewed_A.resolution[2] * math.sqrt(2), trans)
    # save_tiff(fused, 'fused')


if __name__ == '__main__':
    test()
