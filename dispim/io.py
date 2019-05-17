import logging
import os
import shutil
from typing import Tuple, Union

import numpy as np
import tifffile

import dispim.base
from dispim import Volume

logger = logging.getLogger(__name__)


def save_tiff_output(data: dispim.base.Volume, path: str, name: str, b_8bit: bool = False) -> None:
    """
    Save a volume as a single TIFF file to an output directory (created if necessary)
    
    :param data: The volume to save
    :param path: The path of the output directory
    :param name: The name under which the volume should be saved
    :param b_8bit: Whether to save the data as uint8 instead of uint16
    """
    generate_output_dir(path)
    out_path = os.path.join(path, f"{name}.tif")

    save_tiff(data, out_path)


def save_tiff_output_dual(vol_a: Volume, vol_b: Volume, path: str, name: str, b_8bit: bool = False):
    """
    Save two volumes as a single TIFF file to an output directory (created if necessary)

    :param vol_a: The first volume (first channel)
    :param vol_b: The second volume (second channel)
    :param path: The path of the output directory
    :param name: The name under which the volumes should be saved
    :param b_8bit: Whether to save the data as uint8 instead of uint16
    """
    generate_output_dir(path)
    out_path = os.path.join(path, f"{name}.tif")

    save_tiff_dual(vol_a, vol_b, out_path)


def generate_output_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def save_tiff(data: Volume, path: str, b_8bit: bool = False):
    # TODO: Deal with large fractions more robustly
    with tifffile.TiffWriter(path, bigtiff=True) as f:
        f.save(np.moveaxis(data if not b_8bit else data / 2 ** 8, [2, 0, 1], [0, 1, 2]), resolution=(
        np.round(1 / (data.spacing[0] / 10000), decimals=5), np.round(1 / (data.spacing[1] / 10000), decimals=5),
        'CENTIMETER'))


def save_tiff_dual(vol_a: Volume, vol_b: Volume, path: str, b_8bit: bool = False):
    # TODO: Deal with large fractions more robustly
    data = np.array([
        vol_a, vol_b, np.zeros_like(vol_a)
    ])
    with tifffile.TiffWriter(path, bigtiff=True) as f:
        f.save(np.moveaxis(data if not b_8bit else data / 2 ** 8, [0, 3, 1, 2], [1, 0, 2, 3]), resolution=(
        np.round(1 / (vol_a.spacing[0] / 10000), decimals=5), np.round(1 / (vol_b.spacing[1] / 10000), decimals=5),
        'CENTIMETER'))


def save_tiff_chunks(vol: Volume, path: str, size: int, stride: int, b_8bit: bool = False):
    import shutil
    import pathlib

    if os.path.isdir(path):
        shutil.rmtree(path)

    pathlib.Path(path).mkdir(parents=True)

    i = 0
    for x in range(0, vol.shape[0] - size + 1, stride):
        for y in range(0, vol.shape[1] - size + 1, stride):
            for z in range(0, vol.shape[2] - size + 1, stride):
                data = vol[x:x + size, y:y + size, z:z + size]
                chunk_path = os.path.join(path, f'{i}.tif')

                with tifffile.TiffWriter(chunk_path, bigtiff=False) as f:
                    f.save(np.moveaxis(data if not b_8bit else data / 2 ** 8, [2, 0, 1], [0, 1, 2]), resolution=(
                        np.round(1 / (data.spacing[0] / 10000), decimals=5),
                        np.round(1 / (data.spacing[1] / 10000), decimals=5), 'CENTIMETER'))

                i += 1


def load_tiff(path: str, series: int = 0, channel: int = 0, inverted: bool = False,
              flipped: Tuple[bool] = (False, False, False), pixel_size: float = None,
              step_size: float = None) -> Union[Tuple[Volume, Volume], Tuple[Volume]]:
    import json

    with tifffile.TiffFile(path) as f:
        data = f.asarray(series=series)
        logger.debug(f'Data shape is {data.shape}')
        # TODO: Figure out how to properly handle the axis order
        if data.ndim == 3:
            data = np.moveaxis(data, [0, 1, 2], [2, 0, 1])
        elif data.ndim == 4:
            data = np.moveaxis(data, [0, 1, 2, 3], [3, 0, 1, 2])[2 * channel:2 * channel + 2]
        else:
            raise ValueError(f'Invalid data shape: {data.shape}')
        # TODO: Automatically extract metadata

        if pixel_size is None:
            pixel_size = f.pages[0].tags['XResolution'].value[1] / f.pages[0].tags['XResolution'].value[0]
            if f.pages[0].tags['YResolution'].value[1] / f.pages[0].tags['YResolution'].value[0] != pixel_size:
                raise ValueError(
                    f'X and Y resolution differ in metadata ({pixel_size}x{f.pages[0].tags["YResolution"].value[1] / f.pages[0].tags["YResolution"].value[0]})')
            if f.pages[0].tags['ResolutionUnit'].value != 3:
                raise ValueError(f'Unsupported resolution unit: {f.pages[0].tags["ResolutionUnit"].value}')

            pixel_size *= 10000

        if step_size is None:
            try:
                step_size = json.load(f.micromanager_metadata['Summary']['SPIMAcqSettings'])['stepSizeUm']
            except (ValueError, TypeError):
                raise ValueError('No stage step size specified and metadata cannot be accessed')

    if data.ndim == 3:
        if flipped[0]: data = data[::-1, :, :]
        if flipped[1]: data = data[:, ::-1, :]
        if flipped[2]: data = data[:, :, ::-1]

        return Volume(data, spacing=(pixel_size, pixel_size, step_size), is_skewed=True, flipped=flipped,
                      inverted=inverted)
    else:
        if flipped[0]: data[1] = data[1, ::-1, :, :]
        if flipped[1]: data[1] = data[1, :, ::-1, :]
        if flipped[2]: data[1] = data[1, :, :, ::-1]

        return (
            Volume(data[0], spacing=(pixel_size, pixel_size, step_size), is_skewed=True,
                   inverted=inverted),
            Volume(data[1], spacing=(pixel_size, pixel_size, step_size), is_skewed=True, flipped=flipped,
                   inverted=True)
        )


def print_ome_info(path: str):
    with tifffile.TiffFile(path) as f:
        if not f.is_ome:
            raise ValueError(f'{path} is not an OME-TIFF')

        ome_meta = f.ome_metadata
        print('Channels: ')
        print(ome_meta['Image']['Pixels']['Channel'])

        print(f'Channel order: {ome_meta["Image"]["Pixels"]["Channel"]}')

        print(f"XResolution: {f.pages[0].tags['XResolution'].value}")
        print(f"YResolution: {f.pages[0].tags['YResolution'].value}")


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
