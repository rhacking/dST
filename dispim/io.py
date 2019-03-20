import logging
import os
from typing import Tuple, Union

import numpy as np
import tifffile

import dispim
from dispim import Volume

logger = logging.getLogger(__name__)


def save_tiff_output(data: dispim.Volume, path: str, name: str, b_8bit: bool = False):
    generate_output_dir(path)
    out_path = os.path.join(path, f"{name}.tif")

    save_tiff(data, out_path)


def save_tiff_output_dual(vol_a: Volume, vol_b: Volume, path: str, name: str, b_8bit: bool = False):
    generate_output_dir(path)
    out_path = os.path.join(path, f"{name}.tif")

    save_tiff_dual(vol_a, vol_b, out_path)


def generate_output_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def save_tiff(data: Volume, path: str, b_8bit: bool = False):
    with tifffile.TiffWriter(path) as f:
        f.save(np.moveaxis(data if not b_8bit else data / 2 ** 8, [2, 0, 1], [0, 1, 2]), resolution=(1/(data.spacing[0]/10000), 1/(data.spacing[1]/10000), 'CENTIMETER'))


def save_tiff_dual(vol_a: Volume, vol_b: Volume, path: str, b_8bit: bool = False):
    data = np.array([
        vol_a, vol_b, np.zeros_like(vol_a)
    ])
    with tifffile.TiffWriter(path) as f:
        f.save(np.moveaxis(data if not b_8bit else data / 2 ** 8, [0, 3, 1, 2], [1, 0, 2, 3]), resolution=(1/(vol_a.spacing[0]/10000), 1/(vol_b.spacing[1]/10000), 'CENTIMETER'))


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
                raise ValueError(f'X and Y resolution differ in metadata ({pixel_size}x{f.pages[0].tags["YResolution"].value[1]/f.pages[0].tags["YResolution"].value[0]})')
            if f.pages[0].tags['ResolutionUnit'].value != 3:
                raise ValueError(f'Unsupported resolution unit: {f.pages[0].tags["ResolutionUnit"].value}')

            pixel_size *= 10000

        if step_size is None:
            try:
                step_size = json.load(f.micromanager_metadata['Summary']['SPIMAcqSettings'])['stepSizeUm']
            except (ValueError, TypeError):
                raise ValueError('No stage step size specified and metadata cannot be accessed')

    if data.ndim == 3:
        return Volume(data, spacing=(pixel_size, pixel_size, step_size), is_skewed=True, flipped=flipped,
                      inverted=inverted)
    else:
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
