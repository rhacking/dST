#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from typing import List, Tuple, Union, Callable, Optional

import numpy as np
import scipy.ndimage
from dipy.align.transforms import (TranslationTransform3D, TranslationTransform2D,
                                   RigidTransform3D, RigidTransform2D,
                                   AffineTransform3D, AffineTransform2D)

import dispim
from dispim import Volume

logger = logging.getLogger(__name__)

ProcessData = Union[Tuple[Volume, Volume], Tuple[Volume]]


class ProcessStep(object):
    def __init__(self):
        self.accepts_single = False
        self.accepts_dual = True

    def process(self, data: ProcessData) -> ProcessData:
        raise NotImplementedError()


class ProcessDeskew(ProcessStep):
    def __init__(self, invert_a=False, invert_b=True, estimate_true_interval=False):
        super().__init__()
        self.accepts_single = True
        print(invert_a)
        print(invert_b)
        self.invert_a = invert_a
        self.invert_b = invert_b
        self.estimate_true_interval = estimate_true_interval

    def process(self, data: ProcessData) -> ProcessData:
        if len(data) == 2:
            return dispim.unshift_fast(data[0], self.invert_a, estimate_true_interval=self.estimate_true_interval), dispim.unshift_fast(
                data[1], self.invert_b, estimate_true_interval=self.estimate_true_interval)
        else:
            return dispim.unshift_fast(data[0], invert=self.invert_a, estimate_true_interval=self.estimate_true_interval),
            # return dispim.unshift(data[0], self.invert_a),


class ProcessDeskewDiag(ProcessStep):
    def __init__(self, invert_a=False, invert_b=True, estimate_true_interval=False):
        super().__init__()
        self.accepts_single = True
        print(invert_a)
        print(invert_b)
        self.invert_a = invert_a
        self.invert_b = invert_b
        self.estimate_true_interval = estimate_true_interval

    def process(self, data: ProcessData) -> ProcessData:
        if len(data) == 2:
            return dispim.unshift_fast_diag(data[0], self.invert_a, estimate_true_interval=self.estimate_true_interval), dispim.unshift_fast_diag(
                data[1], self.invert_b, estimate_true_interval=self.estimate_true_interval)
        else:
            return dispim.unshift_fast_diag(data[0], invert=self.invert_a, estimate_true_interval=self.estimate_true_interval),
            # return dispim.unshift(data[0], self.invert_a),


class ProcessRegisterCom(ProcessStep):
    def __init__(self):
        super().__init__()

    def process(self, data: ProcessData) -> ProcessData:
        return dispim.register_com(data[0], data[1])


class ProcessRegister(ProcessStep):
    type_mapping = {
        'translation': TranslationTransform3D,
        'rigid': RigidTransform3D,
        'affine': AffineTransform3D
    }

    def __init__(self, transform_type: str = 'translation', sampling_prop: float = 1.0, crop: float = 1.0):
        super().__init__()
        self.transform_cls = self.type_mapping[transform_type]
        self.crop = crop
        self.sampling_prop = sampling_prop

    def process(self, data: ProcessData) -> ProcessData:
        return dispim.register_dipy(data[0], data[1], sampling_prop=self.sampling_prop, crop=self.crop,
                                    transform_cls=self.transform_cls)


class ProcessRegister2d(ProcessStep):
    type_mapping = {
        'translation': TranslationTransform2D,
        'rigid': RigidTransform2D,
        'affine': AffineTransform2D
    }

    def __init__(self, axis=2, transform_type: str = 'translation'):
        super().__init__()
        self.axis = int(axis)
        self.transform_cls = self.type_mapping[transform_type]

    def process(self, data: ProcessData):
        return dispim.register_2d(*data, axis=self.axis, transform_cls=self.transform_cls)


class ProcessApplyRegistration(ProcessStep):
    def __init__(self, order: int = 2):
        super().__init__()
        self.order = order

    def process(self, data: ProcessData) -> ProcessData:
        return dispim.apply_registration(data[0], data[1], order=self.order)


class ProcessFuse(ProcessStep):
    def __init__(self):
        super().__init__()

    def process(self, data: ProcessData) -> ProcessData:
        if np.all(data[0].spacing != data[1].spacing):
            logger.error('Both volumes must have equal resolution to deconvolve. ')
        if np.all(data[0].data.shape != data[1].data.shape):
            logger.error('Both volumes must have equal dimensions to deconvolve. ')

        logger.info("Resolution A: {}, Resolution B: {}".format(data[0].spacing, data[1].spacing))

        return dispim.fuse(data[0], data[1]),


class ProcessBrighten(ProcessStep):
    def __init__(self, f):
        super().__init__()
        self.f = f
        self.accepts_single = True

    def process(self, data: ProcessData):
        if len(data) == 2:
            return data[0].update((data[0].data * self.f).astype(np.uint16)), data[1].update(
                (data[1].data * self.f).astype(np.uint16))
        else:
            return data[0].update((data[0].data * self.f).astype(np.uint16)),


class ProcessDiscardA(ProcessStep):
    def __init__(self):
        super().__init__()

    def process(self, data: ProcessData) -> ProcessData:
        return data[1],


class ProcessDiscardB(ProcessStep):
    def __init__(self):
        super().__init__()

    def process(self, data: ProcessData) -> ProcessData:
        return data[0],


class ProcessExtractPsf(ProcessStep):
    def __init__(self, min_size: int = 25, max_size: int = 90, psf_half_width: int = 5):
        super().__init__()
        self.min_size = int(min_size)
        self.max_size = int(max_size)
        self.psf_half_width = int(psf_half_width)

    def process(self, data: ProcessData):
        psf_a = dispim.extract_psf(data[0], self.min_size, self.max_size, self.psf_half_width)
        psf_b = dispim.extract_psf(data[1], self.min_size, self.max_size, self.psf_half_width)

        return Volume(data[0], psf=psf_a), Volume(data[1], psf=psf_b)


class ProcessDeconvolve(ProcessStep):
    def __init__(self, iters: int = 24, psf_a: str = None, psf_b: str = None):
        super().__init__()
        self.psf_A = psf_a
        self.psf_B = psf_b
        self.iters = int(iters)

    def process(self, data: ProcessData) -> ProcessData:
        if np.all(data[0].spacing != data[1].spacing):
            logger.error('Both volumes must have equal resolution to deconvolve. ')
        if np.all(data[0].data.shape != data[1].data.shape):
            logger.error('Both volumes must have equal dimensions to deconvolve. ')

        logger.info("Resolution A: {}, Resolution B: {}".format(data[0].spacing, data[1].spacing))

        from tifffile import imread

        logger.info('Deconvolving...')
        # blur_a = lambda vol: scipy.ndimage.filters.gaussian_filter1d(vol, sigma_a / data[0].spacing[2], axis=2)
        # blur_b = lambda vol: scipy.ndimage.filters.gaussian_filter1d(vol, sigma_b/data[1].resolution[1], axis=1)

        if self.psf_A is None:
            if data[0].psf is None:
                raise ValueError("No point spread function specified for A")
            psf_A = data[0].psf
        else:
            psf_A = imread(self.psf_A).swapaxes(0, 2).swapaxes(0, 1)

        if self.psf_B is None:
            if data[1].psf is None:
                raise ValueError("No point spread function specified for B")
            psf_B = data[1].psf
        else:
            psf_B = imread(self.psf_B).swapaxes(0, 2).swapaxes(0, 1)

        return dispim.deconvolve(data[0], data[1], self.iters, psf_A, psf_B),


class ProcessDeconvolveChunked(ProcessStep):
    def __init__(self, iters: int = 24, nchunks: int = 3, blind: bool = False, psf_a: str = None, psf_b: str = None):
        super().__init__()
        self.psf_A = psf_a
        self.psf_B = psf_b
        self.nchunks = int(nchunks)
        self.iters = int(iters)
        self.blind = blind

    def process(self, data: ProcessData) -> ProcessData:
        if np.all(data[0].spacing != data[1].spacing):
            logger.error('Both volumes must have equal resolution to deconvolve. ')
        if np.all(data[0].data.shape != data[1].data.shape):
            logger.error('Both volumes must have equal dimensions to deconvolve. ')

        logger.info("Resolution A: {}, Resolution B: {}".format(data[0].spacing, data[1].spacing))

        from tifffile import imread

        logger.info('Deconvolving...')

        if self.psf_A is None:
            if data[0].psf is None:
                raise ValueError("No point spread function specified for A")
            psf_a = data[0].psf
        else:
            psf_a = imread(self.psf_A).swapaxes(0, 2).swapaxes(0, 1)

        if self.psf_B is None:
            if data[1].psf is None:
                raise ValueError("No point spread function specified for B")
            psf_b = data[1].psf
        else:
            psf_b = imread(self.psf_B).swapaxes(0, 2).swapaxes(0, 1)

        return dispim.deconvolve_gpu_chunked(data[0], data[1], self.iters, psf_a, psf_b, nchunks=self.nchunks, blind=self.blind),


class ProcessSaveChunks(ProcessStep):
    def __init__(self, output: str, size: int = 64, stride: Optional[int] = None):
        super().__init__()
        self.output = output
        self.size = size
        self.stride = stride if stride is not None else size
        self.accepts_single = True

    def process(self, data: ProcessData) -> ProcessData:
        import dispim.io as dio
        import os

        if len(data) == 2:
            dio.save_tiff_chunks(data[0], os.path.join(self.output, 'A'), self.size, self.stride)
            dio.save_tiff_chunks(data[1], os.path.join(self.output, 'B'), self.size, self.stride)
        else:
            dio.save_tiff_chunks(data[0], self.output, self.size, self.stride)

        return data


class ProcessCenterCrop(ProcessStep):
    def __init__(self, crop_value: Union[float, Tuple[float, float, float]]):
        super().__init__()
        self.accepts_single = True
        self.crop_value = crop_value

    def process(self, data: ProcessData) -> ProcessData:
        from dispim.util import crop_view
        result = []
        for vol in data:
            result.append(crop_view(vol, self.crop_value))

        # noinspection PyTypeChecker
        return tuple(result)


class ProcessShowFront(ProcessStep):
    def __init__(self):
        super().__init__()
        self.accepts_single = True

    def process(self, data: ProcessData):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(np.sum(data[0].data, axis=2))
        ax[1].imshow(np.sum(data[1].data, axis=2))
        plt.show()

        return data


class ProcessShowAIso(ProcessStep):
    def __init__(self):
        super().__init__()
        self.accepts_single = True

    def process(self, data: ProcessData):
        from scipy.ndimage import rotate

        rotated = rotate(data[0].data, -45, axes=(0, 1))
        rotated = rotate(rotated, -45, axes=(0, 2))

        import matplotlib.pyplot as plt
        plt.imshow(np.sum(rotated, axis=2))
        plt.show()

        return data


class ProcessShowBIso(ProcessStep):
    def __init__(self):
        super().__init__()

    def process(self, data: ProcessData):
        from scipy.ndimage import rotate

        # rotated = rotate(data[1].data, 180, axes=(1, 2))
        # rotated = rotate(rotated, 45, axes=(0, 1))
        # rotated = rotate(rotated, 45, axes=(0, 2))

        rotated = rotate(data[1].data, -45, axes=(0, 1))
        rotated = rotate(rotated, -45, axes=(0, 2))

        import matplotlib.pyplot as plt
        plt.imshow(np.sum(rotated, axis=2))
        plt.show()

        return data


class ProcessShowOverlayIso(ProcessStep):
    def __init__(self):
        super().__init__()

    def process(self, data: ProcessData):
        from scipy.ndimage import rotate

        min_a = np.percentile(data[0].data, 2)
        min_b = np.percentile(data[1].data, 2)

        rotated_a = rotate(data[0].data.astype(np.float), -45, axes=(0, 1), cval=min_a, order=0)
        rotated_a = rotate(rotated_a, -45, axes=(0, 2), cval=min_a, order=0)

        rotated_b = rotate(data[1].data.astype(np.float), -45, axes=(0, 1), cval=min_b, order=0)
        rotated_b = rotate(rotated_b, -45, axes=(0, 2), cval=min_b, order=0)

        max_val = max(np.percentile(rotated_a, 95), np.percentile(rotated_b, 95))
        min_val = min(np.percentile(rotated_a, 5), np.percentile(rotated_b, 5))

        import matplotlib.pyplot as plt
        img = np.stack([(np.mean(rotated_a, axis=2, dtype=np.float) - min_val) / (max_val - min_val),
                        (np.mean(rotated_b, axis=2, dtype=np.float) - min_val) / (max_val - min_val),
                        np.zeros((rotated_a.shape[0], rotated_b.shape[1]))], axis=2)
        plt.imshow(img)
        plt.show()

        return data


class ProcessShowSeparateIso(ProcessStep):
    def __init__(self):
        super().__init__()

    def process(self, data: ProcessData):
        from scipy.ndimage import rotate

        rotated_a = rotate(data[0].data.astype(np.float), -45, axes=(0, 1), cval=-1000)
        rotated_a = rotate(rotated_a, -45, axes=(0, 2), cval=-1000)

        rotated_b = rotate(data[1].data.astype(np.float), -45, axes=(0, 1), cval=-1000)
        rotated_b = rotate(rotated_b, -45, axes=(0, 2), cval=-1000)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(np.sum(rotated_a, axis=2))
        ax[0].set_title('Volume A')
        ax[1].imshow(np.sum(rotated_b, axis=2))
        ax[1].set_title('Volume B')
        plt.show()

        return data


class ProcessScale(ProcessStep):
    def __init__(self, scale_value):
        super().__init__()
        self.accepts_single = True
        self.scale_value = scale_value

    def process(self, data: ProcessData) -> ProcessData:
        result = []
        for vol in data:
            result.append(scipy.ndimage.zoom(vol.data, self.scale_value))

        # noinspection PyTypeChecker
        return tuple(result)


class ProcessRot90(ProcessStep):
    def __init__(self, reverse: bool = True):
        super().__init__()
        self.reverse = reverse

    def process(self, data: ProcessData) -> ProcessData:
        return data[0], data[1].update(np.rot90(data[1].data, k=(3 if self.reverse else 1), axes=(1, 2)),
                                       spacing=(data[1].spacing[0], data[1].spacing[2], data[1].spacing[1]))


class ProcessMakeIsotropic(ProcessStep):
    def __init__(self):
        super().__init__()
        self.accepts_single = True

    def process(self, data: ProcessData) -> ProcessData:
        if len(data) == 2:
            return dispim.make_isotropic(data[0], data[1])
        else:
            return dispim.make_isotropic(data[0], Volume(np.empty((1, 1, 1)), False, (1, 1, 1)))[0],


class ProcessShowDual(ProcessStep):
    def __init__(self):
        super().__init__()

    def process(self, data: ProcessData) -> ProcessData:
        import matplotlib.pyplot as plt
        img = np.stack([np.sum(data[0].data, axis=2, dtype=np.float) / 2 ** 16,
                        np.sum(data[1].data, axis=2, dtype=np.float) / 2 ** 16,
                        np.zeros((data[0].data.shape[0], data[0].data.shape[1]))], axis=2)
        plt.imshow(img)
        plt.show()
        return data


class ProcessShowSliceYZ(ProcessStep):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def process(self, data: ProcessData) -> ProcessData:
        import matplotlib.pyplot as plt
        plt.ion()
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(data[0].data[self.index, :, :])
        ax[1].imshow(data[1].data[self.index, :, :])
        plt.show()
        return data


class Processor(object):
    def __init__(self, steps: List[ProcessStep]):
        self.steps = steps

    def process(self, data: ProcessData, path: str, save_intermediate=False) -> ProcessData:
        import gc
        from dispim.metrics import PROCESS_TIME
        from dispim import metrack
        import time
        with metrack.Context('Processor'):
            for i, step in enumerate(self.steps):
                # TODO: Check this BEFORE processing...
                logger.info("Performing step {} on {} data".format(step.__class__.__name__,
                                                                   "dual" if len(data) == 2 else "single"))
                if ((not step.accepts_dual and len(data) == 2) or
                        (not step.accepts_single and len(data) == 1)):
                    if i > 0:
                        raise ValueError('Step {} is incompatible with the output of step {}'
                                         .format(step.__class__.__name__, self.steps[i - 1].__class__.__name__))
                    else:
                        raise ValueError("Step {} is incompatible with the input data"
                                         .format(step.__class__.__name__))

                start = time.time()

                with metrack.Context(f'{step.__class__.__name__} ({i})'):
                    data = step.process(data)

                end = time.time()

                metrack.append_metric(PROCESS_TIME, (step.__class__.__name__, end-start))

                gc.collect()

                if save_intermediate:
                    data[0].save_tiff(step.__class__.__name__ + "_A", path=path)
                    if len(data) > 1:
                        data[1].save_tiff(step.__class__.__name__ + "_B", path=path)

        return data
