#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from typing import List, Tuple, Union

import numpy as np
import scipy.ndimage

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
    def __init__(self, invert_a=False, invert_b=True):
        super().__init__()
        self.accepts_single = True
        print(invert_a)
        print(invert_b)
        self.invert_a = invert_a
        self.invert_b = invert_b

    def process(self, data: ProcessData) -> ProcessData:
        if len(data) == 2:
            return dispim.unshift_fast(data[0], self.invert_a), dispim.unshift_fast(data[1], self.invert_b)
        else:
            return dispim.unshift_fast(data[0], invert=self.invert_a),
            # return dispim.unshift(data[0], self.invert_a),


class ProcessRegister(ProcessStep):
    def process(self, data: ProcessData) -> ProcessData:
        # vol_b = data[1].rot90()
        return dispim.register_dipy(data[0], data[1],
                                    init_translation=dispim.register_manual_translation(data[0], data[1]))


class ProcessFuse(ProcessStep):
    def __init__(self):
        super().__init__()

    def process(self, data: ProcessData) -> ProcessData:
        if np.all(data[0].resolution != data[1].resolution):
            logger.error('Both volumes must have equal resolution to deconvolve. ')
        if np.all(data[0].data.shape != data[1].data.shape):
            logger.error('Both volumes must have equal dimensions to deconvolve. ')

        logger.info("Resolution A: {}, Resolution B: {}".format(data[0].resolution, data[1].resolution))

        return dispim.fuse(data[0], data[1]),


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


class ProcessDeconvolve(ProcessStep):
    def __init__(self, sigma: float, iters: int = 24):
        super().__init__()
        self.sigma = sigma
        self.iters = iters

    def process(self, data: ProcessData) -> ProcessData:
        if np.all(data[0].resolution != data[1].resolution):
            logger.error('Both volumes must have equal resolution to deconvolve. ')
        if np.all(data[0].data.shape != data[1].data.shape):
            logger.error('Both volumes must have equal dimensions to deconvolve. ')

        logger.info("Resolution A: {}, Resolution B: {}".format(data[0].resolution, data[1].resolution))

        sigma_a = self.sigma
        sigma_b = self.sigma
        # FIXME: Fix blur B
        if self.sigma == 0:
            sigma_a, _ = dispim.compute_psf(data[0])
            logger.info(sigma_a)
            # sigma_b, _ = dispim.compute_psf(data[1])

        logger.info('Deconvolving...')
        blur_a = lambda vol: scipy.ndimage.filters.gaussian_filter1d(vol, sigma_a / data[0].resolution[2], axis=2)
        # blur_b = lambda vol: scipy.ndimage.filters.gaussian_filter1d(vol, sigma_b/data[1].resolution[1], axis=1)

        return dispim.deconvolve(data[0], data[1], self.iters, blur_a, blur_a),


class ProcessDeconvolveSeparate(ProcessStep):
    def __init__(self, sigma: float, iters: int = 24):
        super().__init__()
        self.accepts_single = True
        self.sigma = sigma
        self.iters = iters

    def process(self, data: ProcessData) -> ProcessData:
        if len(data) == 2:
            sigma_a = self.sigma
            sigma_b = self.sigma
            if self.sigma == 0:
                logger.info("Extracting gaussian psf...")
                sigma_a, _ = dispim.compute_psf(data[0])
                sigma_b, _ = dispim.compute_psf(data[1])

            blur_a = lambda vol: scipy.ndimage.filters.gaussian_filter1d(vol, sigma_a / data[0].resolution[2], axis=2)
            blur_b = lambda vol: scipy.ndimage.filters.gaussian_filter1d(vol, sigma_b / data[1].resolution[2], axis=2)

            logger.info('Deconvolving using rl...')
            return dispim.deconvolve_rl(data[0], self.iters, blur_a), dispim.deconvolve_rl(data[1], self.iters, blur_b)
        else:
            sigma = self.sigma
            if self.sigma == 0:
                logger.info("Extracting gaussian psf...")
                sigma, _ = dispim.compute_psf(data[0])
            # blur = lambda vol: scipy.ndimage.filters.gaussian_filter1d(vol, sigma/data[0].resolution[2], axis=2)
            blur = lambda vol: scipy.ndimage.filters.gaussian_filter1d(vol, sigma / data[0].resolution[2], axis=2)
            logger.info('Deconvolving using rl...')
            return dispim.deconvolve_rl(data[0], self.iters, blur),


class ProcessCenterCrop(ProcessStep):
    def __init__(self, crop_value):
        super().__init__()
        self.accepts_single = True
        self.crop_value = crop_value

    def process(self, data: ProcessData) -> ProcessData:
        result = []
        for vol in data:
            crop_offset = np.multiply(vol.data.shape, self.crop_value) // 2
            crop_offset = crop_offset.astype(np.int)
            center = np.floor_divide(vol.data.shape, 2)
            result.append(Volume(vol.data[center[0] - crop_offset[0]:center[0] + crop_offset[0],
                                 center[1] - crop_offset[1]:center[1] + crop_offset[1],
                                 center[2] - crop_offset[2]:center[2] + crop_offset[2]], vol.resolution))

        return tuple(result)


class ProcessScale(ProcessStep):
    def __init__(self, scale_value):
        super().__init__()
        self.accepts_single = True
        self.scale_value = scale_value

    def process(self, data: ProcessData) -> ProcessData:
        result = []
        for vol in data:
            result.append(scipy.ndimage.zoom(vol.data, self.scale_value))

        return tuple(result)


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

    def process(self, data: ProcessData, save_intermediate=False) -> ProcessData:
        import gc, sys
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

            for var, obj in locals().items():
                if sys.getsizeof(obj) > 1024 * 1024 * 512:
                    print(var, obj)

            data = step.process(data)
            gc.collect()

            if save_intermediate:
                data[0].save_tiff(step.__class__.__name__ + "_A")
                if len(data) > 1:
                    data[1].save_tiff(step.__class__.__name__ + "_B")

        return data
