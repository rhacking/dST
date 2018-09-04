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
            return dispim.unshift_fast(data[0], self.invert_a, estimate_true_interval=False), dispim.unshift_fast(
                data[1], self.invert_b, estimate_true_interval=False)
        else:
            return dispim.unshift_fast(data[0], invert=self.invert_a, estimate_true_interval=False),
            # return dispim.unshift(data[0], self.invert_a),


class ProcessRegister(ProcessStep):
    def __init__(self, crop: float = 0.6):
        super().__init__()
        self.crop = crop

    def process(self, data: ProcessData) -> ProcessData:
        return dispim.register_dipy(data[0], data[1], crop=self.crop)


class ProcessRegister2d(ProcessStep):
    def process(self, data: ProcessData):
        return dispim.register_2d(*data)


class ProcessApplyRegistration(ProcessStep):
    def process(self, data: ProcessData) -> ProcessData:
        from scipy.ndimage import affine_transform
        print(data[0].data.shape, data[1].data.shape)
        transform = np.linalg.inv(data[1].grid_to_world)
        # data[1].world_transform[1, 3] /= -np.sqrt(2)
        transform = transform @ data[1].world_transform
        transform = transform @ data[0].grid_to_world
        # transform[2, 3] = 0
        print(np.linalg.inv(data[0].grid_to_world))
        print(data[0].inverted)
        print(data[1].inverted)
        print(data[1].world_transform)
        print(data[1].grid_to_world)
        print(transform)
        print(data[1].shape)
        transformed = affine_transform(data[1].data, (transform))
        transformed = transformed[:data[0].shape[0], :data[0].shape[1], :data[0].shape[2]]
        transformed = np.pad(transformed, ((0, np.max(data[0].shape[0] - transformed.shape[0], 0)),
                                           (0, np.max(data[0].shape[1] - transformed.shape[1], 0)),
                                           (0, np.max(data[0].shape[2] - transformed.shape[2], 0))),
                             mode='constant', constant_values=0)

        return data[0], data[1].update(transformed, inverted=False, spacing=data[0].spacing, is_skewed=False)


class ProcessRegisterSyn(ProcessStep):
    def process(self, data: ProcessData):
        return dispim.register_syn(data[0], data[1])


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
            return data[0].update((data[0].data*self.f).astype(np.uint16)), data[1].update((data[1].data*self.f).astype(np.uint16))
        else:
            return data[0].update((data[0].data*self.f).astype(np.uint16))


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
        if np.all(data[0].spacing != data[1].spacing):
            logger.error('Both volumes must have equal resolution to deconvolve. ')
        if np.all(data[0].data.shape != data[1].data.shape):
            logger.error('Both volumes must have equal dimensions to deconvolve. ')

        logger.info("Resolution A: {}, Resolution B: {}".format(data[0].spacing, data[1].spacing))

        sigma_a = self.sigma
        sigma_b = self.sigma
        # FIXME: Fix blur B
        if self.sigma == 0:
            sigma_a, _ = dispim.compute_psf(data[0])
            logger.info(sigma_a)
            # sigma_b, _ = dispim.compute_psf(data[1])

        logger.info('Deconvolving...')
        blur_a = lambda vol: scipy.ndimage.filters.gaussian_filter1d(vol, sigma_a / data[0].spacing[2], axis=2)
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

            blur_a = lambda vol: scipy.ndimage.filters.gaussian_filter1d(vol, sigma_a / data[0].spacing[2], axis=2)
            blur_b = lambda vol: scipy.ndimage.filters.gaussian_filter1d(vol, sigma_b / data[1].spacing[2], axis=2)

            logger.info('Deconvolving using rl...')
            return dispim.deconvolve_rl(data[0], self.iters, blur_a), dispim.deconvolve_rl(data[1], self.iters, blur_b)
        else:
            sigma = self.sigma
            if self.sigma == 0:
                logger.info("Extracting gaussian psf...")
                sigma, _ = dispim.compute_psf(data[0])
            # blur = lambda vol: scipy.ndimage.filters.gaussian_filter1d(vol, sigma/data[0].resolution[2], axis=2)
            blur = lambda vol: scipy.ndimage.filters.gaussian_filter1d(vol, sigma / data[0].spacing[2], axis=2)
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
                                 center[2] - crop_offset[2]:center[2] + crop_offset[2]], vol.inverted, vol.spacing))

        return tuple(result)


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

        rotated_a = rotate(data[0].data.astype(np.float), -45, axes=(0, 1), cval=-1000)
        rotated_a = rotate(rotated_a, -45, axes=(0, 2), cval=-1000)

        rotated_b = rotate(data[1].data.astype(np.float), -45, axes=(0, 1), cval=-1000)
        rotated_b = rotate(rotated_b, -45, axes=(0, 2), cval=-1000)

        max_val = max(rotated_a.max(), rotated_b.max())*0.05+1000

        import matplotlib.pyplot as plt
        img = np.stack([(np.mean(rotated_a, axis=2, dtype=np.float)+1000) / max_val,
                        (np.mean(rotated_b, axis=2, dtype=np.float)+1000) / max_val,
                        np.zeros((rotated_a.shape[0], rotated_b.shape[1]))], axis=2)
        plt.imshow(img)
        plt.show()

        return data


class ProcessShowSeperateIso(ProcessStep):
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

        return tuple(result)


class ProcessRot90(ProcessStep):
    def __init__(self, reverse: bool = True):
        super().__init__()
        self.reverse = reverse

    def process(self, data: ProcessData) -> ProcessData:
        return data[0], data[1].update(np.rot90(data[1].data, k=(3 if self.reverse else 1), axes=(1, 2)), spacing=(data[1].spacing[0], data[1].spacing[2], data[1].spacing[1]))


class ProcessMakeIsotropic(ProcessStep):
    def __init__(self):
        super().__init__()

    def process(self, data: ProcessData) -> ProcessData:
        return dispim.make_isotropic(data[0], data[1])


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
                data[0].save_tiff(step.__class__.__name__ + "_A", path=path)
                if len(data) > 1:
                    data[1].save_tiff(step.__class__.__name__ + "_B", path=path)

        return data
