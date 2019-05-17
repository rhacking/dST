import logging
from typing import Union, Tuple

import math
import numpy as np

logger = logging.getLogger(__name__)


class Volume(np.ndarray):
    def __new__(cls, input_array, inverted: bool = None,
                spacing: Union[Tuple[float, float, float], np.ndarray] = None,
                is_skewed: bool = None, flipped: Tuple[bool] = None,
                world_transform: np.ndarray = None, psf: np.ndarray = None):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__Ø
        obj = np.asarray(input_array).view(cls)
        # set the new 'info' attribute to the value passed
        obj.inverted = inverted if inverted is not None else getattr(input_array, 'inverted', False)
        obj.spacing = spacing if spacing is not None else getattr(input_array, 'spacing', (1, 1, 1))
        obj.is_skewed = is_skewed if is_skewed is not None else getattr(input_array, 'is_skewed', False)
        obj.flipped = flipped if flipped is not None else getattr(input_array, 'flipped', (False, False, False))
        obj.world_transform = world_transform if world_transform is not None else getattr(input_array,
                                                                                          'world_transform', np.eye(4))
        obj.psf = psf if psf is not None else getattr(input_array, 'psf', None)

        obj.flags.writeable = False
        obj.initialized = True
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(InfoArray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. InfoArray():
        #    obj is None
        #    (we're in the middle of the InfoArray.__new__
        #    constructor, and self.info will be set when we return to
        #    InfoArray.__new__)
        if obj is None: return
        # From view casting - e.g arr.view(InfoArray):
        #    obj is arr
        #    (type(obj) can be InfoArray)
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is InfoArray
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'info', because this
        # method sees all creation of default objects - with the
        # InfoArray.__new__ constructor, but also with
        # arr.view(InfoArray).
        self.inverted = getattr(obj, 'inverted', False)
        self.spacing = getattr(obj, 'spacing', (1, 1, 1))
        self.is_skewed = getattr(obj, 'is_skewed', False)
        self.flipped = getattr(obj, 'flipped', (False, False, False))
        self.world_transform = getattr(obj, 'world_transform', np.eye(4))
        self.psf = getattr(obj, 'psf', None)
        # We do not need to return anything

    @property
    def grid_to_world(self) -> np.ndarray:
        y_shift = False
        if y_shift:
            result = np.array([
                [self.spacing[0], 0, 0, 0],
                [0, self.spacing[1], 0, 0],
                [0, 0, self.spacing[2], 0],
                [0, 0, 0, 1]
            ])

            if self.is_skewed:
                result = result @ np.array([
                    [1, 0, 0, 0],
                    [0, 1, (-1 if self.inverted else 1) * self.spacing[2] / self.spacing[1], 0],
                    [0, 0, 1,
                     0],
                    [0, 0, 0, 1]
                ])
        else:
            result = np.array([
                [self.spacing[0], 0, 0, 0],
                [0, self.spacing[1] / (np.sqrt(2) if self.is_skewed else 1), 0, 0],
                [0, 0, self.spacing[2] * (np.sqrt(2) if self.is_skewed else 1), 0],
                [0, 0, 0, 1]
            ])

            if self.is_skewed:
                result = result @ np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0,
                     (-1 if self.inverted else 1) * self.spacing[1] / math.sqrt(2) / (self.spacing[2] * math.sqrt(2)),
                     1,
                     0],
                    [0, 0, 0, 1]
                ])
        # result = result @ np.array([
        #     [1, 0, 0, self.shape[0] if self.flipped[0] else 0],
        #     [0, 1, 0, self.shape[1] if self.flipped[1] else 0],
        #     [0, 0, 1, self.shape[2] if self.flipped[2] else 0],
        #     [0, 0, 0, 1]
        # ])
        # result = result @ np.array([
        #     [-1 if self.flipped[0] else 1, 0, 0, 0],
        #     [0, -1 if self.flipped[1] else 1, 0, 0],
        #     [0, 0, -1 if self.flipped[2] else 1, 0],
        #     [0, 0, 0, 1]
        # ])
        return result

    def grid_to_world_2d(self, red_axis: int) -> np.ndarray:
        g2w = self.grid_to_world
        axes = np.ones((4,), dtype=np.bool)
        axes[red_axis] = False
        return g2w[axes][:, axes]

    def __setattr__(self, name, value):
        if hasattr(self, 'initialized'):
            """"""
            msg = "'%s' has no attribute %s" % (self.__class__,
                                                name)
            raise AttributeError(msg)
        else:
            np.ndarray.__setattr__(self, name, value)
