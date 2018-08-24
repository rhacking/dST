import numpy as np

from dispim import Volume


def test_spacing():
    vol_a = Volume(np.random.random((100, 100, 100)), False, (.5, .5, 1))
    vol_b = Volume(np.random.random((100, 100, 100)), True, (.5, .5, 1))
    print(vol_a.spacing)
    print(vol_a.world_transform)
    assert np.allclose(vol_a.spacing, (.5, .5, 1)) and np.allclose(vol_b.spacing, (.5, .5, 1))
