import numpy as np
import dispim
from dispim import Volume

import pytest
from numba import njit


@pytest.fixture
def rand_vol_a():
    return Volume(np.random.randint(0, 2 ** 16 - 1, size=(64, 64, 64), dtype=np.uint16))


@pytest.fixture
@njit
def test_data():
    data = np.zeros((64, 64, 64))
    centers = np.random.randint(0, 64, size=(20, 3))

    for x in range(64):
        for y in range(64):
            for z in range(64):
                for i in range(20):
                    if np.linalg.norm(centers[i, :] - np.array([x, y, z], dtype=np.float32)) < 4:
                        data[x, y, z] = 1

    return data


@pytest.fixture
def fractal_noise():
    from perlin import generate_fractal_noise_3d
    np.random.seed(42)
    data = generate_fractal_noise_3d((64, 64, 64), (2, 2, 2), 4)
    data -= data.min()
    data *= 50
    data **= 12

    return data

def test_io():
    try:
        from dispim.io import save_tiff, load_tiff
        a = Volume((np.random.rand(64, 64, 64) * 2000).astype(np.uint16), spacing=(0.3, 0.3, 0.7))

        save_tiff(a, 'temp.tif')
        b: Volume = load_tiff('temp.tif', step_size=0.7)

        assert np.allclose(a, b)
        assert a.dtype == np.uint16
        assert b.dtype == np.uint16
        assert type(b) == Volume
        assert np.allclose(a.spacing, b.spacing)
    finally:
        import os
        if os.path.exists("temp.tif"):
            os.remove("temp.tif")


def test_mut():
    a = Volume((np.random.rand(64, 64, 64) * 2000).astype(np.uint16), spacing=(0.3, 0.3, 0.7))
    with pytest.raises(ValueError):
        a += (np.random.rand(64, 64, 64) * 2000).astype(np.uint16)


def test_volume_update(rand_vol_a: Volume):
    assert not rand_vol_a.inverted
    rand_vol_a = Volume(rand_vol_a, inverted=True)
    assert rand_vol_a.inverted
    rand_vol_a = Volume(rand_vol_a)
    assert rand_vol_a.inverted


def test_vol_defaults(rand_vol_a: Volume):
    assert rand_vol_a.spacing == (1, 1, 1)
    assert not rand_vol_a.inverted
    assert rand_vol_a.flipped == (False, False, False)
    assert not rand_vol_a.is_skewed
    assert np.allclose(rand_vol_a.world_transform, np.eye(4))

    rand_vol_b = rand_vol_a[:32]

    assert rand_vol_b.spacing == (1, 1, 1)
    assert not rand_vol_b.inverted
    assert rand_vol_b.flipped == (False, False, False)
    assert not rand_vol_b.is_skewed
    assert np.allclose(rand_vol_b.world_transform, np.eye(4))


@pytest.mark.parametrize("offset", [
    (8, 5, 0),
    (4, 5, 0),
    (0, 0, 0),
    (6, 0, 0),
    (4, 5, 0),
    (2, 6, 0)
])
def test_register2d(fractal_noise, offset):
    from scipy.ndimage import affine_transform
    a = Volume(fractal_noise[:32, :32, :32], is_skewed=False)
    b = Volume(fractal_noise[offset[0]:32 + offset[0], offset[1]:32 + offset[1], offset[2]:32 + offset[2]],
               is_skewed=False)

    assert np.allclose(a.grid_to_world, np.eye(4))
    assert np.allclose(b.grid_to_world, np.eye(4))

    dispim.register_2d(a, b, axis=2)

    b_transformed = affine_transform(b, b.world_transform, order=1)

    estimated_offset = -(b.world_transform[:3, 3])

    expected_mismatches = 32 * offset[0] + 32 * offset[1] - offset[0] * offset[1]
    expected_mismatches *= 32
    expected_mismatches *= 1.02
    expected_mismatches += 1500

    assert np.allclose(estimated_offset, offset, atol=0.2, rtol=0.05)


@pytest.mark.parametrize("offset", [
    (4, 5, 0),
    (0, 4, 4),
    (0, 0, 5),
    (3, 0, 3),
    (4, 5, 4),
    (5, 2, 3),
    (0, 0, 4)
])
def test_reigster_dipy(test_data, offset):
    a = Volume(test_data[:42, :42, :42], is_skewed=False)
    b = Volume(test_data[offset[0]:42 + offset[0], offset[1]:42 + offset[1], offset[2]:42 + offset[2]],
               is_skewed=False)

    assert np.allclose(a.grid_to_world, np.eye(4))
    assert np.allclose(b.grid_to_world, np.eye(4))

    dispim.register_com(a, b)
    dispim.register_dipy(a, b, crop=1.0)
    dispim.register_dipy(a, b, crop=1.0)

    estimated_offset = -(b.world_transform[:3, 3])
    assert np.allclose(estimated_offset, offset, atol=0.2, rtol=0.05)