from typing import Tuple

import numpy as np
import pymrt.geometry
from scipy.ndimage.filters import convolve
from scipy.ndimage.filters import gaussian_filter

from dispim import Volume


def generate_bead_test_volume(name: str, size: Tuple[int, int, int] = (32, 32, 32),
                              num_beads: int = 32, sigmas: Tuple[float, float, float] = (0, 0, 2)):
    data = np.zeros(size)
    mask = np.array([True] * num_beads + [False] * (np.prod(data.size) - num_beads))
    np.random.shuffle(mask)
    mask = mask.reshape(data.shape)
    data[mask] = 1

    blurred = gaussian_filter(data, sigmas, mode='constant')
    blurred_vol = Volume(blurred, (1, 1, 1))
    blurred_vol.save_tiff('test_' + name)
    blurred_vol.save_tiff_single('test_single_' + name)


def generate_spheres_test_volume(name: str, size: int = 32,
                                 num_beads: int = 32, sigmas: Tuple[float, float, float] = (0, 0, 2)):
    data = np.zeros((size, size, size))
    for i in range(num_beads):
        p = np.random.randint(8, size - 8, 3)
        data[p[0] - 8:p[0] + 8, p[1] - 8:p[1] + 8, p[1] - 8:p[1] + 8] = pymrt.geometry.sphere(16, 0.5, 8)

    vol = Volume(data, (1, 1, 1))
    vol.save_tiff('test_' + name)
    vol.save_tiff_single('test_single_' + name)

    blurred = gaussian_filter(data, sigmas, mode='constant')
    blurred_vol = Volume(blurred, (1, 1, 1))
    blurred_vol.save_tiff('btest_' + name)
    blurred_vol.save_tiff_single('btest_single_' + name)
    return data


if __name__ == '__main__':
    # generate_bead_test_volume('beads_gen', size=(64, 64, 64), num_beads=40, sigmas=(0, 0, 1.3))
    vol = generate_spheres_test_volume('spheres_gen', size=75, num_beads=12, sigmas=(4, 4, 4))
    blur = lambda vol: gaussian_filter(vol, (4, 4, 4), mode='constant')

    # psf = np.zeros((17, 17, 17))
    # psf[8, 8, 8] = 1
    # psf = gaussian_filter(psf, (4, 4, 4), mode='constant')
    from scipy.stats import multivariate_normal

    psf = np.zeros((9, 9, 9))
    for x in range(9):
        for y in range(9):
            for z in range(9):
                psf[x, y, z] = multivariate_normal.pdf((x, y, z), mean=(4, 4, 4),
                                                       cov=np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]]))

    blurred = convolve(vol, psf)
    blurred += (np.random.poisson(lam=25, size=blurred.shape) - 10) / 250.

    blurred_vol = Volume(blurred, (1, 1, 1))
    blurred_vol.save_tiff('blurred_thing')

    # estimate = data
    # for i in range(200):
    #     estimate = estimate * blur(data/(blur(estimate)+1e-6))

    # estimate = restoration.richardson_lucy(blurred, psf, iterations=120)
    #
    # estimate_vol = Volume(estimate, (1, 1, 1))
    # estimate_vol.save_tiff('estimate')

    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    # from scipy.signal import convolve2d as conv2
    #
    # from skimage import color, data, restoration
    #
    # astro = color.rgb2gray(data.astronaut())
    #
    # psf = np.ones((5, 5)) / 25
    # astro_noisy = conv2(astro, psf, 'same')
    # # Add Noise to Image
    # # astro_noisy = astro.copy()
    # # astro_noisy += (np.random.poisson(lam=25, size=astro.shape) - 10) / 255.
    #
    # # Restore Image using Richardson-Lucy algorithm
    # deconvolved_RL = restoration.richardson_lucy(astro_noisy, psf, iterations=30)
    #
    # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5))
    # plt.gray()
    #
    # for a in (ax[0], ax[1], ax[2]):
    #     a.axis('off')
    #
    # ax[0].imshow(astro)
    # ax[0].set_title('Original Data')
    #
    # ax[1].imshow(astro_noisy)
    # ax[1].set_title('Noisy data')
    #
    # ax[2].imshow(deconvolved_RL, vmin=astro_noisy.min(), vmax=astro_noisy.max())
    # ax[2].set_title('Restoration using\nRichardson-Lucy')
    #
    # fig.subplots_adjust(wspace=0.02, hspace=0.2,
    #                     top=0.9, bottom=0.05, left=0, right=1)
    # plt.show()

    # astro = color.rgb2gray(data.astronaut())
    #
    #
    # psf = np.ones((10, 10)) / 100
    #
    #
    # astro_noisy = conv2(astro, psf, 'same')
    # # astro_noisy = astro.copy()
    # # Add Noise to Image
    # # astro_noisy += (np.random.poisson(lam=25, size=astro.shape) - 10) / 1024.
    #
    # # Restore Image using Richardson-Lucy algorithm
    # deconvolved_RL = restoration.richardson_lucy(astro_noisy, psf, iterations=30)
    # print(deconvolved_RL.shape)
    #
    # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5))
    # plt.gray()
    # for a in (ax[0], ax[1], ax[2]):
    #     a.axis('off')
    #
    # ax[0].imshow(astro)
    # ax[0].set_title('Original Data')
    #
    # ax[1].imshow(astro_noisy)
    # ax[1].set_title('Noisy data')
    #
    # ax[2].imshow(deconvolved_RL, vmin=astro_noisy.min(), vmax=astro_noisy.max())
    # ax[2].set_title('Restoration using\nRichardson-Lucy')
    #
    # fig.subplots_adjust(wspace=0.02, hspace=0.2,
    #                     top=0.9, bottom=0.05, left=0, right=1)
    # plt.show()
